"""Étape 7 : Serveur MCP (FastMCP) exposant l'entrepôt DuckDB en lecture seule avec transparence SQL."""

from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

# Assure l'import de `src.*` même si le script est lancé directement (`python src/mcp_server.py`)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import duckdb
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from src.config import WAREHOUSE_DB_PATH
from src.step6_warehouse import get_warehouse_connection, init_warehouse

mcp = FastMCP("FacturesWarehouseMCP", log_level="ERROR")

VALID_STATUSES = ("Payée", "En retard", "En attente")
VALID_CATEGORIES = ("Conseil", "Fournitures", "Informatique", "Maintenance")

GROUP_BY_WHITELIST: dict[str, tuple[str, str]] = {
    "status": ("status", "status"),
    "category": ("category", "category"),
    "supplier_name": ("supplier_name", "supplier_name"),
    "client_name": ("client_name", "client_name"),
    "year": ("EXTRACT(YEAR FROM issue_date)::INTEGER", "year"),
    "month": ("strftime(issue_date, '%Y-%m')", "month"),
}

GOLD_REPORTS: dict[str, tuple[str, str]] = {
    "fournisseurs": (
        "SELECT * FROM gold.kpi_fournisseurs ORDER BY total_ttc DESC",
        "KPI agrégés par fournisseur et catégorie",
    ),
    "mensuel": (
        "SELECT * FROM gold.kpi_mensuels ORDER BY mois ASC",
        "KPI mensuels par mois d'émission",
    ),
    "retards": (
        "SELECT invoice_number, issue_date, due_date, supplier_name, client_name, category, total_ht, total_tva, total_ttc, payment_method "
        "FROM gold.factures_en_retard ORDER BY total_ttc DESC",
        "Factures actuellement au statut 'En retard'",
    ),
    "quarantaine": (
        "SELECT doc_id, filename, extraction_method, invoice_number, supplier_name, client_name, "
        "reported_total_ht, calculated_lines_ht, reported_total_tva, reported_total_ttc, calculated_total_ttc, "
        "rejection_codes, rejection_reasons, quarantined_at "
        "FROM silver.quarantaine ORDER BY filename ASC",
        "Factures écartées en quarantaine lors des contrôles qualité",
    ),
}

VIEW_DESCRIPTIONS: dict[str, str] = {
    "bronze.documents": "Texte brut extrait (PyMuPDF natif ou LightOnOCR-3) et métadonnées techniques d'ingestion (SHA-256, temps ms).",
    "silver.factures": "Factures valides nettoyées, typées (DECIMAL(12,2), DATE) avec fournisseur, client, montants HT/TVA/TTC et qualité.",
    "silver.lignes_facture": "Lignes d'articles détaillées des factures Silver (désignation, quantité, prix unitaire HT, TVA %, total HT).",
    "silver.quarantaine": "Factures rejetées par les contrôles qualité avec codes et motifs détaillés (ex. incohérence TTC).",
    "gold.kpi_mensuels": "Indicateurs mensuels agrégés (nombre de factures, CA HT/TVA/TTC, montants payés/en retard/en attente).",
    "gold.kpi_fournisseurs": "Indicateurs agrégés par fournisseur (CA TTC, panier moyen, nombre et montant des factures en retard).",
    "gold.factures_en_retard": "Vue filtrée des factures au statut 'En retard', triées par montant TTC décroissant.",
    "main.factures": "Alias racine vers silver.factures.",
    "main.lignes_facture": "Alias racine vers silver.lignes_facture.",
    "main.quarantaine": "Alias racine vers silver.quarantaine.",
    "main.bronze_documents": "Alias racine vers bronze.documents.",
}

FORBIDDEN_SQL_KEYWORDS_RE = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|REPLACE|TRUNCATE|COPY|ATTACH|DETACH|INSTALL|LOAD|PRAGMA|EXPORT|IMPORT|CALL)\b",
    re.IGNORECASE,
)
ALLOWED_SQL_START_RE = re.compile(r"^(SELECT|WITH|DESCRIBE|SHOW|EXPLAIN)\b", re.IGNORECASE)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return str(obj)


def _format_response(
    *,
    status: str,
    tool: str,
    executed_sql: list[str],
    sql_params: list[list[Any]] | None = None,
    data: Any = None,
    error_message: str | None = None,
    hint: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "status": status,
        "tool": tool,
        "executed_sql": executed_sql,
        "sql_params": sql_params if sql_params is not None else [],
    }
    if status == "success":
        payload["data"] = data
    else:
        payload["error_message"] = error_message or "Erreur inconnue."
        if hint:
            payload["hint"] = hint
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)


def _rows_to_dicts(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    columns = [desc[0] for desc in (cursor.description or [])]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _ensure_warehouse_ready() -> None:
    if not WAREHOUSE_DB_PATH.exists():
        init_warehouse()


def _build_invoice_filters(
    *,
    supplier_name: str | None = None,
    client_name: str | None = None,
    status: str | None = None,
    category: str | None = None,
    year: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    min_ttc: float | None = None,
    max_ttc: float | None = None,
) -> tuple[str, list[Any], str | None]:
    """Construit une clause WHERE paramétrée (?) et valide les valeurs énumérées/dates."""
    if status is not None and status not in VALID_STATUSES:
        return (
            "",
            [],
            f"Statut invalide '{status}'. Valeurs autorisées : {', '.join(VALID_STATUSES)}.",
        )
    if category is not None and category not in VALID_CATEGORIES:
        return (
            "",
            [],
            f"Catégorie invalide '{category}'. Valeurs autorisées : {', '.join(VALID_CATEGORIES)}.",
        )

    for label, date_val in (("date_from", date_from), ("date_to", date_to)):
        if date_val is not None:
            try:
                date.fromisoformat(date_val)
            except ValueError:
                return "", [], f"Format de date invalide pour {label} ('{date_val}'). Format attendu : YYYY-MM-DD."

    clauses: list[str] = []
    params: list[Any] = []

    if supplier_name is not None and supplier_name.strip():
        clauses.append("supplier_name ILIKE '%' || ? || '%'")
        params.append(supplier_name.strip())
    if client_name is not None and client_name.strip():
        clauses.append("client_name ILIKE '%' || ? || '%'")
        params.append(client_name.strip())
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if category is not None:
        clauses.append("category = ?")
        params.append(category)
    if year is not None:
        clauses.append("EXTRACT(YEAR FROM issue_date) = ?")
        params.append(int(year))
    if date_from is not None:
        clauses.append("issue_date >= CAST(? AS DATE)")
        params.append(date_from)
    if date_to is not None:
        clauses.append("issue_date <= CAST(? AS DATE)")
        params.append(date_to)
    if min_ttc is not None:
        clauses.append("total_ttc >= CAST(? AS DECIMAL(12, 2))")
        params.append(str(min_ttc))
    if max_ttc is not None:
        clauses.append("total_ttc <= CAST(? AS DECIMAL(12, 2))")
        params.append(str(max_ttc))

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where_sql, params, None


def _strip_sql_comments_and_strings(sql: str) -> tuple[str, str]:
    """Retourne (sql_sans_commentaires, sql_sans_commentaires_ni_litteraux) pour la validation."""
    no_comments_chars: list[str] = []
    no_strings_chars: list[str] = []
    i = 0
    n = len(sql)
    in_single = False
    in_double = False

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        if not in_single and not in_double:
            if ch == "-" and nxt == "-":
                while i < n and sql[i] != "\n":
                    i += 1
                no_comments_chars.append(" ")
                no_strings_chars.append(" ")
                continue
            if ch == "/" and nxt == "*":
                i += 2
                while i + 1 < n and not (sql[i] == "*" and sql[i + 1] == "/"):
                    i += 1
                i += 2
                no_comments_chars.append(" ")
                no_strings_chars.append(" ")
                continue
            if ch == "'":
                in_single = True
                no_comments_chars.append(ch)
                no_strings_chars.append(" ")
                i += 1
                continue
            if ch == '"':
                in_double = True
                no_comments_chars.append(ch)
                no_strings_chars.append(ch)
                i += 1
                continue
            no_comments_chars.append(ch)
            no_strings_chars.append(ch)
            i += 1
        elif in_single:
            no_comments_chars.append(ch)
            if ch == "'" and nxt == "'":
                no_comments_chars.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_single = False
            no_strings_chars.append(" ")
            i += 1
        else:  # in_double (identifiant)
            no_comments_chars.append(ch)
            no_strings_chars.append(ch)
            if ch == '"' and nxt == '"':
                no_comments_chars.append(nxt)
                no_strings_chars.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1

    return "".join(no_comments_chars).strip(), "".join(no_strings_chars).strip()


def _validate_readonly_sql(raw_sql: str) -> tuple[str | None, str | None]:
    """Valide qu'une requête SQL libre est strictement en lecture seule et mono-instruction."""
    if not raw_sql or not raw_sql.strip():
        return None, "La requête SQL fournie est vide."

    cleaned, stripped_literals = _strip_sql_comments_and_strings(raw_sql)
    cleaned_no_trailing = cleaned.rstrip(";").strip()
    stripped_no_trailing = stripped_literals.rstrip(";").strip()

    if not cleaned_no_trailing:
        return None, "La requête SQL ne contient aucune instruction exécutable."

    if ";" in stripped_no_trailing:
        return (
            None,
            "Sécurité : plusieurs instructions SQL séparées par ';' sont interdites.",
        )

    if not ALLOWED_SQL_START_RE.match(stripped_no_trailing):
        return (
            None,
            "Sécurité : seules les requêtes de lecture commençant par SELECT, WITH, DESCRIBE, SHOW ou EXPLAIN sont autorisées.",
        )

    forbidden_match = FORBIDDEN_SQL_KEYWORDS_RE.search(stripped_no_trailing)
    if forbidden_match:
        kw = forbidden_match.group(1).upper()
        return (
            None,
            f"Sécurité : mot-clé interdit '{kw}' détecté. L'entrepôt est strictement en lecture seule.",
        )

    return cleaned_no_trailing, None


@mcp.tool(
    name="list_invoices",
    description=(
        "Liste les factures valides de silver.factures avec filtres optionnels combinables "
        "(fournisseur, client, statut, catégorie, année, période, bornes TTC). "
        "À utiliser pour obtenir une liste paginée de factures (max 100). "
        "Pour des totaux agrégés ou un comptage global sans limite de pagination, préférez get_invoices_summary."
    ),
)
def list_invoices(
    supplier_name: str | None = Field(
        default=None,
        description="Filtre partiel insensible à la casse sur le nom du fournisseur (ex: 'Aubry', 'Bernard SARL', 'Maury').",
    ),
    client_name: str | None = Field(
        default=None,
        description="Filtre partiel insensible à la casse sur le nom du client (ex: 'Lombard', 'Dumas').",
    ),
    status: str | None = Field(
        default=None,
        description="Statut exact de paiement. Valeurs possibles : 'Payée', 'En retard', 'En attente'.",
    ),
    category: str | None = Field(
        default=None,
        description="Catégorie exacte de la facture. Valeurs possibles : 'Conseil', 'Fournitures', 'Informatique', 'Maintenance'.",
    ),
    year: int | None = Field(
        default=None,
        description="Année d'émission exacte (ex: 2025 ou 2026).",
    ),
    date_from: str | None = Field(
        default=None,
        description="Date d'émission minimale incluse au format YYYY-MM-DD (ex: '2025-01-01').",
    ),
    date_to: str | None = Field(
        default=None,
        description="Date d'émission maximale incluse au format YYYY-MM-DD (ex: '2025-12-31').",
    ),
    min_ttc: float | None = Field(
        default=None,
        description="Montant TTC minimum en euros (ex: 1000.0).",
    ),
    max_ttc: float | None = Field(
        default=None,
        description="Montant TTC maximum en euros (ex: 10000.0).",
    ),
    limit: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Nombre maximum de factures retournées (défaut 50, maximum 100).",
    ),
) -> str:
    where_sql, params, val_err = _build_invoice_filters(
        supplier_name=supplier_name,
        client_name=client_name,
        status=status,
        category=category,
        year=year,
        date_from=date_from,
        date_to=date_to,
        min_ttc=min_ttc,
        max_ttc=max_ttc,
    )
    if val_err:
        return _format_response(
            status="error",
            tool="list_invoices",
            executed_sql=[],
            error_message=val_err,
            hint="Appelez get_schema pour vérifier les valeurs exactes des colonnes status et category.",
        )

    bounded_limit = max(1, min(int(limit), 100))
    sql = (
        "SELECT invoice_number, issue_date, due_date, category, status, "
        "supplier_name, client_name, total_ht, total_tva, total_ttc, "
        "line_count, payment_method, quality_status, filename "
        f"FROM silver.factures {where_sql} "
        "ORDER BY issue_date DESC, invoice_number ASC "
        "LIMIT ?"
    )
    query_params = [*params, bounded_limit]

    try:
        _ensure_warehouse_ready()
        with get_warehouse_connection(read_only=True) as conn:
            rows = _rows_to_dicts(conn.execute(sql, query_params))
        return _format_response(
            status="success",
            tool="list_invoices",
            executed_sql=[sql],
            sql_params=[query_params],
            data={"count": len(rows), "limit": bounded_limit, "invoices": rows},
        )
    except duckdb.Error as exc:
        return _format_response(
            status="error",
            tool="list_invoices",
            executed_sql=[sql],
            sql_params=[query_params],
            error_message=str(exc),
            hint="Vérifiez les paramètres ou consultez le schéma via get_schema.",
        )


@mcp.tool(
    name="get_invoices_summary",
    description=(
        "Calcule le nombre de factures (COUNT) et les montants agrégés (total HT, TVA, TTC, "
        "panier moyen TTC, min et max TTC) sur silver.factures avec filtres optionnels et "
        "regroupement optionnel (group_by). Outil prioritaire pour toute question quantitative "
        "('Combien...', 'Quel est le montant total...')."
    ),
)
def get_invoices_summary(
    supplier_name: str | None = Field(
        default=None,
        description="Filtre partiel insensible à la casse sur le nom du fournisseur.",
    ),
    client_name: str | None = Field(
        default=None,
        description="Filtre partiel insensible à la casse sur le nom du client.",
    ),
    status: str | None = Field(
        default=None,
        description="Statut exact de paiement. Valeurs possibles : 'Payée', 'En retard', 'En attente'.",
    ),
    category: str | None = Field(
        default=None,
        description="Catégorie exacte. Valeurs possibles : 'Conseil', 'Fournitures', 'Informatique', 'Maintenance'.",
    ),
    year: int | None = Field(
        default=None,
        description="Année d'émission exacte (ex: 2025 ou 2026).",
    ),
    date_from: str | None = Field(
        default=None,
        description="Date d'émission minimale incluse (YYYY-MM-DD).",
    ),
    date_to: str | None = Field(
        default=None,
        description="Date d'émission maximale incluse (YYYY-MM-DD).",
    ),
    group_by: str | None = Field(
        default=None,
        description=(
            "Dimension de regroupement optionnelle parmi : "
            "'status', 'category', 'supplier_name', 'client_name', 'year', 'month'."
        ),
    ),
) -> str:
    where_sql, params, val_err = _build_invoice_filters(
        supplier_name=supplier_name,
        client_name=client_name,
        status=status,
        category=category,
        year=year,
        date_from=date_from,
        date_to=date_to,
    )
    if val_err:
        return _format_response(
            status="error",
            tool="get_invoices_summary",
            executed_sql=[],
            error_message=val_err,
            hint="Valeurs valides pour status: 'Payée', 'En retard', 'En attente'.",
        )

    if group_by is not None and group_by not in GROUP_BY_WHITELIST:
        return _format_response(
            status="error",
            tool="get_invoices_summary",
            executed_sql=[],
            error_message=f"Paramètre group_by invalide : '{group_by}'.",
            hint=f"Valeurs autorisées pour group_by : {', '.join(GROUP_BY_WHITELIST.keys())}.",
        )

    agg_select = (
        "COUNT(*)::INTEGER AS invoice_count, "
        "COALESCE(SUM(total_ht), 0::DECIMAL(12, 2)) AS total_ht, "
        "COALESCE(SUM(total_tva), 0::DECIMAL(12, 2)) AS total_tva, "
        "COALESCE(SUM(total_ttc), 0::DECIMAL(12, 2)) AS total_ttc, "
        "COALESCE(ROUND(AVG(total_ttc), 2), 0::DECIMAL(12, 2)) AS avg_ttc, "
        "MIN(total_ttc) AS min_ttc, "
        "MAX(total_ttc) AS max_ttc"
    )

    try:
        _ensure_warehouse_ready()
        with get_warehouse_connection(read_only=True) as conn:
            overall_sql = f"SELECT {agg_select} FROM silver.factures {where_sql}"
            overall_row = _rows_to_dicts(conn.execute(overall_sql, params))[0]

            if group_by is None:
                return _format_response(
                    status="success",
                    tool="get_invoices_summary",
                    executed_sql=[overall_sql],
                    sql_params=[params],
                    data=overall_row,
                )

            expr, alias = GROUP_BY_WHITELIST[group_by]
            grouped_sql = (
                f"SELECT {expr} AS {alias}, {agg_select} "
                f"FROM silver.factures {where_sql} "
                f"GROUP BY {alias} "
                "ORDER BY total_ttc DESC"
            )
            group_rows = _rows_to_dicts(conn.execute(grouped_sql, params))
            return _format_response(
                status="success",
                tool="get_invoices_summary",
                executed_sql=[overall_sql, grouped_sql],
                sql_params=[params, params],
                data={
                    "group_by": group_by,
                    "overall": overall_row,
                    "groups": group_rows,
                },
            )
    except duckdb.Error as exc:
        return _format_response(
            status="error",
            tool="get_invoices_summary",
            executed_sql=[],
            error_message=str(exc),
            hint="Consultez get_schema pour vérifier les colonnes disponibles.",
        )


@mcp.tool(
    name="get_invoice_details",
    description=(
        "Récupère le détail complet d'une facture précise par son numéro (en-tête Silver, "
        "lignes d'articles détaillées, métadonnées de traçabilité Bronze), et recherche "
        "automatiquement dans silver.quarantaine si la facture a été rejetée lors des "
        "contrôles qualité (ex: 'V-2025-0001')."
    ),
)
def get_invoice_details(
    invoice_number: str = Field(
        ...,
        description="Numéro exact de la facture recherchée (ex: 'A-2025-0001' ou 'V-2025-0001').",
    ),
) -> str:
    inv_num = invoice_number.strip()
    if not inv_num:
        return _format_response(
            status="error",
            tool="get_invoice_details",
            executed_sql=[],
            error_message="Le numéro de facture (invoice_number) ne peut pas être vide.",
            hint="Fournissez un numéro de facture valide, ex: 'A-2025-0001'.",
        )

    sql_invoice = "SELECT * FROM silver.factures WHERE invoice_number = ?"
    sql_lines = (
        "SELECT line_id, line_number, description, quantity, unit_price_ht, "
        "tva_rate_pct, line_total_ht "
        "FROM silver.lignes_facture WHERE invoice_number = ? ORDER BY line_number ASC"
    )
    sql_quarantine = "SELECT * FROM silver.quarantaine WHERE invoice_number = ?"
    sql_bronze = (
        "SELECT doc_id, filename, source_uri, file_size_bytes, page_count, pdf_type, "
        "extraction_method, ocr_model, ocr_longest_edge, char_count, processing_time_ms, ingested_at "
        "FROM bronze.documents WHERE doc_id = ?"
    )

    executed_sql: list[str] = []
    sql_params: list[list[Any]] = []

    try:
        _ensure_warehouse_ready()
        with get_warehouse_connection(read_only=True) as conn:
            executed_sql.append(sql_invoice)
            sql_params.append([inv_num])
            invoice_rows = _rows_to_dicts(conn.execute(sql_invoice, [inv_num]))

            if invoice_rows:
                invoice = invoice_rows[0]
                executed_sql.append(sql_lines)
                sql_params.append([inv_num])
                lines = _rows_to_dicts(conn.execute(sql_lines, [inv_num]))

                executed_sql.append(sql_bronze)
                sql_params.append([invoice["doc_id"]])
                bronze_rows = _rows_to_dicts(conn.execute(sql_bronze, [invoice["doc_id"]]))

                return _format_response(
                    status="success",
                    tool="get_invoice_details",
                    executed_sql=executed_sql,
                    sql_params=sql_params,
                    data={
                        "found_in": "silver.factures",
                        "invoice": invoice,
                        "lines": lines,
                        "quarantine": None,
                        "bronze_lineage": bronze_rows[0] if bronze_rows else None,
                    },
                )

            # Sinon vérifier en quarantaine
            executed_sql.append(sql_quarantine)
            sql_params.append([inv_num])
            quarantine_rows = _rows_to_dicts(conn.execute(sql_quarantine, [inv_num]))

            if quarantine_rows:
                q_row = quarantine_rows[0]
                executed_sql.append(sql_bronze)
                sql_params.append([q_row["doc_id"]])
                bronze_rows = _rows_to_dicts(conn.execute(sql_bronze, [q_row["doc_id"]]))

                return _format_response(
                    status="success",
                    tool="get_invoice_details",
                    executed_sql=executed_sql,
                    sql_params=sql_params,
                    data={
                        "found_in": "silver.quarantaine",
                        "invoice": None,
                        "lines": [],
                        "quarantine": q_row,
                        "bronze_lineage": bronze_rows[0] if bronze_rows else None,
                        "message": (
                            f"La facture '{inv_num}' n'est pas dans silver.factures car elle a été "
                            f"rejetée en quarantaine ({q_row['rejection_codes']})."
                        ),
                    },
                )

            return _format_response(
                status="error",
                tool="get_invoice_details",
                executed_sql=executed_sql,
                sql_params=sql_params,
                error_message=f"Aucune facture trouvée avec le numéro '{inv_num}' ni en Silver ni en quarantaine.",
                hint="Utilisez list_invoices pour rechercher les numéros de facture disponibles ou get_gold_kpis(report_type='quarantaine').",
            )
    except duckdb.Error as exc:
        return _format_response(
            status="error",
            tool="get_invoice_details",
            executed_sql=executed_sql,
            sql_params=sql_params,
            error_message=str(exc),
            hint="Vérifiez l'existence de l'entrepôt avec get_schema.",
        )


@mcp.tool(
    name="get_schema",
    description=(
        "Découvre dynamiquement le schéma complet de l'entrepôt DuckDB (schémas bronze, silver, "
        "gold et main) : liste des vues, colonnes, types SQL, descriptions métier et valeurs "
        "distinctes des colonnes clés (status, category, supplier_name). Indispensable avant "
        "d'écrire une requête SQL libre via execute_readonly_sql."
    ),
)
def get_schema() -> str:
    sql_views = (
        "SELECT table_schema, table_name, table_type "
        "FROM information_schema.tables "
        "WHERE table_schema IN ('bronze', 'silver', 'gold', 'main') "
        "ORDER BY table_schema, table_name"
    )
    sql_cols = (
        "SELECT table_schema, table_name, column_name, data_type, is_nullable "
        "FROM information_schema.columns "
        "WHERE table_schema IN ('bronze', 'silver', 'gold', 'main') "
        "ORDER BY table_schema, table_name, ordinal_position"
    )
    sql_statuses = "SELECT DISTINCT status FROM silver.factures ORDER BY status"
    sql_categories = "SELECT DISTINCT category FROM silver.factures ORDER BY category"
    sql_suppliers = "SELECT DISTINCT supplier_name FROM silver.factures ORDER BY supplier_name"

    executed_sql = [sql_views, sql_cols, sql_statuses, sql_categories, sql_suppliers]

    try:
        _ensure_warehouse_ready()
        with get_warehouse_connection(read_only=True) as conn:
            views = _rows_to_dicts(conn.execute(sql_views))
            cols = _rows_to_dicts(conn.execute(sql_cols))
            statuses = [r[0] for r in conn.execute(sql_statuses).fetchall()]
            categories = [r[0] for r in conn.execute(sql_categories).fetchall()]
            suppliers = [r[0] for r in conn.execute(sql_suppliers).fetchall()]

        cols_by_view: dict[str, list[dict[str, Any]]] = {}
        for col in cols:
            full_name = f"{col['table_schema']}.{col['table_name']}"
            cols_by_view.setdefault(full_name, []).append(
                {
                    "column_name": col["column_name"],
                    "data_type": col["data_type"],
                    "is_nullable": col["is_nullable"],
                }
            )

        schema_catalog: list[dict[str, Any]] = []
        for view in views:
            full_name = f"{view['table_schema']}.{view['table_name']}"
            schema_catalog.append(
                {
                    "schema": view["table_schema"],
                    "name": view["table_name"],
                    "full_name": full_name,
                    "type": view["table_type"],
                    "description": VIEW_DESCRIPTIONS.get(full_name, ""),
                    "columns": cols_by_view.get(full_name, []),
                }
            )

        return _format_response(
            status="success",
            tool="get_schema",
            executed_sql=executed_sql,
            sql_params=[[], [], [], [], []],
            data={
                "views": schema_catalog,
                "distinct_values": {
                    "status": statuses,
                    "category": categories,
                    "supplier_name": suppliers,
                },
            },
        )
    except duckdb.Error as exc:
        return _format_response(
            status="error",
            tool="get_schema",
            executed_sql=executed_sql,
            error_message=str(exc),
            hint="Assurez-vous que l'étape 6 (init_warehouse) a été exécutée.",
        )


@mcp.tool(
    name="execute_readonly_sql",
    description=(
        "Exécute une requête SQL DuckDB libre STRICTEMENT EN LECTURE SEULE (SELECT, WITH, "
        "DESCRIBE, SHOW, EXPLAIN). À n'utiliser que pour des analyses avancées non couvertes "
        "par list_invoices, get_invoices_summary, get_invoice_details ou get_gold_kpis. "
        "Toute tentative d'écriture (DROP, DELETE, INSERT, UPDATE, ALTER, CREATE...) est bloquée."
    ),
)
def execute_readonly_sql(
    sql: str = Field(
        ...,
        description=(
            "Requête SQL DuckDB en lecture seule (SELECT / WITH / DESCRIBE / SHOW / EXPLAIN) "
            "portant sur les vues silver.factures, silver.lignes_facture, silver.quarantaine, "
            "bronze.documents, gold.kpi_mensuels, gold.kpi_fournisseurs ou gold.factures_en_retard."
        ),
    ),
) -> str:
    validated_sql, sec_error = _validate_readonly_sql(sql)
    if sec_error or validated_sql is None:
        return _format_response(
            status="error",
            tool="execute_readonly_sql",
            executed_sql=[sql],
            error_message=sec_error or "Requête SQL invalide.",
            hint=(
                "Seules les requêtes SELECT/WITH/DESCRIBE/SHOW/EXPLAIN mono-instruction en lecture "
                "seule sont autorisées. Vues principales : silver.factures, silver.lignes_facture, "
                "silver.quarantaine, bronze.documents, gold.kpi_mensuels, gold.kpi_fournisseurs, "
                "gold.factures_en_retard. Appelez get_schema pour voir leurs colonnes exactes."
            ),
        )

    max_rows = 200
    try:
        _ensure_warehouse_ready()
        with get_warehouse_connection(read_only=True) as conn:
            cursor = conn.execute(validated_sql)
            columns = [desc[0] for desc in (cursor.description or [])]
            fetched = cursor.fetchmany(max_rows + 1)
            truncated = len(fetched) > max_rows
            rows = [dict(zip(columns, r)) for r in fetched[:max_rows]]

        return _format_response(
            status="success",
            tool="execute_readonly_sql",
            executed_sql=[validated_sql],
            sql_params=[[]],
            data={
                "columns": columns,
                "row_count": len(rows),
                "truncated": truncated,
                "rows": rows,
            },
        )
    except duckdb.Error as exc:
        return _format_response(
            status="error",
            tool="execute_readonly_sql",
            executed_sql=[validated_sql],
            sql_params=[[]],
            error_message=str(exc),
            hint=(
                "Erreur SQL DuckDB. Appelez get_schema pour vérifier les noms exacts des colonnes "
                "dans silver.factures, silver.lignes_facture, silver.quarantaine, bronze.documents, "
                "gold.kpi_mensuels, gold.kpi_fournisseurs et gold.factures_en_retard."
            ),
        )


@mcp.tool(
    name="get_gold_kpis",
    description=(
        "Interroge les vues métier pré-agrégées de la couche Gold (ou la quarantaine Silver) : "
        "KPI par fournisseur, KPI mensuels, liste des factures en retard, ou factures en quarantaine."
    ),
)
def get_gold_kpis(
    report_type: str = Field(
        default="fournisseurs",
        description=(
            "Type de rapport souhaité. Valeurs exactes : "
            "'fournisseurs' (gold.kpi_fournisseurs), "
            "'mensuel' (gold.kpi_mensuels), "
            "'retards' (gold.factures_en_retard), "
            "'quarantaine' (silver.quarantaine)."
        ),
    ),
) -> str:
    key = report_type.strip().lower()
    if key not in GOLD_REPORTS:
        return _format_response(
            status="error",
            tool="get_gold_kpis",
            executed_sql=[],
            error_message=f"Type de rapport inconnu : '{report_type}'.",
            hint=f"Valeurs autorisées pour report_type : {', '.join(GOLD_REPORTS.keys())}.",
        )

    sql, description = GOLD_REPORTS[key]
    try:
        _ensure_warehouse_ready()
        with get_warehouse_connection(read_only=True) as conn:
            rows = _rows_to_dicts(conn.execute(sql))
        return _format_response(
            status="success",
            tool="get_gold_kpis",
            executed_sql=[sql],
            sql_params=[[]],
            data={
                "report_type": key,
                "description": description,
                "count": len(rows),
                "rows": rows,
            },
        )
    except duckdb.Error as exc:
        return _format_response(
            status="error",
            tool="get_gold_kpis",
            executed_sql=[sql],
            error_message=str(exc),
            hint="Vérifiez les vues Gold avec get_schema.",
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
