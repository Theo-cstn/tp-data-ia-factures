"""Étape 6 : Entrepôt de données DuckDB (séparation stockage Parquet / calcul SQL, portabilité et concurrence)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import (
    BASE_DIR,
    BRONZE_PARQUET_PATH,
    SILVER_INVOICES_PATH,
    SILVER_LINES_PATH,
    SILVER_QUARANTINE_PATH,
    WAREHOUSE_DB_PATH,
)

console = Console()

VIEW_CATALOG_SPEC: list[tuple[str, str, str]] = [
    ("bronze", "documents", "data/bronze/documents.parquet"),
    ("silver", "factures", "data/silver/factures.parquet"),
    ("silver", "lignes_facture", "data/silver/lignes_facture.parquet"),
    ("silver", "quarantaine", "data/silver/quarantaine.parquet"),
    ("main", "factures", "alias -> silver.factures"),
    ("main", "lignes_facture", "alias -> silver.lignes_facture"),
    ("main", "quarantaine", "alias -> silver.quarantaine"),
    ("main", "bronze_documents", "alias -> bronze.documents"),
    ("gold", "kpi_mensuels", "vue analytique sur silver.factures"),
    ("gold", "kpi_fournisseurs", "vue analytique sur silver.factures"),
    ("gold", "factures_en_retard", "vue analytique sur silver.factures"),
]


@dataclass
class ConcurrencyTestResult:
    """Résultat du test d'accès concurrent multi-processus en lecture seule."""

    success: bool
    elapsed_ms: float
    worker_a_output: str
    worker_b_output: str


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _validate_source_parquets(base_dir: Path = BASE_DIR) -> None:
    """Vérifie que les fichiers Parquet Bronze et Silver requis existent avant d'initialiser les vues."""
    required_paths = [
        BRONZE_PARQUET_PATH if base_dir == BASE_DIR else base_dir / "data/bronze/documents.parquet",
        SILVER_INVOICES_PATH if base_dir == BASE_DIR else base_dir / "data/silver/factures.parquet",
        SILVER_LINES_PATH if base_dir == BASE_DIR else base_dir / "data/silver/lignes_facture.parquet",
        SILVER_QUARANTINE_PATH if base_dir == BASE_DIR else base_dir / "data/silver/quarantaine.parquet",
    ]
    missing = [p for p in required_paths if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Fichier(s) Parquet manquant(s) : {missing_str}. "
            "Exécutez d'abord les Étapes 3 et 4."
        )


def init_warehouse(
    db_path: Path = WAREHOUSE_DB_PATH,
    base_dir: Path = BASE_DIR,
) -> float:
    """Initialise les schémas et vues SQL dans warehouse.duckdb sans aucune table physique, puis ferme la connexion."""
    _validate_source_parquets(base_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    conn = duckdb.connect(str(db_path), read_only=False)
    try:
        search_path_sql = _escape_sql_literal(str(base_dir))
        conn.execute(f"SET file_search_path = '{search_path_sql}'")

        # Supprimer d'éventuelles tables physiques résiduelles pour garantir 0 duplication
        physical_tables = conn.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
            """
        ).fetchall()
        for schema_name, table_name in physical_tables:
            conn.execute(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"')

        conn.execute("CREATE SCHEMA IF NOT EXISTS bronze")
        conn.execute("CREATE SCHEMA IF NOT EXISTS silver")
        conn.execute("CREATE SCHEMA IF NOT EXISTS gold")

        # Vues Bronze & Silver sur chemins RELATIFS à la racine du projet
        conn.execute(
            """
            CREATE OR REPLACE VIEW bronze.documents AS
            SELECT * FROM read_parquet('data/bronze/documents.parquet')
            """
        )
        conn.execute(
            """
            CREATE OR REPLACE VIEW silver.factures AS
            SELECT * FROM read_parquet('data/silver/factures.parquet')
            """
        )
        conn.execute(
            """
            CREATE OR REPLACE VIEW silver.lignes_facture AS
            SELECT * FROM read_parquet('data/silver/lignes_facture.parquet')
            """
        )
        conn.execute(
            """
            CREATE OR REPLACE VIEW silver.quarantaine AS
            SELECT * FROM read_parquet('data/silver/quarantaine.parquet')
            """
        )

        # Alias pratiques dans le schéma main
        conn.execute("CREATE OR REPLACE VIEW main.factures AS SELECT * FROM silver.factures")
        conn.execute(
            "CREATE OR REPLACE VIEW main.lignes_facture AS SELECT * FROM silver.lignes_facture"
        )
        conn.execute("CREATE OR REPLACE VIEW main.quarantaine AS SELECT * FROM silver.quarantaine")
        conn.execute(
            "CREATE OR REPLACE VIEW main.bronze_documents AS SELECT * FROM bronze.documents"
        )

        # Bonus Couche Gold : vues analytiques métier
        conn.execute(
            """
            CREATE OR REPLACE VIEW gold.kpi_mensuels AS
            SELECT
                strftime(issue_date, '%Y-%m') AS mois,
                COUNT(*)::INTEGER AS nb_factures,
                COALESCE(SUM(total_ht), 0::DECIMAL(12, 2)) AS ca_ht,
                COALESCE(SUM(total_tva), 0::DECIMAL(12, 2)) AS total_tva,
                COALESCE(SUM(total_ttc), 0::DECIMAL(12, 2)) AS ca_ttc,
                COALESCE(SUM(total_ttc) FILTER (WHERE status = 'Payée'), 0::DECIMAL(12, 2)) AS montant_paye,
                COALESCE(SUM(total_ttc) FILTER (WHERE status = 'En retard'), 0::DECIMAL(12, 2)) AS montant_en_retard,
                COALESCE(SUM(total_ttc) FILTER (WHERE status = 'En attente'), 0::DECIMAL(12, 2)) AS montant_en_attente
            FROM silver.factures
            GROUP BY mois
            ORDER BY mois
            """
        )
        conn.execute(
            """
            CREATE OR REPLACE VIEW gold.kpi_fournisseurs AS
            SELECT
                supplier_name,
                supplier_siret,
                category,
                COUNT(*)::INTEGER AS nb_factures,
                COALESCE(SUM(total_ht), 0::DECIMAL(12, 2)) AS total_ht,
                COALESCE(SUM(total_tva), 0::DECIMAL(12, 2)) AS total_tva,
                COALESCE(SUM(total_ttc), 0::DECIMAL(12, 2)) AS total_ttc,
                ROUND(AVG(total_ttc), 2)::DECIMAL(12, 2) AS panier_moyen_ttc,
                COUNT(*) FILTER (WHERE status = 'En retard')::INTEGER AS nb_en_retard,
                COALESCE(SUM(total_ttc) FILTER (WHERE status = 'En retard'), 0::DECIMAL(12, 2)) AS montant_en_retard_ttc
            FROM silver.factures
            GROUP BY supplier_name, supplier_siret, category
            ORDER BY total_ttc DESC
            """
        )
        conn.execute(
            """
            CREATE OR REPLACE VIEW gold.factures_en_retard AS
            SELECT *
            FROM silver.factures
            WHERE status = 'En retard'
            ORDER BY total_ttc DESC
            """
        )
    finally:
        conn.close()

    return (time.perf_counter() - start) * 1000.0


def ensure_warehouse_views(
    db_path: Path = WAREHOUSE_DB_PATH,
    base_dir: Path = BASE_DIR,
) -> None:
    """Garantit l'existence et la fraîcheur des vues dans warehouse.duckdb."""
    init_warehouse(db_path=db_path, base_dir=base_dir)


def get_warehouse_connection(
    read_only: bool = True,
    db_path: Path = WAREHOUSE_DB_PATH,
    base_dir: Path = BASE_DIR,
) -> duckdb.DuckDBPyConnection:
    """Ouvre une connexion DuckDB sur warehouse.duckdb (read_only=True par défaut) avec file_search_path sur la racine."""
    if not db_path.exists():
        init_warehouse(db_path=db_path, base_dir=base_dir)

    conn = duckdb.connect(str(db_path), read_only=read_only)
    search_path_sql = _escape_sql_literal(str(base_dir))
    conn.execute(f"SET file_search_path = '{search_path_sql}'")
    return conn


def run_multiprocess_concurrency_test(
    db_path: Path = WAREHOUSE_DB_PATH,
    base_dir: Path = BASE_DIR,
) -> ConcurrencyTestResult:
    """Lance 2 processus Python distincts interrogeant simultanément warehouse.duckdb en read_only=True."""
    worker_a_code = f"""
import json, time
from pathlib import Path
from src.step6_warehouse import get_warehouse_connection
with get_warehouse_connection(read_only=True, db_path=Path({str(db_path)!r}), base_dir=Path({str(base_dir)!r})) as conn:
    n_fact = conn.execute("SELECT COUNT(*) FROM silver.factures").fetchone()[0]
    time.sleep(0.35)
    late_ttc = conn.execute("SELECT CAST(SUM(total_ttc) AS VARCHAR) FROM gold.factures_en_retard").fetchone()[0]
print(json.dumps({{"worker": "Processus A (simulation DBeaver)", "factures": n_fact, "late_ttc": late_ttc}}))
"""
    worker_b_code = f"""
import json, time
from pathlib import Path
from src.step6_warehouse import get_warehouse_connection
with get_warehouse_connection(read_only=True, db_path=Path({str(db_path)!r}), base_dir=Path({str(base_dir)!r})) as conn:
    time.sleep(0.15)
    n_bronze = conn.execute("SELECT COUNT(*) FROM bronze.documents").fetchone()[0]
    n_months = conn.execute("SELECT COUNT(*) FROM gold.kpi_mensuels").fetchone()[0]
print(json.dumps({{"worker": "Processus B (simulation MCP/Agent)", "bronze": n_bronze, "mois_gold": n_months}}))
"""
    start = time.perf_counter()
    proc_a = subprocess.Popen(
        [sys.executable, "-c", worker_a_code],
        cwd=str(base_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    proc_b = subprocess.Popen(
        [sys.executable, "-c", worker_b_code],
        cwd=str(base_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out_a, err_a = proc_a.communicate(timeout=15)
    out_b, err_b = proc_b.communicate(timeout=15)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    success = proc_a.returncode == 0 and proc_b.returncode == 0
    return ConcurrencyTestResult(
        success=success,
        elapsed_ms=elapsed_ms,
        worker_a_output=out_a.strip() if proc_a.returncode == 0 else err_a.strip(),
        worker_b_output=out_b.strip() if proc_b.returncode == 0 else err_b.strip(),
    )


def display_warehouse_report(init_ms: float, concurrency: ConcurrencyTestResult) -> None:
    """Affiche le catalogue des vues, des requêtes de démonstration et la config DBeaver."""
    with get_warehouse_connection(read_only=True) as conn:
        # Vérification stricte 0 table physique
        phys_count = conn.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
            """
        ).fetchone()[0]

        # 1. Catalogue des vues
        cat_table = Table(
            title=f"Catalogue Entrepôt DuckDB (init en {init_ms:.1f} ms — {phys_count} table physique)",
            show_header=True,
            header_style="bold cyan",
        )
        cat_table.add_column("Schéma", style="magenta", no_wrap=True)
        cat_table.add_column("Vue SQL", style="white", no_wrap=True)
        cat_table.add_column("Source / Définition", style="dim")
        cat_table.add_column("Lignes", justify="right", style="bold green")

        for schema, view_name, source_desc in VIEW_CATALOG_SPEC:
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{schema}"."{view_name}"').fetchone()[0]
            cat_table.add_row(schema, view_name, source_desc, str(row_count))

        console.print(cat_table)

        # 2. Démonstration : répartition par statut (2025 vs 2026)
        status_rows = conn.execute(
            """
            SELECT
                strftime(issue_date, '%Y') AS annee,
                status,
                COUNT(*) AS nb_factures,
                SUM(total_ht) AS total_ht,
                SUM(total_ttc) AS total_ttc
            FROM silver.factures
            GROUP BY annee, status
            ORDER BY annee, total_ttc DESC
            """
        ).fetchall()

        status_table = Table(
            title="Démo SQL (Silver) — Répartition par année et statut",
            show_header=True,
            header_style="bold cyan",
        )
        status_table.add_column("Année", justify="center")
        status_table.add_column("Statut", style="white")
        status_table.add_column("Factures", justify="right")
        status_table.add_column("Total HT (€)", justify="right")
        status_table.add_column("Total TTC (€)", justify="right", style="bold")
        for annee, status, nb, ht, ttc in status_rows:
            status_table.add_row(str(annee), str(status), str(nb), f"{ht:,.2f}", f"{ttc:,.2f}")
        console.print(status_table)

        # 3. Démonstration : Question métier du contexte ("montant TTC des factures en retard en 2025")
        late_2025 = conn.execute(
            """
            SELECT
                COUNT(*) AS nb_factures,
                COALESCE(SUM(total_ttc), 0::DECIMAL(12, 2)) AS montant_ttc
            FROM gold.factures_en_retard
            WHERE strftime(issue_date, '%Y') = '2025'
            """
        ).fetchone()

        # 4. Démonstration : Top fournisseurs (Gold)
        top_suppliers = conn.execute(
            """
            SELECT supplier_name, category, nb_factures, total_ttc, nb_en_retard, montant_en_retard_ttc
            FROM gold.kpi_fournisseurs
            ORDER BY total_ttc DESC
            LIMIT 5
            """
        ).fetchall()

        sup_table = Table(
            title="Démo SQL (Gold) — Top 5 fournisseurs par CA TTC",
            show_header=True,
            header_style="bold cyan",
        )
        sup_table.add_column("Fournisseur", style="white")
        sup_table.add_column("Catégorie")
        sup_table.add_column("Factures", justify="right")
        sup_table.add_column("Total TTC (€)", justify="right", style="bold green")
        sup_table.add_column("En retard", justify="right")
        sup_table.add_column("Montant retard (€)", justify="right", style="yellow")
        for name, cat, nb, ttc, nb_ret, mt_ret in top_suppliers:
            sup_table.add_row(
                name,
                cat,
                str(nb),
                f"{ttc:,.2f}",
                str(nb_ret),
                f"{mt_ret:,.2f}",
            )
        console.print(sup_table)

    # 5. Résultat du test de concurrence + guide DBeaver
    conc_status = "[bold green]SUCCÈS[/bold green]" if concurrency.success else "[bold red]ÉCHEC[/bold red]"
    info_lines = [
        f"[bold]Question test ('factures en retard en 2025') :[/bold] "
        f"{late_2025[0]} facture(s) pour [bold yellow]{late_2025[1]:,.2f} € TTC[/bold yellow]",
        "",
        f"[bold]Test d'accès concurrent multi-processus (read_only=True) :[/bold] {conc_status} ({concurrency.elapsed_ms:.0f} ms)",
        f"  • {concurrency.worker_a_output}",
        f"  • {concurrency.worker_b_output}",
        "",
        "[bold]Connexion depuis DBeaver (sans bloquer Python / MCP) :[/bold]",
        f"  • Pilote : [cyan]DuckDB[/cyan] | Chemin BDD : [cyan]{WAREHOUSE_DB_PATH}[/cyan]",
        "  • Propriété JDBC indispensable (Driver Properties) : [bold yellow]duckdb.read_only = true[/bold yellow]",
        f"  • Répertoire de travail / file_search_path : [dim]{BASE_DIR}[/dim]",
    ]
    console.print(
        Panel(
            "\n".join(info_lines),
            title="Concurrence & Connexion Outils Graphiques (DBeaver)",
            border_style="blue",
        )
    )


def execute_sql_cli(sql_query: str) -> int:
    """Exécute une requête SQL en lecture seule et l'affiche sous forme de tableau Rich."""
    try:
        with get_warehouse_connection(read_only=True) as conn:
            cursor = conn.execute(sql_query)
            description = cursor.description or []
            rows = cursor.fetchall()
    except duckdb.Error as exc:
        console.print(f"[bold red]Erreur SQL (lecture seule) :[/bold red] {exc}")
        return 1

    if not description:
        console.print("[dim]Requête exécutée sans jeu de résultats.[/dim]")
        return 0

    table = Table(
        title=f"Résultat SQL ({len(rows)} ligne(s))",
        show_header=True,
        header_style="bold cyan",
    )
    for col in description:
        table.add_column(str(col[0]))

    for row in rows:
        table.add_row(*["" if val is None else str(val) for val in row])

    console.print(table)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI de l'étape 6."""
    parser = argparse.ArgumentParser(
        description="Étape 6 : Initialisation et interrogation en lecture seule de l'entrepôt DuckDB.",
    )
    parser.add_argument(
        "--sql",
        type=str,
        default=None,
        help="Exécute une requête SQL en lecture seule sur warehouse.duckdb et affiche le résultat.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée principal de l'étape 6."""
    args = parse_args(argv)

    if args.sql:
        return execute_sql_cli(args.sql)

    try:
        init_ms = init_warehouse()
    except (FileNotFoundError, duckdb.Error) as exc:
        console.print(f"[bold red]Erreur d'initialisation de l'entrepôt :[/bold red] {exc}")
        return 1

    concurrency = run_multiprocess_concurrency_test()
    display_warehouse_report(init_ms, concurrency)
    return 0 if concurrency.success else 1


if __name__ == "__main__":
    sys.exit(main())
