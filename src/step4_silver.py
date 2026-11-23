"""Étape 4 : Couche Silver (parsing robuste natif + OCR, typage strict Decimal/Date, qualité et quarantaine)."""

from __future__ import annotations

import html
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from html.parser import HTMLParser
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import (
    BRONZE_PARQUET_PATH,
    SILVER_DIR,
    SILVER_INVOICES_PATH,
    SILVER_LINES_PATH,
    SILVER_QUARANTINE_PATH,
)

console = Console()

CENT = Decimal("0.01")
TOLERANCE_LINE = Decimal("0.02")
TOLERANCE_SUM = Decimal("0.02")
TOLERANCE_TVA = Decimal("0.05")
TOLERANCE_TTC = Decimal("0.02")

FACTURES_SCHEMA = pa.schema(
    [
        pa.field("invoice_number", pa.string(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("filename", pa.string(), nullable=False),
        pa.field("extraction_method", pa.string(), nullable=False),
        pa.field("issue_date", pa.date32(), nullable=False),
        pa.field("due_date", pa.date32(), nullable=True),
        pa.field("category", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("supplier_name", pa.string(), nullable=False),
        pa.field("supplier_address", pa.string(), nullable=False),
        pa.field("supplier_postal_code", pa.string(), nullable=False),
        pa.field("supplier_city", pa.string(), nullable=False),
        pa.field("supplier_siret", pa.string(), nullable=False),
        pa.field("supplier_tva_intracom", pa.string(), nullable=False),
        pa.field("client_name", pa.string(), nullable=False),
        pa.field("client_address", pa.string(), nullable=False),
        pa.field("client_postal_code", pa.string(), nullable=False),
        pa.field("client_city", pa.string(), nullable=False),
        pa.field("client_siret", pa.string(), nullable=False),
        pa.field("total_ht", pa.decimal128(12, 2), nullable=False),
        pa.field("total_tva", pa.decimal128(12, 2), nullable=False),
        pa.field("total_ttc", pa.decimal128(12, 2), nullable=False),
        pa.field("line_count", pa.int32(), nullable=False),
        pa.field("payment_method", pa.string(), nullable=False),
        pa.field("iban", pa.string(), nullable=False),
        pa.field("quality_status", pa.string(), nullable=False),
        pa.field("quality_issues", pa.string(), nullable=True),
        pa.field("processed_at", pa.string(), nullable=False),
    ]
)

LIGNES_SCHEMA = pa.schema(
    [
        pa.field("line_id", pa.string(), nullable=False),
        pa.field("invoice_number", pa.string(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("line_number", pa.int32(), nullable=False),
        pa.field("description", pa.string(), nullable=False),
        pa.field("quantity", pa.int32(), nullable=False),
        pa.field("unit_price_ht", pa.decimal128(12, 2), nullable=False),
        pa.field("tva_rate_pct", pa.decimal128(5, 2), nullable=False),
        pa.field("line_total_ht", pa.decimal128(12, 2), nullable=False),
    ]
)

QUARANTAINE_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("filename", pa.string(), nullable=False),
        pa.field("extraction_method", pa.string(), nullable=False),
        pa.field("invoice_number", pa.string(), nullable=True),
        pa.field("supplier_name", pa.string(), nullable=True),
        pa.field("client_name", pa.string(), nullable=True),
        pa.field("reported_total_ht", pa.decimal128(12, 2), nullable=True),
        pa.field("calculated_lines_ht", pa.decimal128(12, 2), nullable=True),
        pa.field("reported_total_tva", pa.decimal128(12, 2), nullable=True),
        pa.field("reported_total_ttc", pa.decimal128(12, 2), nullable=True),
        pa.field("calculated_total_ttc", pa.decimal128(12, 2), nullable=True),
        pa.field("rejection_codes", pa.string(), nullable=False),
        pa.field("rejection_reasons", pa.string(), nullable=False),
        pa.field("quarantined_at", pa.string(), nullable=False),
    ]
)


class _HTMLTableParser(HTMLParser):
    """Extracteur minimaliste de tableaux HTML <table> en liste de lignes et cellules."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._cell_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif tag == "tr" and self._in_table:
            self._in_row = True
            self._current_row = []
        elif tag in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._cell_chunks = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._in_cell:
            self._in_cell = False
            cell_text = " ".join(html.unescape("".join(self._cell_chunks)).split())
            self._current_row.append(cell_text)
        elif tag == "tr" and self._in_row:
            self._in_row = False
            if self._current_row:
                self._current_table.append(self._current_row)
        elif tag == "table" and self._in_table:
            self._in_table = False
            if self._current_table:
                self.tables.append(self._current_table)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_chunks.append(data)


@dataclass
class ParsedLine:
    """Ligne de facture brute parsée."""

    line_number: int
    description: str
    quantity: int | None
    unit_price_ht: Decimal | None
    tva_rate_pct: Decimal | None
    line_total_ht: Decimal | None
    raw_error: str | None = None


@dataclass
class ParsedInvoice:
    """Facture parsée avant contrôles qualité."""

    doc_id: str
    filename: str
    extraction_method: str
    invoice_number: str = ""
    issue_date: date | None = None
    due_date: date | None = None
    category: str = ""
    status: str = ""
    supplier_name: str = ""
    supplier_address: str = ""
    supplier_postal_code: str = ""
    supplier_city: str = ""
    supplier_siret: str = ""
    supplier_tva_intracom: str = ""
    client_name: str = ""
    client_address: str = ""
    client_postal_code: str = ""
    client_city: str = ""
    client_siret: str = ""
    total_ht: Decimal | None = None
    summary_tva_rate: Decimal | None = None
    total_tva: Decimal | None = None
    total_ttc: Decimal | None = None
    payment_method: str = ""
    iban: str = ""
    lines: list[ParsedLine] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


@dataclass
class SilverRunStats:
    """Statistiques d'exécution de la couche Silver."""

    bronze_docs_read: int = 0
    valid_count: int = 0
    warned_count: int = 0
    rejected_count: int = 0
    total_lines_silver: int = 0


def _clean_md(text: str) -> str:
    """Supprime le formatage Markdown basique et normalise les espaces."""
    cleaned = text.replace("**", "").strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    return " ".join(cleaned.split())


def _parse_decimal(raw: str) -> Decimal | None:
    """Parse un montant ou pourcentage au format FR/EN en Decimal quantisé à 0.01."""
    if not raw:
        return None
    cleaned = raw.replace("€", "").replace("EUR", "").replace("%", "")
    cleaned = cleaned.replace("\xa0", "").replace("\u202f", "").replace(" ", "")
    cleaned = cleaned.replace(",", ".").strip()
    if not cleaned:
        return None
    try:
        return Decimal(cleaned).quantize(CENT, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None


def _parse_int(raw: str) -> int | None:
    """Parse un entier en supprimant les espaces."""
    cleaned = raw.replace("\xa0", "").replace("\u202f", "").replace(" ", "").strip()
    if not cleaned or not re.fullmatch(r"-?\d+", cleaned):
        return None
    return int(cleaned)


def _parse_date(raw: str) -> date | None:
    """Parse une date JJ/MM/AAAA ou AAAA-MM-JJ en datetime.date."""
    cleaned = raw.strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _split_postal_city(line: str) -> tuple[str, str]:
    """Sépare une ligne 'CODE_POSTAL Ville'."""
    match = re.match(r"^(\d{5})\s+(.+)$", line.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", line.strip()


def _extract_kv(line: str) -> tuple[str, str] | None:
    """Extrait une paire 'Clé : Valeur' d'une ligne nettoyée du Markdown."""
    cleaned = _clean_md(line)
    match = re.match(r"^([^:]+?)\s*:\s*(.+)$", cleaned)
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _extract_html_tables(raw_text: str) -> list[list[list[str]]]:
    """Extrait tous les tableaux HTML présents dans raw_text."""
    parser = _HTMLTableParser()
    parser.feed(raw_text)
    parser.close()
    return parser.tables


def _parse_line_cells(line_number: int, cells: list[str]) -> ParsedLine:
    """Construit un ParsedLine à partir de 5 cellules [Désignation, Qté, PU HT, TVA, Total HT]."""
    if len(cells) < 5:
        return ParsedLine(
            line_number=line_number,
            description=" ".join(cells),
            quantity=None,
            unit_price_ht=None,
            tva_rate_pct=None,
            line_total_ht=None,
            raw_error=f"Ligne {line_number} incomplète ({len(cells)} colonne(s) au lieu de 5).",
        )

    desc, qty_raw, pu_raw, tva_raw, total_raw = [c.strip() for c in cells[:5]]
    qty = _parse_int(qty_raw)
    pu = _parse_decimal(pu_raw)
    tva = _parse_decimal(tva_raw)
    line_total = _parse_decimal(total_raw)

    errors: list[str] = []
    if not desc:
        errors.append("désignation vide")
    if qty is None:
        errors.append(f"quantité invalide ({qty_raw!r})")
    if pu is None:
        errors.append(f"prix unitaire invalide ({pu_raw!r})")
    if tva is None:
        errors.append(f"taux TVA invalide ({tva_raw!r})")
    if line_total is None:
        errors.append(f"total HT invalide ({total_raw!r})")

    return ParsedLine(
        line_number=line_number,
        description=desc,
        quantity=qty,
        unit_price_ht=pu,
        tva_rate_pct=tva,
        line_total_ht=line_total,
        raw_error=f"Ligne {line_number} : {', '.join(errors)}" if errors else None,
    )


def _apply_total_label(invoice: ParsedInvoice, label: str, value_raw: str) -> None:
    """Affecte un libellé de total (Total HT, TVA X %, Total TVA, Total TTC) à l'invoice."""
    norm = _clean_md(label).lower()
    amount = _parse_decimal(value_raw)
    if amount is None:
        return

    if norm == "total ht":
        invoice.total_ht = amount
    elif norm == "total tva":
        invoice.total_tva = amount
    elif norm == "total ttc":
        invoice.total_ttc = amount
    else:
        tva_match = re.match(r"^tva\s+(\d+(?:[.,]\d+)?)\s*%$", norm)
        if tva_match:
            invoice.summary_tva_rate = _parse_decimal(tva_match.group(1))
            if invoice.total_tva is None:
                invoice.total_tva = amount


def _parse_ocr_tables_and_totals(raw_text: str, invoice: ParsedInvoice) -> None:
    """Parse les tableaux HTML et les totaux (tableau HTML ou lignes Markdown) d'un texte OCR."""
    tables = _extract_html_tables(raw_text)
    if not tables:
        invoice.parse_errors.append("Aucun tableau HTML <table> trouvé dans la sortie OCR.")
        return

    items_table = tables[0]
    summary_label_re = re.compile(
        r"^(total\s+ht|total\s+tva|total\s+ttc|tva\s+\d+(?:[.,]\d+)?\s*%)$",
        re.IGNORECASE,
    )
    data_rows = []
    for row in items_table:
        if not row:
            continue
        first_cell = _clean_md(row[0]).lower()
        if first_cell in {"désignation", "designation", "description"}:
            continue
        if len(row) == 2 and summary_label_re.match(first_cell):
            _apply_total_label(invoice, row[0], row[1])
            continue
        data_rows.append(row)

    for idx, row in enumerate(data_rows, start=1):
        parsed_line = _parse_line_cells(idx, row)
        if parsed_line.raw_error:
            invoice.parse_errors.append(parsed_line.raw_error)
        invoice.lines.append(parsed_line)

    # Cas 1 : Second tableau HTML contenant les totaux
    if len(tables) >= 2:
        for totals_table in tables[1:]:
            for row in totals_table:
                if len(row) >= 2:
                    _apply_total_label(invoice, row[0], row[1])

    # Cas 2 : Lignes Markdown/texte après la première table (ex. "**Total HT** 6 306,05 €")
    after_first_table = raw_text.split("</table>", 1)[1] if "</table>" in raw_text else raw_text
    after_clean_lines = [
        _clean_md(ln)
        for ln in after_first_table.splitlines()
        if ln.strip() and not ln.strip().startswith("<") and ln.strip() != "---"
    ]

    total_pattern = re.compile(
        r"^(Total\s+HT|Total\s+TVA|Total\s+TTC|TVA\s+\d+(?:[.,]\d+)?\s*%)\s*:?\s+(.+)$",
        re.IGNORECASE,
    )
    for idx, line in enumerate(after_clean_lines):
        match = total_pattern.match(line)
        if match:
            _apply_total_label(invoice, match.group(1), match.group(2))
        elif line.lower() in {"total ht", "total tva", "total ttc"} or re.match(
            r"^tva\s+\d+(?:[.,]\d+)?\s*%$", line.lower()
        ):
            if idx + 1 < len(after_clean_lines):
                _apply_total_label(invoice, line, after_clean_lines[idx + 1])


def _parse_native_table_and_totals(lines: list[str], invoice: ParsedInvoice) -> None:
    """Parse les lignes d'articles (blocs de 5 lignes) et les totaux d'un PDF natif."""
    start_idx = None
    for idx in range(len(lines) - 4):
        window = [_clean_md(lines[idx + k]).lower() for k in range(5)]
        if window[0] in {"désignation", "designation"} and window[4] == "total ht":
            start_idx = idx + 5
            break

    if start_idx is None:
        invoice.parse_errors.append("En-tête du tableau des lignes introuvable dans le texte natif.")
        return

    summary_idx = None
    for idx in range(start_idx, len(lines)):
        if _clean_md(lines[idx]).lower() == "total ht":
            summary_idx = idx
            break

    if summary_idx is None:
        invoice.parse_errors.append("Section récapitulative 'Total HT' introuvable dans le texte natif.")
        return

    item_tokens = lines[start_idx:summary_idx]
    if len(item_tokens) % 5 != 0:
        invoice.parse_errors.append(
            f"Nombre de cellules de lignes inattendu ({len(item_tokens)} lignes, non multiple de 5)."
        )

    line_num = 1
    for i in range(0, len(item_tokens) - (len(item_tokens) % 5), 5):
        parsed_line = _parse_line_cells(line_num, item_tokens[i : i + 5])
        if parsed_line.raw_error:
            invoice.parse_errors.append(parsed_line.raw_error)
        invoice.lines.append(parsed_line)
        line_num += 1

    # Totaux jusqu'au bloc pied de page
    footer_markers = ("mode de paiement", "iban")
    end_summary_idx = len(lines)
    for idx in range(summary_idx, len(lines)):
        low = _clean_md(lines[idx]).lower()
        if any(low.startswith(marker) for marker in footer_markers):
            end_summary_idx = idx
            break

    summary_lines = [_clean_md(ln) for ln in lines[summary_idx:end_summary_idx]]
    idx = 0
    while idx < len(summary_lines):
        line = summary_lines[idx]
        inline_match = re.match(
            r"^(Total\s+HT|Total\s+TVA|Total\s+TTC|TVA\s+\d+(?:[.,]\d+)?\s*%)\s*:?\s+(.+)$",
            line,
            re.IGNORECASE,
        )
        if inline_match:
            _apply_total_label(invoice, inline_match.group(1), inline_match.group(2))
            idx += 1
        elif idx + 1 < len(summary_lines):
            _apply_total_label(invoice, line, summary_lines[idx + 1])
            idx += 2
        else:
            idx += 1


def parse_bronze_document(doc: dict) -> ParsedInvoice:
    """Parse un document brut de la couche Bronze (format natif ou OCR LightOnOCR-3)."""
    raw_text: str = doc.get("raw_text") or ""
    invoice = ParsedInvoice(
        doc_id=doc["doc_id"],
        filename=doc["filename"],
        extraction_method=doc["extraction_method"],
    )

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip() and ln.strip() != "---"]

    # 1. En-tête fournisseur (avant "Numéro de facture")
    inv_num_idx = next(
        (i for i, ln in enumerate(lines) if "numéro de facture" in _clean_md(ln).lower()),
        None,
    )
    header_end = inv_num_idx if inv_num_idx is not None else min(len(lines), 8)
    supplier_lines: list[str] = []
    for raw_ln in lines[:header_end]:
        cleaned = _clean_md(raw_ln)
        if cleaned.upper() == "FACTURE":
            continue
        kv = _extract_kv(raw_ln)
        if kv and kv[0].lower() == "siret":
            invoice.supplier_siret = kv[1].replace(" ", "")
        elif kv and "tva intracom" in kv[0].lower():
            invoice.supplier_tva_intracom = kv[1].replace(" ", "")
        else:
            supplier_lines.append(cleaned)

    if len(supplier_lines) >= 1:
        invoice.supplier_name = supplier_lines[0]
    if len(supplier_lines) >= 2:
        invoice.supplier_address = supplier_lines[1]
    if len(supplier_lines) >= 3:
        cp, city = _split_postal_city(supplier_lines[2])
        invoice.supplier_postal_code = cp
        invoice.supplier_city = city

    # 2. Métadonnées globales & pied de page (paires Clé : Valeur)
    for raw_ln in lines:
        if raw_ln.startswith("<"):
            continue
        kv = _extract_kv(raw_ln)
        if not kv:
            continue
        key, val = kv
        key_low = key.lower()
        if key_low == "numéro de facture":
            invoice.invoice_number = val
        elif key_low == "date d'émission":
            invoice.issue_date = _parse_date(val)
        elif key_low == "date d'échéance":
            invoice.due_date = _parse_date(val)
        elif key_low == "catégorie":
            invoice.category = val
        elif key_low == "statut":
            invoice.status = val
        elif key_low == "client":
            invoice.client_name = val
        elif key_low == "siret client":
            invoice.client_siret = val.replace(" ", "")
        elif key_low == "mode de paiement":
            invoice.payment_method = val
        elif key_low == "iban":
            invoice.iban = val.replace(" ", "").upper()

    # 3. Adresse client (entre "Client :" et "SIRET client :")
    client_idx = next(
        (i for i, ln in enumerate(lines) if _clean_md(ln).lower().startswith("client :")),
        None,
    )
    if client_idx is not None:
        client_addr_lines: list[str] = []
        for raw_ln in lines[client_idx + 1 :]:
            cleaned = _clean_md(raw_ln)
            low = cleaned.lower()
            if (
                low.startswith("siret client")
                or low in {"désignation", "designation"}
                or raw_ln.startswith("<table")
            ):
                break
            client_addr_lines.append(cleaned)
        if len(client_addr_lines) >= 1:
            invoice.client_address = client_addr_lines[0]
        if len(client_addr_lines) >= 2:
            cp, city = _split_postal_city(client_addr_lines[1])
            invoice.client_postal_code = cp
            invoice.client_city = city

    # 4. Lignes de facture et totaux
    if "<table" in raw_text.lower():
        _parse_ocr_tables_and_totals(raw_text, invoice)
    else:
        _parse_native_table_and_totals(lines, invoice)

    return invoice


def validate_invoice(
    invoice: ParsedInvoice,
    seen_invoice_numbers: set[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], Decimal | None, Decimal | None]:
    """Applique les contrôles qualité et retourne (rejections, warnings, calc_lines_ht, calc_ttc)."""
    rejections: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []

    # --- A. Contrôles bloquants (REJET) ---
    if invoice.parse_errors:
        rejections.append(("ERREUR_PARSING", "; ".join(invoice.parse_errors)))

    missing_required: list[str] = []
    if not invoice.invoice_number:
        missing_required.append("invoice_number")
    if not invoice.supplier_name:
        missing_required.append("supplier_name")
    if not invoice.client_name:
        missing_required.append("client_name")
    if invoice.issue_date is None:
        missing_required.append("issue_date")
    if invoice.total_ht is None:
        missing_required.append("total_ht")
    if invoice.total_tva is None:
        missing_required.append("total_tva")
    if invoice.total_ttc is None:
        missing_required.append("total_ttc")
    if not invoice.lines:
        missing_required.append("lignes_facture")

    if missing_required:
        rejections.append(
            (
                "CHAMPS_OBLIGATOIRES_MANQUANTS",
                f"Champ(s) obligatoire(s) manquant(s) : {', '.join(missing_required)}.",
            )
        )

    if invoice.invoice_number and invoice.invoice_number in seen_invoice_numbers:
        rejections.append(
            (
                "DOUBLON_FACTURE",
                f"Numéro de facture '{invoice.invoice_number}' déjà ingéré dans Silver (doublon écarté).",
            )
        )

    # Montants négatifs ou nuls
    neg_or_zero_reasons: list[str] = []
    if invoice.total_ht is not None and invoice.total_ht <= Decimal("0.00"):
        neg_or_zero_reasons.append(f"total_ht={invoice.total_ht}")
    if invoice.total_ttc is not None and invoice.total_ttc <= Decimal("0.00"):
        neg_or_zero_reasons.append(f"total_ttc={invoice.total_ttc}")
    if invoice.total_tva is not None and invoice.total_tva < Decimal("0.00"):
        neg_or_zero_reasons.append(f"total_tva={invoice.total_tva}")

    valid_line_totals: list[Decimal] = []
    for line in invoice.lines:
        if line.quantity is not None and line.quantity <= 0:
            neg_or_zero_reasons.append(f"ligne {line.line_number} qté={line.quantity}")
        if line.unit_price_ht is not None and line.unit_price_ht <= Decimal("0.00"):
            neg_or_zero_reasons.append(f"ligne {line.line_number} pu_ht={line.unit_price_ht}")
        if line.line_total_ht is not None:
            if line.line_total_ht <= Decimal("0.00"):
                neg_or_zero_reasons.append(f"ligne {line.line_number} total_ht={line.line_total_ht}")
            valid_line_totals.append(line.line_total_ht)

        if (
            line.quantity is not None
            and line.unit_price_ht is not None
            and line.line_total_ht is not None
        ):
            expected_line_ht = (Decimal(line.quantity) * line.unit_price_ht).quantize(
                CENT, rounding=ROUND_HALF_UP
            )
            diff_line = abs(expected_line_ht - line.line_total_ht)
            if diff_line > TOLERANCE_LINE:
                rejections.append(
                    (
                        "INCOHERENCE_LIGNE_HT",
                        f"Ligne {line.line_number} : {line.quantity} × {line.unit_price_ht} € = "
                        f"{expected_line_ht} € mais lu {line.line_total_ht} € (écart {diff_line} €).",
                    )
                )

    if neg_or_zero_reasons:
        rejections.append(
            (
                "MONTANT_NEGATIF_OU_NUL",
                f"Valeur(s) <= 0 détectée(s) : {', '.join(neg_or_zero_reasons)}.",
            )
        )

    calc_lines_ht: Decimal | None = None
    if valid_line_totals and len(valid_line_totals) == len(invoice.lines):
        calc_lines_ht = sum(valid_line_totals, Decimal("0.00")).quantize(
            CENT, rounding=ROUND_HALF_UP
        )
        if invoice.total_ht is not None:
            diff_sum = abs(calc_lines_ht - invoice.total_ht)
            if diff_sum > TOLERANCE_SUM:
                rejections.append(
                    (
                        "INCOHERENCE_SOMME_LIGNES",
                        f"Somme des lignes HT = {calc_lines_ht} € mais Total HT déclaré = "
                        f"{invoice.total_ht} € (écart {diff_sum} €).",
                    )
                )

    # Vérification TVA
    tva_rate = invoice.summary_tva_rate
    if tva_rate is None and invoice.lines:
        rates = {ln.tva_rate_pct for ln in invoice.lines if ln.tva_rate_pct is not None}
        if len(rates) == 1:
            tva_rate = next(iter(rates))

    if invoice.total_ht is not None and invoice.total_tva is not None and tva_rate is not None:
        expected_tva = (invoice.total_ht * (tva_rate / Decimal("100"))).quantize(
            CENT, rounding=ROUND_HALF_UP
        )
        diff_tva = abs(expected_tva - invoice.total_tva)
        if diff_tva > TOLERANCE_TVA:
            rejections.append(
                (
                    "INCOHERENCE_TVA",
                    f"TVA attendue ({invoice.total_ht} € × {tva_rate}%) = {expected_tva} € "
                    f"mais Total TVA déclaré = {invoice.total_tva} € (écart {diff_tva} €).",
                )
            )

    # Vérification TTC
    calc_ttc: Decimal | None = None
    if invoice.total_ht is not None and invoice.total_tva is not None:
        calc_ttc = (invoice.total_ht + invoice.total_tva).quantize(CENT, rounding=ROUND_HALF_UP)
        if invoice.total_ttc is not None:
            diff_ttc = abs(calc_ttc - invoice.total_ttc)
            if diff_ttc > TOLERANCE_TTC:
                rejections.append(
                    (
                        "INCOHERENCE_TTC",
                        f"HT ({invoice.total_ht} €) + TVA ({invoice.total_tva} €) = {calc_ttc} € "
                        f"mais Total TTC déclaré = {invoice.total_ttc} € (écart {diff_ttc} €).",
                    )
                )

    # --- B. Contrôles non-bloquants (SIGNALEMENT) ---
    if invoice.issue_date is not None and invoice.due_date is not None:
        if invoice.due_date < invoice.issue_date:
            warnings.append(
                (
                    "DATE_ECHEANCE_INCOHERENTE",
                    f"Date d'échéance ({invoice.due_date.isoformat()}) antérieure à la date "
                    f"d'émission ({invoice.issue_date.isoformat()}).",
                )
            )
    elif invoice.due_date is None:
        warnings.append(("DATE_ECHEANCE_INCOHERENTE", "Date d'échéance absente ou illisible."))

    if not re.fullmatch(r"\d{14}", invoice.supplier_siret):
        warnings.append(
            (
                "SIRET_INVALIDE",
                f"SIRET fournisseur invalide ou absent ({invoice.supplier_siret!r}).",
            )
        )
    if not re.fullmatch(r"\d{14}", invoice.client_siret):
        warnings.append(
            (
                "SIRET_INVALIDE",
                f"SIRET client invalide ou absent ({invoice.client_siret!r}).",
            )
        )

    if not invoice.iban or not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", invoice.iban):
        warnings.append(
            (
                "IBAN_MANQUANT_OU_INVALIDE",
                f"IBAN manquant ou format anormal ({invoice.iban!r}).",
            )
        )

    if not invoice.status or not invoice.category:
        warnings.append(
            (
                "STATUT_OU_CATEGORIE_INCONNU",
                f"Champ secondaire manquant (statut={invoice.status!r}, catégorie={invoice.category!r}).",
            )
        )

    return rejections, warnings, calc_lines_ht, calc_ttc


def _write_parquet_atomic(rows: list[dict], schema: pa.Schema, target_path: Path) -> None:
    """Écrit une table PyArrow typée dans un fichier Parquet de manière atomique."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    tmp_path = target_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path, compression="snappy")
    os.replace(tmp_path, target_path)


def run_silver_pipeline(
    bronze_parquet_path: Path = BRONZE_PARQUET_PATH,
    invoices_path: Path = SILVER_INVOICES_PATH,
    lines_path: Path = SILVER_LINES_PATH,
    quarantine_path: Path = SILVER_QUARANTINE_PATH,
) -> tuple[SilverRunStats, list[dict], list[dict], list[dict]]:
    """Transforme la couche Bronze en tables Silver typées et table de quarantaine."""
    if not bronze_parquet_path.exists():
        raise FileNotFoundError(
            f"Fichier Bronze introuvable : {bronze_parquet_path}. Exécutez d'abord l'Étape 3."
        )

    bronze_table = pq.read_table(bronze_parquet_path)
    bronze_docs = bronze_table.to_pylist()

    # Priorité à l'extraction native en cas de doublon natif/scan, puis ordre déterministe
    bronze_docs.sort(
        key=lambda d: (
            0 if d.get("extraction_method") == "pymupdf_native" else 1,
            d.get("filename", ""),
        )
    )

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()

    stats = SilverRunStats(bronze_docs_read=len(bronze_docs))
    seen_invoice_numbers: set[str] = set()

    factures_rows: list[dict] = []
    lignes_rows: list[dict] = []
    quarantine_rows: list[dict] = []

    for doc in bronze_docs:
        parsed = parse_bronze_document(doc)
        rejections, warnings, calc_lines_ht, calc_ttc = validate_invoice(
            parsed, seen_invoice_numbers
        )

        if rejections:
            # Dédupliquer les codes tout en conservant l'ordre
            codes = list(dict.fromkeys(code for code, _ in rejections))
            reasons = [f"[{code}] {msg}" for code, msg in rejections]
            quarantine_rows.append(
                {
                    "doc_id": parsed.doc_id,
                    "filename": parsed.filename,
                    "extraction_method": parsed.extraction_method,
                    "invoice_number": parsed.invoice_number or None,
                    "supplier_name": parsed.supplier_name or None,
                    "client_name": parsed.client_name or None,
                    "reported_total_ht": parsed.total_ht,
                    "calculated_lines_ht": calc_lines_ht,
                    "reported_total_tva": parsed.total_tva,
                    "reported_total_ttc": parsed.total_ttc,
                    "calculated_total_ttc": calc_ttc,
                    "rejection_codes": ", ".join(codes),
                    "rejection_reasons": " | ".join(reasons),
                    "quarantined_at": now_iso,
                }
            )
            stats.rejected_count += 1
            continue

        seen_invoice_numbers.add(parsed.invoice_number)

        if warnings:
            quality_status = "SIGNALEE"
            quality_issues = " | ".join(f"[{code}] {msg}" for code, msg in warnings)
            stats.warned_count += 1
        else:
            quality_status = "VALIDE"
            quality_issues = None
            stats.valid_count += 1

        factures_rows.append(
            {
                "invoice_number": parsed.invoice_number,
                "doc_id": parsed.doc_id,
                "filename": parsed.filename,
                "extraction_method": parsed.extraction_method,
                "issue_date": parsed.issue_date,
                "due_date": parsed.due_date,
                "category": parsed.category,
                "status": parsed.status,
                "supplier_name": parsed.supplier_name,
                "supplier_address": parsed.supplier_address,
                "supplier_postal_code": parsed.supplier_postal_code,
                "supplier_city": parsed.supplier_city,
                "supplier_siret": parsed.supplier_siret,
                "supplier_tva_intracom": parsed.supplier_tva_intracom,
                "client_name": parsed.client_name,
                "client_address": parsed.client_address,
                "client_postal_code": parsed.client_postal_code,
                "client_city": parsed.client_city,
                "client_siret": parsed.client_siret,
                "total_ht": parsed.total_ht,
                "total_tva": parsed.total_tva,
                "total_ttc": parsed.total_ttc,
                "line_count": len(parsed.lines),
                "payment_method": parsed.payment_method,
                "iban": parsed.iban,
                "quality_status": quality_status,
                "quality_issues": quality_issues,
                "processed_at": now_iso,
            }
        )

        for line in parsed.lines:
            lignes_rows.append(
                {
                    "line_id": f"{parsed.invoice_number}_L{line.line_number}",
                    "invoice_number": parsed.invoice_number,
                    "doc_id": parsed.doc_id,
                    "line_number": line.line_number,
                    "description": line.description,
                    "quantity": line.quantity,
                    "unit_price_ht": line.unit_price_ht,
                    "tva_rate_pct": line.tva_rate_pct,
                    "line_total_ht": line.line_total_ht,
                }
            )

    stats.total_lines_silver = len(lignes_rows)

    _write_parquet_atomic(factures_rows, FACTURES_SCHEMA, invoices_path)
    _write_parquet_atomic(lignes_rows, LIGNES_SCHEMA, lines_path)
    _write_parquet_atomic(quarantine_rows, QUARANTAINE_SCHEMA, quarantine_path)

    return stats, factures_rows, lignes_rows, quarantine_rows


def display_silver_report(
    stats: SilverRunStats,
    factures_rows: list[dict],
    quarantine_rows: list[dict],
) -> None:
    """Affiche le bilan global, la table de quarantaine et les factures signalées avec Rich."""
    summary_table = Table(
        title="Résumé Couche Silver & Contrôles Qualité (Étape 4)",
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Indicateur", style="white")
    summary_table.add_column("Valeur", justify="right", style="bold")

    summary_table.add_row("Documents Bronze lus", str(stats.bronze_docs_read))
    summary_table.add_row("Factures VALIDES", f"[bold green]{stats.valid_count}[/bold green]")
    summary_table.add_row("Factures SIGNALÉES", f"[bold yellow]{stats.warned_count}[/bold yellow]")
    summary_table.add_row(
        "Factures REJETÉES (quarantaine)", f"[bold red]{stats.rejected_count}[/bold red]"
    )
    summary_table.add_row("Total lignes de facture en Silver", f"[cyan]{stats.total_lines_silver}[/cyan]")

    console.print(summary_table)

    if quarantine_rows:
        q_table = Table(
            title=f"Factures mises en QUARANTAINE ({len(quarantine_rows)})",
            show_header=True,
            header_style="bold red",
            show_lines=True,
        )
        q_table.add_column("Fichier", style="cyan")
        q_table.add_column("N° Facture", no_wrap=True)
        q_table.add_column("Méthode", no_wrap=True)
        q_table.add_column("Codes rejet", style="bold red", no_wrap=True)
        q_table.add_column("Motifs détaillés & montants")

        for row in quarantine_rows:
            q_table.add_row(
                row["filename"],
                row["invoice_number"] or "N/A",
                row["extraction_method"],
                row["rejection_codes"],
                row["rejection_reasons"],
            )
        console.print(q_table)
    else:
        console.print("[green]Aucune facture rejetée en quarantaine.[/green]")

    signaled_rows = [r for r in factures_rows if r["quality_status"] == "SIGNALEE"]
    if signaled_rows:
        s_table = Table(
            title=f"Factures SIGNALÉES conservées en Silver ({len(signaled_rows)})",
            show_header=True,
            header_style="bold yellow",
            show_lines=True,
        )
        s_table.add_column("Fichier", style="cyan", no_wrap=True)
        s_table.add_column("N° Facture", no_wrap=True)
        s_table.add_column("Méthode", no_wrap=True)
        s_table.add_column("Alertes qualité")

        for row in signaled_rows:
            s_table.add_row(
                row["filename"],
                row["invoice_number"],
                row["extraction_method"],
                row["quality_issues"] or "",
            )
        console.print(s_table)
    else:
        console.print("[dim]Aucune facture signalée (quality_status = 'SIGNALEE').[/dim]")

    console.print(
        Panel(
            f"[bold]Factures Silver :[/bold] [cyan]{SILVER_INVOICES_PATH}[/cyan]\n"
            f"[bold]Lignes Silver :[/bold] [cyan]{SILVER_LINES_PATH}[/cyan]\n"
            f"[bold]Quarantaine :[/bold] [cyan]{SILVER_QUARANTINE_PATH}[/cyan]",
            border_style="blue",
        )
    )


def main() -> int:
    """Point d'entrée principal de l'étape 4."""
    try:
        stats, factures_rows, _, quarantine_rows = run_silver_pipeline()
    except FileNotFoundError as exc:
        console.print(f"[bold red]Erreur :[/bold red] {exc}")
        return 1

    display_silver_report(stats, factures_rows, quarantine_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
