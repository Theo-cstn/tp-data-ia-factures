"""Étape 2 : Appel au modèle d'OCR LightOnOCR-3, inspection du format de sortie et mesure de latence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pymupdf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import (
    ENV_FILE,
    MANIFEST_STEP1_PATH,
    OCR_TARGET_LONGEST_EDGE,
    RAW_PDFS_DIR,
    TEXT_CHAR_THRESHOLD,
    has_valid_ocr_credentials,
)
from src.ocr import LightOnOCRProvider, OCRError, OCRPageResult, OCRProvider

console = Console()


def _find_first_pdf_by_type(pdf_type: str) -> Path | None:
    """Trouve le premier PDF du type demandé ('image' ou 'texte') via le manifeste ou le disque."""
    if MANIFEST_STEP1_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_STEP1_PATH.read_text(encoding="utf-8"))
            for entry in manifest.get("files", []):
                if entry.get("pdf_type") == pdf_type:
                    rel_path = entry.get("relative_path") or entry.get("filename")
                    if rel_path:
                        candidate = RAW_PDFS_DIR / rel_path
                        if candidate.is_file():
                            return candidate
        except (OSError, json.JSONDecodeError):
            pass

    if not RAW_PDFS_DIR.exists():
        return None

    for pdf_path in sorted(RAW_PDFS_DIR.rglob("*.pdf")):
        try:
            with pymupdf.open(pdf_path) as doc:
                char_count = sum(
                    len("".join((page.get_text("text") or "").split()))
                    for page in doc
                )
            detected = "texte" if char_count >= TEXT_CHAR_THRESHOLD else "image"
            if detected == pdf_type:
                return pdf_path
        except Exception:  # noqa: BLE001
            continue

    return None


def resolve_target_pdf(file_arg: str | None) -> Path:
    """Résout le chemin du PDF à OCRiser depuis --file ou sélectionne le premier PDF scanné."""
    if file_arg:
        direct = Path(file_arg)
        if direct.is_file():
            return direct
        in_raw = RAW_PDFS_DIR / file_arg
        if in_raw.is_file():
            return in_raw
        raise FileNotFoundError(
            f"Fichier PDF introuvable : '{file_arg}' (ni directement, ni dans {RAW_PDFS_DIR})."
        )

    default_scan = _find_first_pdf_by_type("image")
    if default_scan is None:
        raise FileNotFoundError(
            f"Aucun PDF de type 'image' trouvé dans {RAW_PDFS_DIR}. "
            "Exécutez d'abord l'Étape 1 (uv run python -m src.step1_ingest) ou passez --file."
        )
    return default_scan


def extract_native_page_text(pdf_path: Path, page_number: int) -> tuple[str, int]:
    """Extrait le texte natif d'une page d'un PDF texte via PyMuPDF."""
    with pymupdf.open(pdf_path) as doc:
        if len(doc) == 0:
            raise ValueError(f"Le fichier {pdf_path.name} ne contient aucune page.")
        target_page = page_number if 0 <= page_number < len(doc) else 0
        text = doc[target_page].get_text("text") or ""
    return text, target_page


def display_ocr_result(pdf_path: Path, result: OCRPageResult) -> None:
    """Affiche les métadonnées de l'appel OCR et la sortie brute exacte du modèle."""
    meta_table = Table(
        title="Métadonnées de l'appel OCR (Étape 2)",
        show_header=False,
        border_style="blue",
    )
    meta_table.add_column("Champ", style="bold cyan", no_wrap=True)
    meta_table.add_column("Valeur", style="white")

    size_kb = result.image_size_bytes / 1024
    elapsed_s = result.processing_time_ms / 1000.0

    meta_table.add_row("Fichier PDF", pdf_path.name)
    meta_table.add_row("Page", str(result.page_number))
    meta_table.add_row(
        "Résolution image PNG",
        f"{result.image_width} × {result.image_height} px (cible bord long : {result.target_longest_edge} px)",
    )
    meta_table.add_row(
        "Poids du PNG envoyé",
        f"{size_kb:.1f} Ko ({result.image_size_bytes:,} octets)".replace(",", " "),
    )
    meta_table.add_row("Modèle OCR", result.model)
    meta_table.add_row(
        "Temps de traitement",
        f"[bold yellow]{result.processing_time_ms:.1f} ms ({elapsed_s:.2f} s)[/bold yellow]",
    )

    console.print(meta_table)
    console.print(
        Panel(
            Text(result.text),
            title=f"Sortie brute OCR — {pdf_path.name} (page {result.page_number})",
            border_style="green",
        )
    )


def display_native_comparison(page_number: int) -> None:
    """Affiche le texte natif d'un PDF 'texte' pour comparer sa structure à la sortie OCR."""
    native_pdf = _find_first_pdf_by_type("texte")
    if native_pdf is None:
        console.print(
            "[yellow]Aucun PDF de type 'texte' trouvé pour la comparaison native.[/yellow]"
        )
        return

    native_text, used_page = extract_native_page_text(native_pdf, page_number)
    console.print(
        Panel(
            Text(native_text),
            title=f"Comparaison — Texte natif PyMuPDF : {native_pdf.name} (page {used_page})",
            border_style="cyan",
        )
    )


def run_step2(
    provider: OCRProvider,
    file_arg: str | None = None,
    page: int = 0,
    longest_edge: int = OCR_TARGET_LONGEST_EDGE,
    compare_native: bool = False,
) -> int:
    """Exécute la reconnaissance OCR sur une page et affiche le résultat."""
    try:
        pdf_path = resolve_target_pdf(file_arg)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Erreur :[/bold red] {exc}")
        return 1

    with console.status(
        f"[bold cyan]Appel OCR en cours sur {pdf_path.name} (page {page}, bord long={longest_edge} px)...[/bold cyan]"
    ):
        try:
            result = provider.extract_page(
                pdf_path=pdf_path,
                page_number=page,
                target_longest_edge=longest_edge,
            )
        except (OCRError, FileNotFoundError, IndexError, ValueError) as exc:
            console.print(f"[bold red]Échec de l'OCR :[/bold red] {exc}")
            return 1

    display_ocr_result(pdf_path, result)

    if compare_native:
        display_native_comparison(page)

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI de l'étape 2."""
    parser = argparse.ArgumentParser(
        description="Étape 2 : Envoi d'une page PDF à LightOnOCR-3, inspection du format et mesure du temps.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Nom ou chemin du fichier PDF à OCRiser (par défaut : premier PDF de type 'image' dans data/raw_pdfs/).",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=0,
        help="Numéro de page à traiter (indexé à 0, défaut : 0).",
    )
    parser.add_argument(
        "--longest-edge",
        type=int,
        default=OCR_TARGET_LONGEST_EDGE,
        help=f"Résolution cible en pixels pour le bord le plus long (défaut : {OCR_TARGET_LONGEST_EDGE}).",
    )
    parser.add_argument(
        "--compare-native",
        action="store_true",
        help="Affiche aussi le texte natif d'un PDF 'texte' pour comparer la structure avec la sortie OCR.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée principal de l'étape 2."""
    args = parse_args(argv)

    if not has_valid_ocr_credentials():
        console.print(
            Panel(
                "[bold red]Clé API Hyperfluid (HYPERFLUID_API_KEY) manquante ou non configurée.[/bold red]\n\n"
                f"Le fichier [cyan]{ENV_FILE}[/cyan] contient encore la valeur par défaut "
                "([yellow]your_hyperfluid_api_key[/yellow]).\n"
                "Veuillez renseigner votre clé d'API Hyperfluid dans [cyan].env[/cyan], puis relancer :\n"
                "  [bold]uv run python -m src.step2_ocr[/bold]",
                title="Configuration OCR requise",
                border_style="red",
            )
        )
        return 1

    provider = LightOnOCRProvider()
    return run_step2(
        provider=provider,
        file_arg=args.file,
        page=args.page,
        longest_edge=args.longest_edge,
        compare_native=args.compare_native,
    )


if __name__ == "__main__":
    sys.exit(main())
