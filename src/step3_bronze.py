"""Étape 3 : Couche Bronze en Parquet (extraction brute native/OCR, SHA-256, incrémentalité et tolérance aux pannes)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from src.config import (
    BRONZE_DIR,
    BRONZE_ERRORS_LOG_PATH,
    BRONZE_PARQUET_PATH,
    OCR_MODEL,
    OCR_TARGET_LONGEST_EDGE,
    RAW_PDFS_DIR,
    S3_BUCKET_NAME,
    TEXT_CHAR_THRESHOLD,
    has_valid_ocr_credentials,
)
from src.ocr import LightOnOCRProvider, OCRError, OCRPageResult, OCRProvider

console = Console()

BRONZE_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("file_sha256", pa.string(), nullable=False),
        pa.field("filename", pa.string(), nullable=False),
        pa.field("source_uri", pa.string(), nullable=False),
        pa.field("file_size_bytes", pa.int64(), nullable=False),
        pa.field("page_count", pa.int32(), nullable=False),
        pa.field("pdf_type", pa.string(), nullable=False),
        pa.field("extraction_method", pa.string(), nullable=False),
        pa.field("ocr_model", pa.string(), nullable=True),
        pa.field("ocr_longest_edge", pa.int32(), nullable=True),
        pa.field("raw_text", pa.string(), nullable=False),
        pa.field("char_count", pa.int32(), nullable=False),
        pa.field("processing_time_ms", pa.float64(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


@dataclass
class BronzeRunStats:
    """Statistiques d'exécution de l'ingestion Bronze."""

    total_local_pdfs: int = 0
    skipped_existing: int = 0
    extracted_native: int = 0
    extracted_ocr: int = 0
    errors_captured: int = 0
    total_bronze_docs: int = 0
    total_processing_time_ms: float = 0.0

    @property
    def extracted_total(self) -> int:
        return self.extracted_native + self.extracted_ocr

    @property
    def avg_processing_time_ms(self) -> float:
        if self.extracted_total == 0:
            return 0.0
        return self.total_processing_time_ms / self.extracted_total


class SimulatedErrorOCRProvider(OCRProvider):
    """Fournisseur OCR simulé levant une erreur réseau pour tester la tolérance aux pannes."""

    def __init__(self, model: str = OCR_MODEL) -> None:
        self.model = model

    def extract_page(
        self,
        pdf_path: Path,
        page_number: int = 0,
        target_longest_edge: int = OCR_TARGET_LONGEST_EDGE,
    ) -> OCRPageResult:
        try:
            raise httpx.ConnectError(
                "Simulation d'indisponibilité du service OCR (--simulate-ocr-error)"
            )
        except httpx.ConnectError as exc:
            raise OCRError(str(exc)) from exc


class MissingCredentialsOCRProvider(OCRProvider):
    """Fournisseur OCR de repli lorsque la clé API Hyperfluid n'est pas configurée."""

    def __init__(self, model: str = OCR_MODEL) -> None:
        self.model = model

    def extract_page(
        self,
        pdf_path: Path,
        page_number: int = 0,
        target_longest_edge: int = OCR_TARGET_LONGEST_EDGE,
    ) -> OCRPageResult:
        raise OCRError(
            "Clé API Hyperfluid (HYPERFLUID_API_KEY) non configurée dans .env : appel OCR impossible."
        )


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Calcule l'empreinte SHA-256 hexadécimale d'un fichier binaire."""
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_existing_bronze_table(parquet_path: Path) -> tuple[pa.Table | None, set[str]]:
    """Charge la table Parquet Bronze existante et retourne l'ensemble des SHA-256 déjà ingérés."""
    if not parquet_path.exists():
        return None, set()

    table = pq.read_table(parquet_path)
    hashes = set(table.column("file_sha256").to_pylist())
    return table, hashes


def append_record_to_parquet(
    record: dict,
    parquet_path: Path,
    existing_table: pa.Table | None,
) -> pa.Table:
    """Ajoute un enregistrement en mode append-only dans le fichier Parquet (écriture atomique)."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    new_row_table = pa.Table.from_pylist([record], schema=BRONZE_SCHEMA)

    if existing_table is not None:
        if existing_table.schema != BRONZE_SCHEMA:
            existing_table = existing_table.cast(BRONZE_SCHEMA)
        updated_table = pa.concat_tables([existing_table, new_row_table])
    else:
        updated_table = new_row_table

    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(updated_table, tmp_path, compression="snappy")
    os.replace(tmp_path, parquet_path)
    return updated_table


def log_ingestion_error(
    errors_log_path: Path,
    filename: str,
    file_sha256: str,
    exc: Exception,
) -> None:
    """Enregistre une erreur d'extraction dans le journal JSONL Bronze."""
    errors_log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "file_sha256": file_sha256,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    with errors_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def extract_bronze_record(
    pdf_path: Path,
    raw_pdfs_dir: Path,
    file_sha256: str,
    ocr_provider: OCRProvider,
    bucket_name: str = S3_BUCKET_NAME,
    target_longest_edge: int = OCR_TARGET_LONGEST_EDGE,
) -> dict:
    """Inspecte et extrait le texte brut (natif ou OCR) d'un PDF pour la couche Bronze."""
    rel_name = pdf_path.relative_to(raw_pdfs_dir).as_posix()
    source_uri = f"s3://{bucket_name}/{rel_name}"
    file_size_bytes = pdf_path.stat().st_size

    start_native = time.perf_counter()
    with pymupdf.open(pdf_path) as doc:
        page_count = len(doc)
        native_pages = [page.get_text("text") or "" for page in doc]
    native_elapsed_ms = (time.perf_counter() - start_native) * 1000.0

    significant_chars = sum(len("".join(text.split())) for text in native_pages)

    if significant_chars >= TEXT_CHAR_THRESHOLD:
        pdf_type = "texte"
        extraction_method = "pymupdf_native"
        ocr_model = None
        ocr_longest_edge = None
        raw_text = "\n\n".join(native_pages)
        processing_time_ms = native_elapsed_ms
    else:
        pdf_type = "image"
        extraction_method = "ocr_api"
        ocr_results = ocr_provider.extract_document(
            pdf_path=pdf_path,
            target_longest_edge=target_longest_edge,
        )
        raw_text = "\n\n".join(res.text for res in ocr_results)
        processing_time_ms = sum(res.processing_time_ms for res in ocr_results)
        ocr_model = ocr_results[0].model if ocr_results else getattr(ocr_provider, "model", OCR_MODEL)
        ocr_longest_edge = target_longest_edge

    return {
        "doc_id": file_sha256,
        "file_sha256": file_sha256,
        "filename": rel_name,
        "source_uri": source_uri,
        "file_size_bytes": file_size_bytes,
        "page_count": page_count,
        "pdf_type": pdf_type,
        "extraction_method": extraction_method,
        "ocr_model": ocr_model,
        "ocr_longest_edge": ocr_longest_edge,
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "processing_time_ms": processing_time_ms,
        "ingested_at": datetime.now(timezone.utc),
    }


def run_bronze_ingestion(
    raw_pdfs_dir: Path = RAW_PDFS_DIR,
    bronze_parquet_path: Path = BRONZE_PARQUET_PATH,
    errors_log_path: Path = BRONZE_ERRORS_LOG_PATH,
    ocr_provider: OCRProvider | None = None,
    limit: int | None = None,
    simulate_ocr_error: bool = False,
) -> BronzeRunStats:
    """Exécute le pipeline d'ingestion incrémentale vers la couche Bronze."""
    if limit is not None and limit < 0:
        raise ValueError("L'option --limit doit être un entier positif ou nul.")

    if not raw_pdfs_dir.exists():
        raise FileNotFoundError(
            f"Le répertoire {raw_pdfs_dir} est introuvable. Exécutez d'abord l'Étape 1."
        )

    pdf_files = sorted(raw_pdfs_dir.rglob("*.pdf"))
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    bronze_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    existing_table, existing_hashes = load_existing_bronze_table(bronze_parquet_path)

    if ocr_provider is None:
        if simulate_ocr_error:
            ocr_provider = SimulatedErrorOCRProvider()
        elif has_valid_ocr_credentials():
            ocr_provider = LightOnOCRProvider()
        else:
            console.print(
                Panel(
                    "[bold yellow]Clé API Hyperfluid non configurée dans .env.[/bold yellow]\n"
                    "Les PDF natifs ('texte') seront ingérés normalement, mais les PDF scannés "
                    "('image') échoueront et seront tracés dans [cyan]ingestion_errors.jsonl[/cyan].",
                    title="Avertissement OCR",
                    border_style="yellow",
                )
            )
            ocr_provider = MissingCredentialsOCRProvider()

    stats = BronzeRunStats(
        total_local_pdfs=len(pdf_files),
        total_bronze_docs=len(existing_hashes),
    )

    attempted_new = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Ingestion couche Bronze...", total=len(pdf_files))

        for pdf_path in pdf_files:
            rel_name = pdf_path.relative_to(raw_pdfs_dir).as_posix()
            file_sha256 = compute_sha256(pdf_path)

            if file_sha256 in existing_hashes:
                stats.skipped_existing += 1
                progress.advance(task_id)
                continue

            if limit is not None and attempted_new >= limit:
                progress.advance(task_id)
                continue

            attempted_new += 1
            progress.update(task_id, description=f"Bronze : {rel_name}")

            try:
                record = extract_bronze_record(
                    pdf_path=pdf_path,
                    raw_pdfs_dir=raw_pdfs_dir,
                    file_sha256=file_sha256,
                    ocr_provider=ocr_provider,
                )
                existing_table = append_record_to_parquet(
                    record=record,
                    parquet_path=bronze_parquet_path,
                    existing_table=existing_table,
                )
                existing_hashes.add(file_sha256)

                if record["pdf_type"] == "texte":
                    stats.extracted_native += 1
                else:
                    stats.extracted_ocr += 1

                stats.total_processing_time_ms += record["processing_time_ms"]
                stats.total_bronze_docs = existing_table.num_rows
            except Exception as exc:  # noqa: BLE001
                stats.errors_captured += 1
                log_ingestion_error(
                    errors_log_path=errors_log_path,
                    filename=rel_name,
                    file_sha256=file_sha256,
                    exc=exc,
                )
                console.print(
                    f"[bold yellow]Avertissement :[/bold yellow] échec pour [cyan]{rel_name}[/cyan] "
                    f"({type(exc).__name__}: {exc}) — tracé dans {errors_log_path.name}"
                )

            progress.advance(task_id)

    if existing_table is not None:
        stats.total_bronze_docs = existing_table.num_rows

    return stats


def display_bronze_summary(
    stats: BronzeRunStats,
    parquet_path: Path,
    errors_log_path: Path,
) -> None:
    """Affiche le tableau récapitulatif de l'exécution Bronze avec Rich."""
    table = Table(
        title="Résumé d'ingestion — Couche Bronze (Étape 3)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Indicateur", style="white")
    table.add_column("Valeur", justify="right", style="bold")

    table.add_row("PDF locaux analysés", str(stats.total_local_pdfs))
    table.add_row("Déjà présents en Bronze (ignorés / SHA-256)", f"[dim]{stats.skipped_existing}[/dim]")
    table.add_row("Nouveaux extraits en natif ('texte')", f"[green]{stats.extracted_native}[/green]")
    table.add_row("Nouveaux extraits via OCR ('image')", f"[yellow]{stats.extracted_ocr}[/yellow]")
    table.add_row(
        "Erreurs capturées (à retraiter)",
        f"[red]{stats.errors_captured}[/red]" if stats.errors_captured else "0",
    )
    table.add_row("Total documents en Bronze", f"[cyan]{stats.total_bronze_docs}[/cyan]")

    total_s = stats.total_processing_time_ms / 1000.0
    avg_ms = stats.avg_processing_time_ms
    table.add_row(
        "Temps de traitement (nouveaux docs)",
        f"{total_s:.2f} s (moy. {avg_ms:.1f} ms/doc)",
    )

    console.print(table)
    console.print(
        Panel(
            f"[bold]Parquet Bronze :[/bold] [cyan]{parquet_path}[/cyan]\n"
            f"[bold]Journal erreurs :[/bold] [cyan]{errors_log_path}[/cyan]",
            border_style="blue",
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI de l'étape 3."""
    parser = argparse.ArgumentParser(
        description="Étape 3 : Ingestion incrémentale des factures vers la couche Bronze (Parquet).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite le nombre de nouveaux fichiers à traiter lors de cette exécution.",
    )
    parser.add_argument(
        "--simulate-ocr-error",
        action="store_true",
        help="Simule une indisponibilité de l'API OCR pour tester la tolérance aux pannes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée principal de l'étape 3."""
    args = parse_args(argv)

    try:
        stats = run_bronze_ingestion(
            limit=args.limit,
            simulate_ocr_error=args.simulate_ocr_error,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[bold red]Erreur :[/bold red] {exc}")
        return 1

    display_bronze_summary(
        stats=stats,
        parquet_path=BRONZE_PARQUET_PATH,
        errors_log_path=BRONZE_ERRORS_LOG_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
