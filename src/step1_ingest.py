"""Étape 1 : Récupération incrémentale des factures PDF depuis S3 et détection texte vs image."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pymupdf
from botocore.exceptions import BotoCoreError, ClientError
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from src.config import (
    AWS_ACCESS_KEY_ID,
    AWS_DEFAULT_REGION,
    AWS_ENDPOINT_URL,
    AWS_SECRET_ACCESS_KEY,
    ENV_FILE,
    MANIFEST_STEP1_PATH,
    RAW_PDFS_DIR,
    S3_BUCKET_NAME,
    TEXT_CHAR_THRESHOLD,
    has_valid_s3_credentials,
)

console = Console()


@dataclass
class SyncStats:
    """Statistiques de synchronisation S3 -> local."""

    total_remote_pdfs: int
    targeted_pdfs: int
    downloaded: int
    skipped_existing: int


@dataclass
class PdfInspectionResult:
    """Résultat de l'inspection PyMuPDF d'un fichier PDF."""

    filename: str
    relative_path: str
    size_bytes: int
    page_count: int
    text_char_count: int
    image_count: int
    pdf_type: str  # "texte", "image", ou "erreur"
    error: str | None = None


def create_s3_client():
    """Instancie le client boto3 configuré pour le endpoint S3."""
    return boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )


def list_remote_pdfs(s3_client, bucket_name: str) -> list[dict]:
    """Liste tous les objets .pdf présents dans le bucket S3."""
    paginator = s3_client.get_paginator("list_objects_v2")
    pdf_objects: list[dict] = []

    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key.lower().endswith(".pdf") and not key.endswith("/"):
                pdf_objects.append(obj)

    pdf_objects.sort(key=lambda item: item["Key"])
    return pdf_objects


def sync_pdfs_from_s3(
    dest_dir: Path,
    bucket_name: str,
    limit: int | None = None,
) -> SyncStats:
    """Télécharge de façon incrémentale les PDF du bucket S3 vers dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    s3_client = create_s3_client()

    remote_pdfs = list_remote_pdfs(s3_client, bucket_name)
    total_remote = len(remote_pdfs)

    if limit is not None:
        if limit < 0:
            raise ValueError("L'option --limit doit être un entier positif ou nul.")
        target_pdfs = remote_pdfs[:limit]
    else:
        target_pdfs = remote_pdfs

    downloaded = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(
            f"Synchronisation depuis s3://{bucket_name}...",
            total=len(target_pdfs),
        )

        for obj in target_pdfs:
            key: str = obj["Key"]
            remote_size: int = int(obj["Size"])
            local_path = dest_dir / key
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if local_path.exists() and local_path.stat().st_size == remote_size:
                skipped += 1
            else:
                s3_client.download_file(bucket_name, key, str(local_path))
                downloaded += 1

            progress.advance(task_id)

    return SyncStats(
        total_remote_pdfs=total_remote,
        targeted_pdfs=len(target_pdfs),
        downloaded=downloaded,
        skipped_existing=skipped,
    )


def inspect_pdf(pdf_path: Path, base_dir: Path, char_threshold: int = TEXT_CHAR_THRESHOLD) -> PdfInspectionResult:
    """Inspecte un PDF local via PyMuPDF et détermine s'il est de type 'texte' ou 'image'."""
    rel_path = pdf_path.relative_to(base_dir).as_posix()
    size_bytes = pdf_path.stat().st_size

    try:
        with pymupdf.open(pdf_path) as doc:
            page_count = len(doc)
            total_text_chars = 0
            total_images = 0

            for page in doc:
                raw_text = page.get_text("text") or ""
                significant_chars = "".join(raw_text.split())
                total_text_chars += len(significant_chars)
                total_images += len(page.get_images(full=True))

        pdf_type = "texte" if total_text_chars >= char_threshold else "image"
        return PdfInspectionResult(
            filename=pdf_path.name,
            relative_path=rel_path,
            size_bytes=size_bytes,
            page_count=page_count,
            text_char_count=total_text_chars,
            image_count=total_images,
            pdf_type=pdf_type,
        )
    except Exception as exc:  # noqa: BLE001
        return PdfInspectionResult(
            filename=pdf_path.name,
            relative_path=rel_path,
            size_bytes=size_bytes,
            page_count=0,
            text_char_count=0,
            image_count=0,
            pdf_type="erreur",
            error=str(exc),
        )


def inspect_local_pdfs(raw_pdfs_dir: Path) -> list[PdfInspectionResult]:
    """Inspecte tous les fichiers PDF locaux présents dans raw_pdfs_dir."""
    if not raw_pdfs_dir.exists():
        return []

    pdf_files = sorted(raw_pdfs_dir.rglob("*.pdf"))
    results: list[PdfInspectionResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Inspection PyMuPDF des PDF...", total=len(pdf_files))
        for pdf_path in pdf_files:
            results.append(inspect_pdf(pdf_path, raw_pdfs_dir))
            progress.advance(task_id)

    return results


def save_manifest(
    results: list[PdfInspectionResult],
    manifest_path: Path,
    bucket_name: str,
    raw_pdfs_dir: Path,
) -> dict:
    """Sauvegarde les résultats d'inspection au format JSON."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    text_count = sum(1 for r in results if r.pdf_type == "texte")
    image_count = sum(1 for r in results if r.pdf_type == "image")
    error_count = sum(1 for r in results if r.pdf_type == "erreur")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket_name,
        "raw_pdfs_dir": raw_pdfs_dir.as_posix(),
        "text_char_threshold": TEXT_CHAR_THRESHOLD,
        "summary": {
            "total_pdfs": len(results),
            "text_pdfs": text_count,
            "image_pdfs": image_count,
            "error_pdfs": error_count,
        },
        "files": [asdict(r) for r in results],
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


def display_report(
    sync_stats: SyncStats,
    results: list[PdfInspectionResult],
    manifest_path: Path,
) -> None:
    """Affiche le tableau récapitulatif et les statistiques avec Rich."""
    table = Table(
        title="Inspection des factures PDF (Étape 1)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Fichier", style="white", no_wrap=True)
    table.add_column("Taille (Ko)", justify="right")
    table.add_column("Pages", justify="right")
    table.add_column("Caractères (hors espaces)", justify="right")
    table.add_column("Images", justify="right")
    table.add_column("Type détecté", justify="center")

    for res in results:
        size_kb = f"{res.size_bytes / 1024:.1f}"
        if res.pdf_type == "texte":
            type_badge = "[bold green]texte[/bold green]"
        elif res.pdf_type == "image":
            type_badge = "[bold yellow]image[/bold yellow]"
        else:
            type_badge = "[bold red]erreur[/bold red]"

        table.add_row(
            res.relative_path,
            size_kb,
            str(res.page_count),
            str(res.text_char_count),
            str(res.image_count),
            type_badge,
        )

    console.print(table)

    total = len(results)
    text_count = sum(1 for r in results if r.pdf_type == "texte")
    image_count = sum(1 for r in results if r.pdf_type == "image")
    error_count = sum(1 for r in results if r.pdf_type == "erreur")

    text_pct = (text_count / total * 100) if total else 0.0
    image_pct = (image_count / total * 100) if total else 0.0

    summary_lines = [
        f"[bold]Bucket S3 :[/bold] s3://{S3_BUCKET_NAME} ({sync_stats.total_remote_pdfs} PDF distants)",
        (
            f"[bold]Synchronisation :[/bold] {sync_stats.downloaded} téléchargé(s), "
            f"{sync_stats.skipped_existing} déjà à jour (sur {sync_stats.targeted_pdfs} ciblé(s))"
        ),
        "",
        f"[bold]Total PDF locaux inspectés :[/bold] {total}",
        f"[green]• PDF 'texte' (natifs, >= {TEXT_CHAR_THRESHOLD} car.) :[/green] {text_count} ({text_pct:.1f}%)",
        f"[yellow]• PDF 'image' (scannés, OCR requis) :[/yellow] {image_count} ({image_pct:.1f}%)",
    ]
    if error_count > 0:
        summary_lines.append(f"[red]• PDF en erreur :[/red] {error_count}")

    summary_lines.append("")
    summary_lines.append(f"[dim]Manifeste sauvegardé : {manifest_path}[/dim]")

    console.print(
        Panel(
            "\n".join(summary_lines),
            title="Résumé Étape 1",
            border_style="blue",
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Étape 1 : Téléchargement incrémental depuis S3 et détection texte vs image des factures PDF.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite le nombre de fichiers PDF traités/téléchargés depuis S3 (sans limite par défaut).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée principal de l'étape 1."""
    args = parse_args(argv)

    if not has_valid_s3_credentials():
        console.print(
            Panel(
                "[bold red]Identifiants S3 manquants ou non configurés.[/bold red]\n\n"
                f"Le fichier [cyan]{ENV_FILE}[/cyan] contient encore les valeurs par défaut "
                "([yellow]your_access_key_id[/yellow] / [yellow]your_secret_access_key[/yellow]).\n"
                "Veuillez renseigner vos clés d'accès S3 dans [cyan].env[/cyan], puis relancer :\n"
                "  [bold]uv run python -m src.step1_ingest[/bold]",
                title="Configuration requise",
                border_style="red",
            )
        )
        return 1

    try:
        sync_stats = sync_pdfs_from_s3(
            dest_dir=RAW_PDFS_DIR,
            bucket_name=S3_BUCKET_NAME,
            limit=args.limit,
        )
    except ValueError as exc:
        console.print(f"[bold red]Erreur d'argument :[/bold red] {exc}")
        return 1
    except (ClientError, BotoCoreError) as exc:
        console.print(f"[bold red]Erreur lors de l'accès au bucket S3 :[/bold red] {exc}")
        return 1

    results = inspect_local_pdfs(RAW_PDFS_DIR)
    save_manifest(
        results=results,
        manifest_path=MANIFEST_STEP1_PATH,
        bucket_name=S3_BUCKET_NAME,
        raw_pdfs_dir=RAW_PDFS_DIR,
    )
    display_report(sync_stats, results, MANIFEST_STEP1_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
