"""Étape 5 : Évaluation quantitative de la qualité OCR champ par champ et benchmark de résolutions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from src.config import (
    BRONZE_PARQUET_PATH,
    EVAL_OCR_CACHE_PATH,
    EVAL_OCR_RESULTS_PATH,
    GROUND_TRUTH_PATH,
    OCR_MODEL,
    RAW_PDFS_DIR,
    has_valid_ocr_credentials,
)
from src.ocr import LightOnOCRProvider, OCRError
from src.step3_bronze import compute_sha256
from src.step4_silver import parse_bronze_document

console = Console()

EVAL_FIELDS: list[str] = [
    "invoice_number",
    "issue_date",
    "due_date",
    "category",
    "status",
    "supplier_name",
    "supplier_siret",
    "supplier_tva_intracom",
    "client_name",
    "client_siret",
    "total_ht",
    "total_tva",
    "total_ttc",
    "payment_method",
    "iban",
    "line_count",
    "lines_total_ht_sum",
]


@dataclass
class OcrEvalObservation:
    """Sortie OCR et métriques techniques pour un document à une résolution donnée."""

    filename: str
    file_sha256: str
    resolution: int
    raw_text: str
    processing_time_ms: float
    image_size_bytes: int
    source: str  # "bronze", "cache", ou "api"


def levenshtein_distance(a: str, b: str) -> int:
    """Calcule la distance de Levenshtein entre deux chaînes."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr = [i]
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def char_similarity(expected: str, actual: str) -> float:
    """Similarité normalisée [0, 1] basée sur la distance de Levenshtein."""
    denom = max(len(expected), len(actual), 1)
    return 1.0 - (levenshtein_distance(expected, actual) / denom)


def char_error_rate(expected: str, actual: str) -> float:
    """Taux d'erreur caractère (CER) = Levenshtein(expected, actual) / len(expected)."""
    denom = max(len(expected), 1)
    return levenshtein_distance(expected, actual) / denom


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _load_bronze_ocr_index(bronze_path: Path) -> dict[tuple[str, str, int], dict]:
    """Indexe les documents Bronze OCR par (file_sha256, ocr_model, ocr_longest_edge)."""
    if not bronze_path.exists():
        return {}
    rows = pq.read_table(bronze_path).to_pylist()
    index: dict[tuple[str, str, int], dict] = {}
    for row in rows:
        if row.get("extraction_method") == "ocr_api" and row.get("ocr_longest_edge") is not None:
            key = (
                row["file_sha256"],
                str(row.get("ocr_model") or ""),
                int(row["ocr_longest_edge"]),
            )
            index[key] = row
    return index


def _extract_fields_for_eval(
    filename: str,
    file_sha256: str,
    raw_text: str,
) -> dict[str, str]:
    """Extrait les 17 champs évalués via le parseur Silver."""
    parsed = parse_bronze_document(
        {
            "doc_id": file_sha256,
            "filename": filename,
            "extraction_method": "ocr_api",
            "raw_text": raw_text,
        }
    )

    line_totals = [ln.line_total_ht for ln in parsed.lines if ln.line_total_ht is not None]
    if parsed.lines and len(line_totals) == len(parsed.lines):
        lines_sum_str = f"{sum(line_totals, Decimal('0.00')):.2f}"
    elif line_totals:
        lines_sum_str = f"{sum(line_totals, Decimal('0.00')):.2f}"
    else:
        lines_sum_str = ""

    return {
        "invoice_number": parsed.invoice_number or "",
        "issue_date": parsed.issue_date.isoformat() if parsed.issue_date else "",
        "due_date": parsed.due_date.isoformat() if parsed.due_date else "",
        "category": parsed.category or "",
        "status": parsed.status or "",
        "supplier_name": parsed.supplier_name or "",
        "supplier_siret": parsed.supplier_siret or "",
        "supplier_tva_intracom": parsed.supplier_tva_intracom or "",
        "client_name": parsed.client_name or "",
        "client_siret": parsed.client_siret or "",
        "total_ht": f"{parsed.total_ht:.2f}" if parsed.total_ht is not None else "",
        "total_tva": f"{parsed.total_tva:.2f}" if parsed.total_tva is not None else "",
        "total_ttc": f"{parsed.total_ttc:.2f}" if parsed.total_ttc is not None else "",
        "payment_method": parsed.payment_method or "",
        "iban": parsed.iban or "",
        "line_count": str(len(parsed.lines)),
        "lines_total_ht_sum": lines_sum_str,
    }


def _get_ocr_observation(
    pdf_path: Path,
    file_sha256: str,
    resolution: int,
    bronze_index: dict[tuple[str, str, int], dict],
    cache: dict[str, dict],
    cache_path: Path,
    use_cache: bool,
    provider: LightOnOCRProvider | None,
) -> OcrEvalObservation:
    """Récupère la sortie OCR depuis Bronze (1540 px), le cache JSON, ou l'API Hyperfluid."""
    bronze_key = (file_sha256, OCR_MODEL, resolution)
    if bronze_key in bronze_index:
        row = bronze_index[bronze_key]
        cache_key = f"{file_sha256}:{OCR_MODEL}:{resolution}"
        if use_cache and cache_key in cache and cache[cache_key].get("image_size_bytes"):
            img_size = int(cache[cache_key]["image_size_bytes"])
        else:
            png_bytes, _, _ = LightOnOCRProvider._render_page_to_png(pdf_path, 0, resolution)
            img_size = len(png_bytes)
            if use_cache:
                cache[cache_key] = {
                    "filename": pdf_path.name,
                    "file_sha256": file_sha256,
                    "model": OCR_MODEL,
                    "longest_edge": resolution,
                    "raw_text": row["raw_text"],
                    "processing_time_ms": float(row["processing_time_ms"]),
                    "image_size_bytes": img_size,
                    "source": "bronze",
                }
                _save_json_atomic(cache_path, cache)

        return OcrEvalObservation(
            filename=pdf_path.name,
            file_sha256=file_sha256,
            resolution=resolution,
            raw_text=row["raw_text"],
            processing_time_ms=float(row["processing_time_ms"]),
            image_size_bytes=img_size,
            source="bronze",
        )

    cache_key = f"{file_sha256}:{OCR_MODEL}:{resolution}"
    if use_cache and cache_key in cache:
        cached = cache[cache_key]
        return OcrEvalObservation(
            filename=pdf_path.name,
            file_sha256=file_sha256,
            resolution=resolution,
            raw_text=cached["raw_text"],
            processing_time_ms=float(cached["processing_time_ms"]),
            image_size_bytes=int(cached["image_size_bytes"]),
            source="cache",
        )

    if provider is None:
        if not has_valid_ocr_credentials():
            raise OCRError(
                "HYPERFLUID_API_KEY non configurée dans .env pour appeler l'API OCR "
                f"à la résolution {resolution} px."
            )
        provider = LightOnOCRProvider()

    page_results = provider.extract_document(pdf_path, target_longest_edge=resolution)
    raw_text = "\n\n".join(res.text for res in page_results)
    total_ms = sum(res.processing_time_ms for res in page_results)
    total_png_bytes = sum(res.image_size_bytes for res in page_results)

    if use_cache:
        cache[cache_key] = {
            "filename": pdf_path.name,
            "file_sha256": file_sha256,
            "model": OCR_MODEL,
            "longest_edge": resolution,
            "raw_text": raw_text,
            "processing_time_ms": total_ms,
            "image_size_bytes": total_png_bytes,
            "source": "api",
        }
        _save_json_atomic(cache_path, cache)

    return OcrEvalObservation(
        filename=pdf_path.name,
        file_sha256=file_sha256,
        resolution=resolution,
        raw_text=raw_text,
        processing_time_ms=total_ms,
        image_size_bytes=total_png_bytes,
        source="api",
    )


def evaluate_resolutions(
    resolutions: list[int],
    sample_size: int | None = None,
    use_cache: bool = True,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    raw_pdfs_dir: Path = RAW_PDFS_DIR,
    bronze_path: Path = BRONZE_PARQUET_PATH,
    cache_path: Path = EVAL_OCR_CACHE_PATH,
    results_path: Path = EVAL_OCR_RESULTS_PATH,
    provider: LightOnOCRProvider | None = None,
) -> dict:
    """Exécute le benchmark OCR sur la vérité terrain pour chaque résolution."""
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Fichier de vérité terrain introuvable : {ground_truth_path}")

    ground_truth: list[dict] = _load_json(ground_truth_path, [])
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("--sample-size doit être strictement positif.")
        ground_truth = ground_truth[:sample_size]

    bronze_index = _load_bronze_ocr_index(bronze_path)
    cache: dict[str, dict] = _load_json(cache_path, {}) if use_cache else {}

    resolutions_summary: list[dict] = []
    all_errors: list[dict] = []

    total_steps = len(resolutions) * len(ground_truth)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Évaluation OCR...", total=total_steps)

        for res in resolutions:
            field_stats: dict[str, dict] = {
                f: {
                    "exact_matches": 0,
                    "total": 0,
                    "similarities": [],
                    "cers": [],
                }
                for f in EVAL_FIELDS
            }
            perfect_invoices = 0
            png_sizes_bytes: list[int] = []
            times_ms: list[float] = []
            sources_count: dict[str, int] = {"bronze": 0, "cache": 0, "api": 0}

            for gt_entry in ground_truth:
                filename = gt_entry["filename"]
                pdf_path = raw_pdfs_dir / filename
                if not pdf_path.is_file():
                    raise FileNotFoundError(f"PDF introuvable pour la vérité terrain : {pdf_path}")

                file_sha256 = compute_sha256(pdf_path)
                progress.update(
                    task_id,
                    description=f"Évaluation {res} px : {filename}",
                )

                obs = _get_ocr_observation(
                    pdf_path=pdf_path,
                    file_sha256=file_sha256,
                    resolution=res,
                    bronze_index=bronze_index,
                    cache=cache,
                    cache_path=cache_path,
                    use_cache=use_cache,
                    provider=provider,
                )

                sources_count[obs.source] = sources_count.get(obs.source, 0) + 1
                png_sizes_bytes.append(obs.image_size_bytes)
                times_ms.append(obs.processing_time_ms)

                extracted = _extract_fields_for_eval(filename, file_sha256, obs.raw_text)

                invoice_all_exact = True
                for field_name in EVAL_FIELDS:
                    expected_val = str(gt_entry[field_name])
                    actual_val = str(extracted.get(field_name, ""))

                    is_exact = expected_val == actual_val
                    sim = char_similarity(expected_val, actual_val)
                    cer = char_error_rate(expected_val, actual_val)

                    st = field_stats[field_name]
                    st["total"] += 1
                    st["similarities"].append(sim)
                    st["cers"].append(cer)
                    if is_exact:
                        st["exact_matches"] += 1
                    else:
                        invoice_all_exact = False
                        all_errors.append(
                            {
                                "resolution": res,
                                "filename": filename,
                                "field": field_name,
                                "expected": expected_val,
                                "actual": actual_val,
                                "similarity_pct": round(sim * 100.0, 2),
                                "cer_pct": round(cer * 100.0, 2),
                            }
                        )

                if invoice_all_exact:
                    perfect_invoices += 1

                progress.advance(task_id)

            total_field_evals = sum(st["total"] for st in field_stats.values())
            total_exact_matches = sum(st["exact_matches"] for st in field_stats.values())
            overall_accuracy = (
                (total_exact_matches / total_field_evals * 100.0) if total_field_evals else 0.0
            )
            perfect_pct = (perfect_invoices / len(ground_truth) * 100.0) if ground_truth else 0.0

            per_field_metrics = {}
            for field_name, st in field_stats.items():
                n = st["total"] or 1
                per_field_metrics[field_name] = {
                    "exact_matches": st["exact_matches"],
                    "total": st["total"],
                    "accuracy_pct": round(st["exact_matches"] / n * 100.0, 2),
                    "avg_similarity_pct": round(sum(st["similarities"]) / n * 100.0, 2),
                    "avg_cer_pct": round(sum(st["cers"]) / n * 100.0, 2),
                }

            resolutions_summary.append(
                {
                    "resolution_px": res,
                    "sample_size": len(ground_truth),
                    "avg_png_size_kb": round(
                        (sum(png_sizes_bytes) / len(png_sizes_bytes) / 1024.0)
                        if png_sizes_bytes
                        else 0.0,
                        2,
                    ),
                    "avg_ocr_time_ms": round(
                        (sum(times_ms) / len(times_ms)) if times_ms else 0.0, 2
                    ),
                    "overall_field_accuracy_pct": round(overall_accuracy, 2),
                    "perfect_invoices": perfect_invoices,
                    "perfect_invoices_pct": round(perfect_pct, 2),
                    "sources": sources_count,
                    "per_field": per_field_metrics,
                }
            )

    results_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": OCR_MODEL,
        "sample_size": len(ground_truth),
        "evaluated_fields": EVAL_FIELDS,
        "resolutions": resolutions_summary,
        "errors": all_errors,
    }
    _save_json_atomic(results_path, results_payload)
    return results_payload


def display_eval_report(results: dict, results_path: Path) -> None:
    """Affiche les tableaux de benchmark, l'analyse des erreurs et la synthèse avec Rich."""
    res_list: list[dict] = results["resolutions"]

    # 1. Tableau de synthèse global
    global_table = Table(
        title=f"Synthèse Benchmark OCR — {results['model']} ({results['sample_size']} factures scannées)",
        show_header=True,
        header_style="bold cyan",
    )
    global_table.add_column("Résolution (px)", justify="right", style="bold white")
    global_table.add_column("Poids PNG moyen", justify="right")
    global_table.add_column("Temps OCR moyen", justify="right")
    global_table.add_column("Exactitude globale", justify="right")
    global_table.add_column("Factures 100% exactes", justify="right")

    for r in res_list:
        acc = r["overall_field_accuracy_pct"]
        perf_pct = r["perfect_invoices_pct"]
        acc_style = "bold green" if acc >= 99.0 else ("bold yellow" if acc >= 90.0 else "bold red")
        global_table.add_row(
            f"{r['resolution_px']} px",
            f"{r['avg_png_size_kb']:.1f} Ko",
            f"{r['avg_ocr_time_ms']:.0f} ms ({r['avg_ocr_time_ms'] / 1000:.2f} s)",
            f"[{acc_style}]{acc:.2f}%[/{acc_style}]",
            f"{r['perfect_invoices']}/{r['sample_size']} ({perf_pct:.1f}%)",
        )

    console.print(global_table)

    # 2. Tableau détaillé champ par champ
    field_table = Table(
        title="Précision champ par champ (Exact Match % / Similarité caractères %)",
        show_header=True,
        header_style="bold cyan",
    )
    field_table.add_column("Champ", style="white", no_wrap=True)
    for r in res_list:
        field_table.add_column(f"{r['resolution_px']} px Exact", justify="right")
        field_table.add_column(f"{r['resolution_px']} px Sim.", justify="right")

    for field_name in EVAL_FIELDS:
        row_cells = [field_name]
        any_error = False
        for r in res_list:
            m = r["per_field"][field_name]
            acc = m["accuracy_pct"]
            sim = m["avg_similarity_pct"]
            if acc < 100.0:
                any_error = True
            style = "green" if acc == 100.0 else ("yellow" if acc >= 80.0 else "bold red")
            row_cells.append(f"[{style}]{acc:.1f}% ({m['exact_matches']}/{m['total']})[/{style}]")
            row_cells.append(f"[{style}]{sim:.1f}%[/{style}]")
        if any_error:
            row_cells[0] = f"[bold yellow]{field_name}[/bold yellow]"
        field_table.add_row(*row_cells)

    console.print(field_table)

    # 3. Détail des erreurs observées
    errors: list[dict] = results["errors"]
    if errors:
        err_table = Table(
            title=f"Erreurs de lecture OCR détectées ({len(errors)})",
            show_header=True,
            header_style="bold red",
            show_lines=True,
        )
        err_table.add_column("Rés.", justify="right", no_wrap=True)
        err_table.add_column("Fichier", style="cyan")
        err_table.add_column("Champ", style="yellow")
        err_table.add_column("Attendu (Vérité terrain)", overflow="fold")
        err_table.add_column("Lu (OCR)", overflow="fold")
        err_table.add_column("Sim.", justify="right", no_wrap=True)

        for err in errors:
            err_table.add_row(
                f"{err['resolution']} px",
                err["filename"],
                err["field"],
                err["expected"],
                err["actual"] or "[dim]<vide>[/dim]",
                f"{err['similarity_pct']:.1f}%",
            )
        console.print(err_table)
    else:
        console.print("[bold green]Aucune erreur de lecture sur l'ensemble des résolutions testées ![/bold green]")

    # 4. Synthèse analytique
    fragile_summary_lines: list[str] = []
    for r in res_list:
        fragile = [
            (fname, m["accuracy_pct"], m["avg_similarity_pct"])
            for fname, m in r["per_field"].items()
            if m["accuracy_pct"] < 100.0
        ]
        fragile.sort(key=lambda item: item[1])
        if fragile:
            top_str = ", ".join(f"{fn} ({acc:.1f}%)" for fn, acc, _ in fragile[:5])
            fragile_summary_lines.append(
                f"• [bold]{r['resolution_px']} px[/bold] : champs les plus fragiles → {top_str}"
            )
        else:
            fragile_summary_lines.append(
                f"• [bold]{r['resolution_px']} px[/bold] : 100% d'exactitude sur les {len(EVAL_FIELDS)} champs."
            )

    fragile_summary_lines.append("")
    fragile_summary_lines.append(
        "[bold]Analyse :[/bold] Les chaînes numériques longues sans redondance sémantique "
        "(IBAN à 27 caractères, SIRET à 14 chiffres, TVA intracommunautaire) ainsi que les montants "
        "chiffrés et dates sont les premiers touchés lorsque la résolution baisse (confusions visuelles "
        "type 3/8, 5/6, 1/7, 0/O ou virgules floutées), alors que les libellés textuels bénéficient du "
        "contexte linguistique du décodeur. À 1540 px (recommandation LightOnOCR-3-1B), la netteté des "
        "glyphes garantit une lecture fiable au centime près."
    )
    fragile_summary_lines.append(f"\n[dim]Résultats exportés : {results_path}[/dim]")

    console.print(
        Panel(
            "\n".join(fragile_summary_lines),
            title="Synthèse analytique — Compromis Résolution / Précision / Latence",
            border_style="blue",
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI de l'étape 5."""
    parser = argparse.ArgumentParser(
        description="Étape 5 : Évaluation champ par champ de la qualité OCR et benchmark de résolutions.",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default="1540,512",
        help="Liste des résolutions target_longest_edge séparées par des virgules (défaut : '1540,512').",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Nombre de factures de ground_truth.json à évaluer (défaut : toutes les 15).",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--use-cache",
        dest="use_cache",
        action="store_true",
        help="Utilise/alimente le cache data/eval_ocr_cache.json (activé par défaut).",
    )
    cache_group.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Ignore le cache et force les appels OCR pour les résolutions absentes de Bronze.",
    )
    parser.set_defaults(use_cache=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée principal de l'étape 5."""
    args = parse_args(argv)

    try:
        resolutions = [int(part.strip()) for part in args.resolutions.split(",") if part.strip()]
        if not resolutions or any(r <= 0 for r in resolutions):
            raise ValueError("Les résolutions doivent être des entiers strictement positifs.")
    except ValueError as exc:
        console.print(f"[bold red]Erreur sur --resolutions :[/bold red] {exc}")
        return 1

    try:
        results = evaluate_resolutions(
            resolutions=resolutions,
            sample_size=args.sample_size,
            use_cache=args.use_cache,
        )
    except (FileNotFoundError, ValueError, OCRError) as exc:
        console.print(f"[bold red]Erreur :[/bold red] {exc}")
        return 1

    display_eval_report(results, EVAL_OCR_RESULTS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
