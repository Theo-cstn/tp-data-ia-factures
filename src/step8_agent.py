"""Étape 8 : Agent IA multi-LLM en français relié au serveur MCP avec transparence SQL."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from contextlib import AsyncExitStack
from decimal import Decimal, InvalidOperation

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src.agent.chat import InvoiceAgentChat, ToolTraceEntry, TurnTrace
from src.agent.llm import LLMError, create_llm_provider
from src.config import (
    ENV_FILE,
    has_valid_llm_credentials,
    resolve_llm_config,
)
from src.mcp_client import MCPClient
from src.step6_warehouse import get_warehouse_connection

console = Console()

VALIDATION_QUESTIONS = [
    "Quel est le montant TTC des factures en retard en 2025 ?",
    "Et peux-tu me donner le détail de la facture V-2025-0001 ?",
]


def _render_tool_trace(step_idx: int, entry: ToolTraceEntry) -> None:
    """Affiche [1] Outil & args, [2] SQL exécuté, [3] Extrait résultat JSON."""
    args_json = json.dumps(entry.arguments, ensure_ascii=False, indent=2)
    console.print(
        Panel(
            Syntax(args_json, "json", theme="monokai", word_wrap=True),
            title=f"[bold magenta][1] Outil MCP #{step_idx} : {entry.tool_name}[/bold magenta]",
            border_style="magenta",
        )
    )

    if entry.executed_sql:
        sql_blocks: list[str] = []
        for idx, sql in enumerate(entry.executed_sql):
            params = entry.sql_params[idx] if idx < len(entry.sql_params) else []
            if params:
                sql_blocks.append(f"-- Paramètres liés (?) : {json.dumps(params, ensure_ascii=False)}\n{sql}")
            else:
                sql_blocks.append(sql)
        sql_text = "\n\n".join(sql_blocks)
        console.print(
            Panel(
                Syntax(sql_text, "sql", theme="monokai", word_wrap=True),
                title="[bold cyan][2] Requête(s) SQL exécutée(s) (executed_sql)[/bold cyan]",
                border_style="cyan",
            )
        )

    excerpt = _summarize_tool_result(entry.raw_result)
    border = "red" if entry.is_error else "yellow"
    console.print(
        Panel(
            Syntax(excerpt, "json", theme="monokai", word_wrap=True),
            title="[bold yellow][3] Extrait résultat JSON retourné par MCP[/bold yellow]",
            border_style=border,
        )
    )


def _summarize_tool_result(raw_json: str, max_chars: int = 1200) -> str:
    """Formate un extrait lisible du JSON retourné par le serveur MCP."""
    try:
        parsed = json.loads(raw_json)
        # Évite de répéter executed_sql déjà affiché dans le bloc [2]
        compact = {k: v for k, v in parsed.items() if k not in {"executed_sql", "sql_params"}}
        formatted = json.dumps(compact, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        formatted = raw_json

    if len(formatted) > max_chars:
        return formatted[:max_chars] + "\n... [tronqué pour lisibilité]"
    return formatted


def _render_final_answer(trace: TurnTrace) -> None:
    """Affiche [4] la réponse finale en français avec le modèle et la latence."""
    elapsed_s = trace.elapsed_ms / 1000.0
    console.print(
        Panel(
            Markdown(trace.answer or "_Aucune réponse textuelle produite._"),
            title=(
                f"[bold green][4] Réponse de l'Agent ({trace.provider}/{trace.model} "
                f"— {elapsed_s:.2f} s)[/bold green]"
            ),
            border_style="green",
        )
    )


def _normalize_numeric_tokens(text: str) -> set[str]:
    """Extrait et normalise tous les nombres (FR/EN) présents dans un texte."""
    normalized: set[str] = set()
    # Repère les nombres incluant séparateurs de milliers (espace, espace insécable, virgule, point)
    for match in re.finditer(r"\d+(?:[\s\xa0\u202f.,]\d+)*", text):
        raw = re.sub(r"[\s\xa0\u202f]", "", match.group(0))
        if not raw:
            continue
        if "," in raw and "." in raw:
            last_comma = raw.rfind(",")
            last_dot = raw.rfind(".")
            if last_comma > last_dot:
                raw = raw.replace(".", "").replace(",", ".")
            else:
                raw = raw.replace(",", "")
        elif "," in raw:
            parts = raw.split(",")
            if len(parts[-1]) in {1, 2}:
                raw = "".join(parts[:-1]) + "." + parts[-1]
            else:
                raw = raw.replace(",", "")
        elif raw.count(".") > 1:
            parts = raw.split(".")
            if len(parts[-1]) in {1, 2}:
                raw = "".join(parts[:-1]) + "." + parts[-1]
            else:
                raw = raw.replace(".", "")

        normalized.add(raw)
        try:
            dec = Decimal(raw)
            normalized.add(f"{dec:.2f}")
            if dec == dec.to_integral_value():
                normalized.add(str(int(dec)))
        except InvalidOperation:
            pass
    return normalized


def _verify_ground_truth_sql(q1_answer: str, q2_answer: str) -> tuple[bool, bool, str, str]:
    """Vérifie directement dans DuckDB l'exactitude des chiffres annoncés par le LLM."""
    with get_warehouse_connection(read_only=True) as conn:
        q1_count, q1_ttc = conn.execute(
            """
            SELECT COUNT(*)::INTEGER, COALESCE(SUM(total_ttc), 0::DECIMAL(12, 2))
            FROM silver.factures
            WHERE status = 'En retard' AND EXTRACT(YEAR FROM issue_date) = 2025
            """
        ).fetchone()

        q2_row = conn.execute(
            """
            SELECT invoice_number, reported_total_ht, reported_total_tva, reported_total_ttc,
                   calculated_total_ttc, rejection_codes
            FROM silver.quarantaine
            WHERE invoice_number = 'V-2025-0001'
            """
        ).fetchone()

    q1_nums = _normalize_numeric_tokens(q1_answer)
    q1_expected_ttc = f"{q1_ttc:.2f}"
    q1_ok = q1_expected_ttc in q1_nums
    q1_detail = f"SQL attendu : {q1_count} factures / {q1_expected_ttc} € TTC"

    q2_nums = _normalize_numeric_tokens(q2_answer)
    q2_low = q2_answer.lower()
    if q2_row is not None:
        _, rep_ht, _, rep_ttc, calc_ttc, rej_code = q2_row
        q2_has_quarantine = ("quarantaine" in q2_low) or ("rejet" in q2_low) or (
            rej_code.lower() in q2_low
        )
        q2_has_amounts = (f"{rep_ht:.2f}" in q2_nums) and (
            f"{rep_ttc:.2f}" in q2_nums or f"{calc_ttc:.2f}" in q2_nums
        )
        q2_ok = q2_has_quarantine and q2_has_amounts
        q2_detail = f"SQL attendu : quarantaine ({rej_code}, HT {rep_ht:.2f} €, TTC {rep_ttc:.2f} €)"
    else:
        q2_ok = False
        q2_detail = "V-2025-0001 introuvable en quarantaine"

    return q1_ok, q2_ok, q1_detail, q2_detail


async def _run_queries_session(
    queries: list[str],
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> list[TurnTrace]:
    """Lance une session MCP + LLM et exécute une liste de questions en conservant l'historique."""
    llm = create_llm_provider(provider=provider, model=model, base_url=base_url)
    traces: list[TurnTrace] = []

    async with AsyncExitStack() as stack:
        mcp_client = await stack.enter_async_context(
            MCPClient(command="uv", args=["run", "python", "-m", "src.mcp_server"])
        )
        chat = InvoiceAgentChat(llm_provider=llm, clients={"warehouse": mcp_client})

        for query in queries:
            console.rule(f"[bold blue]Question : {query}[/bold blue]")
            step_counter = 0

            def _on_tool(entry: ToolTraceEntry) -> None:
                nonlocal step_counter
                step_counter += 1
                _render_tool_trace(step_counter, entry)

            trace = await chat.run(query, on_tool_executed=_on_tool)
            _render_final_answer(trace)
            traces.append(trace)

    return traces


async def _run_interactive_session(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> int:
    """Mode interactif multi-tours dans le terminal."""
    cfg = resolve_llm_config(provider=provider, model=model, base_url=base_url)
    llm = create_llm_provider(provider=provider, model=model, base_url=base_url)

    console.print(
        Panel(
            f"[bold cyan]Agent IA Factures — Mode Conversation Interactive[/bold cyan]\n"
            f"Fournisseur : [bold]{cfg['provider']}[/bold] | Modèle : [bold]{cfg['model']}[/bold]\n"
            f"Tapez votre question en français, [yellow]/reset[/yellow] pour réinitialiser l'historique, "
            f"ou [yellow]quit[/yellow] / [yellow]exit[/yellow] / [yellow]q[/yellow] pour quitter.",
            border_style="blue",
        )
    )

    async with AsyncExitStack() as stack:
        mcp_client = await stack.enter_async_context(
            MCPClient(command="uv", args=["run", "python", "-m", "src.mcp_server"])
        )
        chat = InvoiceAgentChat(llm_provider=llm, clients={"warehouse": mcp_client})

        while True:
            try:
                user_input = console.input("\n[bold green]Vous > [/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Arrêt propre de l'agent. Au revoir ![/dim]")
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                console.print("[dim]Arrêt propre de l'agent. Au revoir ![/dim]")
                break
            if user_input.lower() == "/reset":
                chat.messages.clear()
                console.print("[yellow]Historique de conversation réinitialisé.[/yellow]")
                continue

            step_counter = 0

            def _on_tool(entry: ToolTraceEntry) -> None:
                nonlocal step_counter
                step_counter += 1
                _render_tool_trace(step_counter, entry)

            try:
                trace = await chat.run(user_input, on_tool_executed=_on_tool)
                _render_final_answer(trace)
            except LLMError as exc:
                console.print(f"[bold red]Erreur LLM :[/bold red] {exc}")

    return 0


async def _run_compare_models(
    models: list[str],
    provider: str | None = None,
) -> int:
    """Exécute le jeu de validation sur plusieurs LLMs sans modifier le code et vérifie les chiffres par SQL."""
    eff_provider = (provider or "hyperfluid").strip().lower()
    comparison_rows: list[dict] = []

    console.print(
        Panel(
            f"[bold cyan]Validation Multi-LLM & Vérification SQL Directe[/bold cyan]\n"
            f"Modèles testés : [bold]{', '.join(models)}[/bold]\n"
            f"Q1 : [italic]{VALIDATION_QUESTIONS[0]}[/italic]\n"
            f"Q2 (suivi multi-tours) : [italic]{VALIDATION_QUESTIONS[1]}[/italic]",
            border_style="blue",
        )
    )

    for model_name in models:
        console.rule(f"[bold magenta]Exécution avec le modèle : {model_name}[/bold magenta]")
        if not has_valid_llm_credentials(provider=eff_provider, model=model_name):
            console.print(
                f"[bold red]Clé API manquante pour {eff_provider}/{model_name}.[/bold red]"
            )
            return 1

        try:
            traces = await _run_queries_session(
                queries=VALIDATION_QUESTIONS,
                provider=eff_provider,
                model=model_name,
            )
        except LLMError as exc:
            console.print(f"[bold red]Échec pour {model_name} :[/bold red] {exc}")
            comparison_rows.append(
                {
                    "model": model_name,
                    "total_s": 0.0,
                    "tools": "ERREUR",
                    "sql_count": 0,
                    "q1_ok": False,
                    "q2_ok": False,
                }
            )
            continue

        total_ms = sum(t.elapsed_ms for t in traces)
        tools_used = [e.tool_name for t in traces for e in t.tool_traces]
        sql_count = sum(len(e.executed_sql) for t in traces for e in t.tool_traces)

        q1_ans = traces[0].answer if len(traces) > 0 else ""
        q2_ans = traces[1].answer if len(traces) > 1 else ""
        q1_ok, q2_ok, q1_ref, q2_ref = _verify_ground_truth_sql(q1_ans, q2_ans)

        comparison_rows.append(
            {
                "model": model_name,
                "total_s": total_ms / 1000.0,
                "tools": " → ".join(tools_used) if tools_used else "Aucun",
                "sql_count": sql_count,
                "q1_ok": q1_ok,
                "q2_ok": q2_ok,
                "q1_ref": q1_ref,
                "q2_ref": q2_ref,
            }
        )

    table = Table(
        title="Comparatif Multi-LLM (Étape 8 — Exactitude vérifiée par SQL DuckDB)",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Modèle LLM", style="bold white")
    table.add_column("Temps", justify="right")
    table.add_column("Outils MCP appelés", min_width=24, overflow="fold")
    table.add_column("SQL", justify="right")
    table.add_column("Q1 (TTC retards)", justify="center")
    table.add_column("Q2 (V-2025-0001)", justify="center")

    for row in comparison_rows:
        b1 = "[bold green]EXACT[/bold green]" if row["q1_ok"] else "[bold red]ERREUR[/bold red]"
        b2 = "[bold green]EXACT[/bold green]" if row["q2_ok"] else "[bold red]ERREUR[/bold red]"
        table.add_row(
            row["model"],
            f"{row['total_s']:.2f} s",
            row["tools"],
            str(row["sql_count"]),
            b1,
            b2,
        )

    console.print(table)
    if comparison_rows and "q1_ref" in comparison_rows[0]:
        console.print(
            Panel(
                f"• [bold]Référence SQL Q1 :[/bold] {comparison_rows[0]['q1_ref']}\n"
                f"• [bold]Référence SQL Q2 :[/bold] {comparison_rows[0]['q2_ref']}",
                title="Vérité SQL de référence (DuckDB)",
                border_style="blue",
            )
        )

    return 0 if all(r["q1_ok"] and r["q2_ok"] for r in comparison_rows) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI de l'étape 8."""
    parser = argparse.ArgumentParser(
        description="Étape 8 : Agent IA multi-LLM connecté au serveur MCP (conversation et benchmark).",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="Pose une question (répétable pour tester le suivi multi-tours) puis quitte.",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare plusieurs LLMs sur les questions de validation et vérifie les chiffres par SQL.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gemma-4-12b,granite-4-2-8b",
        help="Liste des modèles séparés par virgule pour --compare-models (défaut : 'gemma-4-12b,granite-4-2-8b').",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Surcharge LLM_PROVIDER (hyperfluid, openai_compatible, claude, google).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Surcharge LLM_MODEL pour cette exécution.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Surcharge LLM_BASE_URL pour cette exécution.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    args = parse_args(argv)

    if args.compare_models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if not models:
            console.print("[bold red]Erreur : aucun modèle fourni dans --models.[/bold red]")
            return 1
        return asyncio.run(_run_compare_models(models=models, provider=args.provider))

    if not has_valid_llm_credentials(
        provider=args.provider, model=args.model, base_url=args.base_url
    ):
        cfg = resolve_llm_config(provider=args.provider, model=args.model, base_url=args.base_url)
        console.print(
            Panel(
                f"[bold red]Identifiants LLM manquants pour '{cfg['provider']}/{cfg['model']}'.[/bold red]\n\n"
                f"Veuillez configurer la clé API correspondante dans [cyan]{ENV_FILE}[/cyan], puis relancer :\n"
                "  [bold]uv run python -m src.step8_agent[/bold]",
                title="Configuration LLM requise",
                border_style="red",
            )
        )
        return 1

    if args.query:
        try:
            asyncio.run(
                _run_queries_session(
                    queries=args.query,
                    provider=args.provider,
                    model=args.model,
                    base_url=args.base_url,
                )
            )
            return 0
        except LLMError as exc:
            console.print(f"[bold red]Erreur LLM :[/bold red] {exc}")
            return 1

    return asyncio.run(
        _run_interactive_session(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
