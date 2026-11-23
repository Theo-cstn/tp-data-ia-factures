"""Étape 7 : Suite de tests automatisés du serveur MCP (outils, transparence SQL et sécurité read-only)."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from decimal import Decimal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import BASE_DIR
from src.mcp_client import MCPClient, extract_tool_json

console = Console()


@dataclass
class MCPTestCaseResult:
    """Résultat d'un scénario de test automatisé sur le serveur MCP."""

    name: str
    tool: str
    args: dict
    executed_sql: str
    excerpt: str
    passed: bool


async def run_mcp_test_suite() -> list[MCPTestCaseResult]:
    """Lance le serveur MCP en sous-processus stdio et valide tous les outils et garde-fous."""
    results: list[MCPTestCaseResult] = []

    async with MCPClient(
        command="uv",
        args=["run", "python", "-m", "src.mcp_server"],
    ) as client:
        # 1. list_tools()
        tools = await client.list_tools()
        tool_names = sorted(t.name for t in tools)
        expected_tools = sorted(
            [
                "list_invoices",
                "get_invoices_summary",
                "get_invoice_details",
                "get_schema",
                "execute_readonly_sql",
                "get_gold_kpis",
            ]
        )
        results.append(
            MCPTestCaseResult(
                name="1. Exposition des 6 outils MCP",
                tool="list_tools",
                args={},
                executed_sql="N/A (découverte MCP)",
                excerpt=", ".join(tool_names),
                passed=(tool_names == expected_tools),
            )
        )

        # 2. get_schema()
        schema_res = extract_tool_json(await client.call_tool("get_schema", {}))
        view_names = {v["full_name"] for v in schema_res.get("data", {}).get("views", [])}
        required_views = {
            "bronze.documents",
            "silver.factures",
            "silver.lignes_facture",
            "silver.quarantaine",
            "gold.kpi_mensuels",
            "gold.kpi_fournisseurs",
            "gold.factures_en_retard",
        }
        schema_ok = (
            schema_res.get("status") == "success"
            and bool(schema_res.get("executed_sql"))
            and required_views.issubset(view_names)
        )
        results.append(
            MCPTestCaseResult(
                name="2. Découverte du schéma (bronze/silver/gold)",
                tool="get_schema",
                args={},
                executed_sql="; ".join(schema_res.get("executed_sql", [])[:2]) + " ...",
                excerpt=f"{len(view_names)} vues découvertes",
                passed=schema_ok,
            )
        )

        # 3. get_invoices_summary(status="En retard", year=2025)
        sum_args = {"status": "En retard", "year": 2025}
        sum_res = extract_tool_json(await client.call_tool("get_invoices_summary", sum_args))
        sum_data = sum_res.get("data", {})
        sum_ok = (
            sum_res.get("status") == "success"
            and bool(sum_res.get("executed_sql"))
            and sum_data.get("invoice_count") == 12
            and Decimal(str(sum_data.get("total_ttc", "0"))) == Decimal("106886.43")
        )
        results.append(
            MCPTestCaseResult(
                name="3. Résumé factures en retard 2025 (12 / 106 886,43 €)",
                tool="get_invoices_summary",
                args=sum_args,
                executed_sql=" | ".join(sum_res.get("executed_sql", [])),
                excerpt=f"count={sum_data.get('invoice_count')}, total_ttc={sum_data.get('total_ttc')} €",
                passed=sum_ok,
            )
        )

        # 4a. list_invoices(supplier_name="Aubry", status="En retard")
        list_args = {"supplier_name": "Aubry", "status": "En retard"}
        list_res = extract_tool_json(await client.call_tool("list_invoices", list_args))
        list_data = list_res.get("data", {})
        list_ok = (
            list_res.get("status") == "success"
            and bool(list_res.get("executed_sql"))
            and list_data.get("count") == 4
            and all(
                inv.get("supplier_name") == "Aubry" and inv.get("status") == "En retard"
                for inv in list_data.get("invoices", [])
            )
        )
        results.append(
            MCPTestCaseResult(
                name="4a. Liste filtrée (Aubry + En retard)",
                tool="list_invoices",
                args=list_args,
                executed_sql=" | ".join(list_res.get("executed_sql", [])),
                excerpt=f"count={list_data.get('count')} factures",
                passed=list_ok,
            )
        )

        # 4b. Test d'injection SQL dans un filtre
        sqli_args = {"supplier_name": "' OR 1=1 --"}
        sqli_res = extract_tool_json(await client.call_tool("list_invoices", sqli_args))
        sqli_data = sqli_res.get("data", {})
        sqli_ok = (
            sqli_res.get("status") == "success"
            and sqli_data.get("count") == 0
            and "?" in " ".join(sqli_res.get("executed_sql", []))
        )
        results.append(
            MCPTestCaseResult(
                name="4b. Neutralisation injection SQL (requête paramétrée)",
                tool="list_invoices",
                args=sqli_args,
                executed_sql=" | ".join(sqli_res.get("executed_sql", [])),
                excerpt=f"count={sqli_data.get('count')} (injection bloquée par ?)",
                passed=sqli_ok,
            )
        )

        # 5a. get_invoice_details("A-2025-0001") -> valide avec 3 lignes
        det1_args = {"invoice_number": "A-2025-0001"}
        det1_res = extract_tool_json(await client.call_tool("get_invoice_details", det1_args))
        det1_data = det1_res.get("data", {})
        det1_ok = (
            det1_res.get("status") == "success"
            and det1_data.get("found_in") == "silver.factures"
            and len(det1_data.get("lines", [])) == 3
            and len(det1_res.get("executed_sql", [])) == 3
        )
        results.append(
            MCPTestCaseResult(
                name="5a. Détail facture valide (A-2025-0001, 3 lignes)",
                tool="get_invoice_details",
                args=det1_args,
                executed_sql=f"{len(det1_res.get('executed_sql', []))} requêtes SQL",
                excerpt=f"found_in={det1_data.get('found_in')}, lines={len(det1_data.get('lines', []))}, ttc={det1_data.get('invoice', {}).get('total_ttc')}",
                passed=det1_ok,
            )
        )

        # 5b. get_invoice_details("V-2025-0001") -> rejetée en quarantaine
        det2_args = {"invoice_number": "V-2025-0001"}
        det2_res = extract_tool_json(await client.call_tool("get_invoice_details", det2_args))
        det2_data = det2_res.get("data", {})
        q_info = det2_data.get("quarantine") or {}
        det2_ok = (
            det2_res.get("status") == "success"
            and det2_data.get("found_in") == "silver.quarantaine"
            and "INCOHERENCE_TTC" in str(q_info.get("rejection_codes", ""))
        )
        results.append(
            MCPTestCaseResult(
                name="5b. Détail facture quarantaine (V-2025-0001)",
                tool="get_invoice_details",
                args=det2_args,
                executed_sql=f"{len(det2_res.get('executed_sql', []))} requêtes SQL",
                excerpt=f"quarantaine ({q_info.get('rejection_codes')})",
                passed=det2_ok,
            )
        )

        # 6. get_gold_kpis(report_type="fournisseurs")
        gold_args = {"report_type": "fournisseurs"}
        gold_res = extract_tool_json(await client.call_tool("get_gold_kpis", gold_args))
        gold_data = gold_res.get("data", {})
        gold_ok = (
            gold_res.get("status") == "success"
            and gold_data.get("count") == 8
            and bool(gold_res.get("executed_sql"))
        )
        top_sup = (gold_data.get("rows") or [{}])[0]
        results.append(
            MCPTestCaseResult(
                name="6. KPI Couche Gold (fournisseurs)",
                tool="get_gold_kpis",
                args=gold_args,
                executed_sql=" | ".join(gold_res.get("executed_sql", [])),
                excerpt=f"{gold_data.get('count')} fournisseurs (top: {top_sup.get('supplier_name')})",
                passed=gold_ok,
            )
        )

        # 7a. execute_readonly_sql (SELECT valide)
        sel_sql = (
            "SELECT category, COUNT(*) AS nb, SUM(total_ttc) AS total_ttc "
            "FROM silver.factures GROUP BY category ORDER BY total_ttc DESC"
        )
        sel_res = extract_tool_json(
            await client.call_tool("execute_readonly_sql", {"sql": sel_sql})
        )
        sel_data = sel_res.get("data", {})
        sel_ok = (
            sel_res.get("status") == "success"
            and sel_data.get("row_count") == 4
            and bool(sel_res.get("executed_sql"))
        )
        results.append(
            MCPTestCaseResult(
                name="7a. SQL libre SELECT analytique",
                tool="execute_readonly_sql",
                args={"sql": sel_sql},
                executed_sql=" | ".join(sel_res.get("executed_sql", [])),
                excerpt=f"{sel_data.get('row_count')} catégories retournées",
                passed=sel_ok,
            )
        )

        # 7b. execute_readonly_sql (colonne inexistante -> auto-correction avec hint)
        bad_sql = "SELECT montant_ttc FROM silver.factures"
        bad_res = extract_tool_json(
            await client.call_tool("execute_readonly_sql", {"sql": bad_sql})
        )
        bad_ok = (
            bad_res.get("status") == "error"
            and bool(bad_res.get("error_message"))
            and "get_schema" in str(bad_res.get("hint", ""))
        )
        results.append(
            MCPTestCaseResult(
                name="7b. SQL libre colonne invalide -> hint d'auto-correction",
                tool="execute_readonly_sql",
                args={"sql": bad_sql},
                executed_sql=" | ".join(bad_res.get("executed_sql", [])),
                excerpt=str(bad_res.get("error_message", ""))[:65],
                passed=bad_ok,
            )
        )

        # 7c. 3 tentatives d'écriture/destruction strictement bloquées
        destructive_queries = [
            ("7c-1. Sécurité : blocage DROP VIEW", "DROP VIEW factures"),
            ("7c-2. Sécurité : blocage DELETE", "DELETE FROM silver.factures"),
            ("7c-3. Sécurité : blocage CREATE TABLE", "CREATE TABLE hack AS SELECT 1"),
        ]
        for label, d_sql in destructive_queries:
            d_res = extract_tool_json(
                await client.call_tool("execute_readonly_sql", {"sql": d_sql})
            )
            d_ok = d_res.get("status") == "error"
            results.append(
                MCPTestCaseResult(
                    name=label,
                    tool="execute_readonly_sql",
                    args={"sql": d_sql},
                    executed_sql=" | ".join(d_res.get("executed_sql", [])),
                    excerpt=str(d_res.get("error_message", ""))[:65],
                    passed=d_ok,
                )
            )

    return results


def display_test_report(results: list[MCPTestCaseResult]) -> None:
    """Affiche le rapport Rich des tests MCP et la configuration Claude Desktop / Claude Code."""
    table = Table(
        title="Validation Automatisée du Serveur MCP (Étape 7)",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Test", style="white")
    table.add_column("Outil", style="magenta", no_wrap=True)
    table.add_column("executed_sql", style="dim", overflow="fold")
    table.add_column("Extrait résultat", overflow="fold")
    table.add_column("Statut", justify="center", no_wrap=True)

    for r in results:
        badge = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
        table.add_row(r.name, r.tool, r.executed_sql, r.excerpt, badge)

    console.print(table)

    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    summary_style = "green" if passed_count == total_count else "red"

    claude_config = {
        "mcpServers": {
            "factures-warehouse": {
                "command": "uv",
                "args": [
                    "--directory",
                    str(BASE_DIR),
                    "run",
                    "python",
                    "-m",
                    "src.mcp_server",
                ],
            }
        }
    }

    console.print(
        Panel(
            f"[bold {summary_style}]Bilan : {passed_count}/{total_count} tests PASS[/bold {summary_style}]\n\n"
            "[bold]Configuration prête à copier pour Claude Desktop / Claude Code :[/bold]\n"
            f"[cyan]{json.dumps(claude_config, indent=2, ensure_ascii=False)}[/cyan]",
            title="Rapport Étape 7 & Intégration Client MCP",
            border_style="blue",
        )
    )


def main() -> int:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    results = asyncio.run(run_mcp_test_suite())
    display_test_report(results)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
