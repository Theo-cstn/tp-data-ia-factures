"""Boucle conversationnelle agentique multi-tours reliée au serveur MCP (adaptée de cli_project/core/chat.py)."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent.llm import BaseLLMProvider, ToolCall, ToolResult
from src.agent.tools import ToolManager
from src.mcp_client import MCPClient

SYSTEM_PROMPT = """Tu es un assistant expert en analyse financière et comptable spécialisé sur les factures fournisseurs de l'entreprise.

RÈGLES IMPÉRATIVES :
1. Tu DOIS toujours répondre en français, de manière claire, précise et structurée.
2. Pour obtenir des données, tu DOIS utiliser EXCLUSIVEMENT les outils MCP mis à ta disposition.
3. Il t'est FORMELLEMENT INTERDIT d'inventer des chiffres, des dates, des noms de fournisseurs/clients, des statuts ou des numéros de facture.
4. Ne fais JAMAIS de sommes complexes ou d'agrégations de tête : appuie-toi toujours sur les valeurs exactes calculées par les outils SQL/MCP et restitue les montants exacts au centime près.
5. Si un outil retourne une erreur (status="error"), lis le champ "error_message" et le champ "hint" pour corriger ton appel ou consulter le schéma.
6. Si une facture n'est pas dans les factures valides (ex: en quarantaine), signale-le explicitement en précisant le motif de rejet.

GUIDE DE CHOIX DES OUTILS MCP :
- `get_invoices_summary` : à privilégier pour toute question de comptage ou de totaux ("Combien de factures...", "Quel est le montant HT/TVA/TTC...", avec filtres optionnels `status`, `category`, `year`, `supplier_name`, `client_name`, `date_from`, `date_to`, `group_by`).
  Valeurs exactes connues : `status` ∈ {'Payée', 'En retard', 'En attente'}, `category` ∈ {'Conseil', 'Fournitures', 'Informatique', 'Maintenance'}.
- `list_invoices` : pour lister des factures individuelles selon des filtres.
- `get_invoice_details` : pour consulter l'en-tête complet, les lignes d'articles et l'éventuel statut de quarantaine d'une facture précise (ex: 'A-2025-0001' ou 'V-2025-0001').
- `get_gold_kpis` : pour obtenir directement les bilans pré-calculés ('fournisseurs', 'mensuel', 'retards', 'quarantaine').
- `get_schema` : pour inspecter les vues et colonnes disponibles avant d'écrire du SQL sur-mesure.
- `execute_readonly_sql` : uniquement pour des requêtes analytiques complexes (jointures avec `silver.lignes_facture`, classements, calculs spécifiques) non couvertes par les autres outils.
"""


@dataclass
class ToolTraceEntry:
    """Trace structurée d'un appel d'outil MCP et des requêtes SQL exécutées."""

    tool_name: str
    arguments: dict[str, Any]
    raw_result: str
    parsed_result: dict[str, Any] | None
    executed_sql: list[str]
    sql_params: list[list[Any]]
    is_error: bool


@dataclass
class TurnTrace:
    """Trace complète d'un tour de conversation utilisateur -> agent."""

    query: str
    answer: str
    model: str
    provider: str
    elapsed_ms: float
    tool_traces: list[ToolTraceEntry] = field(default_factory=list)
    iterations: int = 0


def _parse_tool_json(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class InvoiceAgentChat:
    """Gère l'historique multi-tours et la boucle d'appels d'outils MCP pour un LLM donné."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        clients: dict[str, MCPClient],
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = 10,
    ) -> None:
        self.llm: BaseLLMProvider = llm_provider
        self.clients: dict[str, MCPClient] = clients
        self.system_prompt: str = system_prompt
        self.max_iterations: int = max_iterations
        self.messages: list[dict[str, Any]] = []

    async def run(
        self,
        query: str,
        on_tool_executed: Callable[[ToolTraceEntry], None] | None = None,
    ) -> TurnTrace:
        """Exécute un tour de conversation complet jusqu'à la réponse finale."""
        start = time.perf_counter()
        self.llm.add_user_message(self.messages, query)

        tools = await ToolManager.get_all_tools(self.clients)
        tool_traces: list[ToolTraceEntry] = []
        final_text = ""
        iterations = 0

        for _ in range(self.max_iterations):
            iterations += 1
            response = await asyncio.to_thread(
                self.llm.chat,
                messages=self.messages,
                system=self.system_prompt,
                temperature=0.0,
                tools=tools,
            )
            self.llm.add_assistant_message(self.messages, response)

            if not response.tool_calls:
                final_text = self.llm.text_from_message(response)
                break

            tool_results: list[ToolResult] = await ToolManager.execute_tool_requests(
                self.clients, response
            )

            for tc, res in zip(response.tool_calls, tool_results):
                parsed = _parse_tool_json(res.content)
                exec_sql = list(parsed.get("executed_sql", [])) if parsed else []
                sql_params = list(parsed.get("sql_params", [])) if parsed else []
                entry = ToolTraceEntry(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    raw_result=res.content,
                    parsed_result=parsed,
                    executed_sql=exec_sql,
                    sql_params=sql_params,
                    is_error=res.is_error or (parsed is not None and parsed.get("status") == "error"),
                )
                tool_traces.append(entry)
                if on_tool_executed is not None:
                    on_tool_executed(entry)

            self.llm.add_tool_results(self.messages, tool_results)
        else:
            final_text = (
                "Limite d'itérations atteinte avant qu'une réponse finale ne soit produite."
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return TurnTrace(
            query=query,
            answer=final_text,
            model=self.llm.model,
            provider=self.llm.provider_name,
            elapsed_ms=elapsed_ms,
            tool_traces=tool_traces,
            iterations=iterations,
        )
