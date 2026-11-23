"""Gestionnaire d'outils MCP pour la boucle agentique (adapté de cli_project/core/tools.py)."""

from __future__ import annotations

import json
from typing import Any, Optional

from mcp.types import CallToolResult, TextContent

from src.agent.llm import LLMResponse, ToolResult
from src.mcp_client import MCPClient


class ToolManager:
    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[dict[str, Any]]:
        """Récupère tous les outils exposés par les clients MCP connectés."""
        tools: list[dict[str, Any]] = []
        for client in clients.values():
            tool_models = await client.list_tools()
            tools.extend(
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tool_models
            )
        return tools

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        for client in clients:
            tools = await client.list_tools()
            if any(t.name == tool_name for t in tools):
                return client
        return None

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], message: LLMResponse
    ) -> list[ToolResult]:
        """Exécute les appels d'outils demandés par le LLM via MCP."""
        tool_results: list[ToolResult] = []
        for tool_request in message.tool_calls:
            tool_use_id = tool_request.id
            tool_name = tool_request.name
            tool_input = tool_request.arguments

            client = await cls._find_client_with_tool(
                list(clients.values()), tool_name
            )
            if not client:
                tool_results.append(
                    ToolResult(
                        id=tool_use_id,
                        name=tool_name,
                        content=json.dumps(
                            {
                                "status": "error",
                                "error_message": f"Outil MCP introuvable : '{tool_name}'.",
                            },
                            ensure_ascii=False,
                        ),
                        is_error=True,
                    )
                )
                continue

            try:
                tool_output: CallToolResult = await client.call_tool(tool_name, tool_input)
                items = tool_output.content if tool_output else []
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                raw_content = "\n".join(content_list).strip()
                tool_results.append(
                    ToolResult(
                        id=tool_use_id,
                        name=tool_name,
                        content=raw_content,
                        is_error=bool(tool_output and tool_output.isError),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                tool_results.append(
                    ToolResult(
                        id=tool_use_id,
                        name=tool_name,
                        content=json.dumps(
                            {
                                "status": "error",
                                "error_message": f"Erreur d'exécution de '{tool_name}' : {exc}",
                            },
                            ensure_ascii=False,
                        ),
                        is_error=True,
                    )
                )

        return tool_results
