"""Structures génériques et connecteurs LLM interchangeables (adapté de cli_project/core/llm.py)."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from src.config import resolve_llm_config


class LLMError(RuntimeError):
    """Erreur levée lors de l'appel à un fournisseur de LLM."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    id: str
    name: str
    content: str
    is_error: bool


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    raw: Any = None

    @property
    def stop_reason(self) -> str:
        return "tool_use" if self.tool_calls else "end_turn"


class BaseLLMProvider(ABC):
    """Interface abstraite commune à tous les fournisseurs de LLM."""

    provider_name: str
    model: str

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Envoie la conversation au LLM et retourne un LLMResponse normalisé."""

    @abstractmethod
    def add_user_message(self, messages: list[dict[str, Any]], content: str) -> None:
        """Ajoute un message utilisateur à l'historique."""

    @abstractmethod
    def add_assistant_message(
        self, messages: list[dict[str, Any]], response: LLMResponse
    ) -> None:
        """Ajoute la réponse de l'assistant (et ses tool_calls) à l'historique."""

    @abstractmethod
    def add_tool_results(
        self, messages: list[dict[str, Any]], results: list[ToolResult]
    ) -> None:
        """Ajoute les résultats d'exécution des outils MCP à l'historique."""

    def text_from_message(self, response: LLMResponse) -> str:
        return response.text


def _clean_model_text(text: str) -> str:
    """Nettoie les balises internes de raisonnement éventuellement émises par certains modèles vLLM."""
    cleaned = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _parse_json_arguments(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        raw_args = raw_args.strip()
        if not raw_args:
            return {}
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_fallback_tool_calls(
    content: str, known_tool_names: set[str]
) -> tuple[str, list[ToolCall]]:
    """Extrait d'éventuels appels d'outils encodés en JSON dans le texte libre d'un modèle open-weights."""
    if not content or not known_tool_names:
        return content, []

    found_calls: list[ToolCall] = []
    cleaned_content = content

    # 1. Blocs <tool_call>...</tool_call> ou ```json ... ```
    block_patterns = [
        re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE),
        re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL | re.IGNORECASE),
    ]

    for pattern in block_patterns:
        for match in list(pattern.finditer(cleaned_content)):
            candidate = match.group(1).strip()
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            items = parsed if isinstance(parsed, list) else [parsed]
            matched_any = False
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("tool") or item.get("function")
                if isinstance(name, dict):
                    name = name.get("name")
                if isinstance(name, str) and name in known_tool_names:
                    args = (
                        item.get("arguments")
                        or item.get("parameters")
                        or item.get("args")
                        or item.get("input")
                        or {}
                    )
                    found_calls.append(
                        ToolCall(
                            id=f"fallback_tc_{len(found_calls) + 1}",
                            name=name,
                            arguments=_parse_json_arguments(args),
                        )
                    )
                    matched_any = True

            if matched_any:
                cleaned_content = cleaned_content.replace(match.group(0), "").strip()

    if found_calls:
        return cleaned_content, found_calls

    # 2. Objet JSON nu contenant {"name": "<outil>", ...}
    stripped = cleaned_content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                name = parsed.get("name") or parsed.get("tool")
                if isinstance(name, str) and name in known_tool_names:
                    args = (
                        parsed.get("arguments")
                        or parsed.get("parameters")
                        or parsed.get("args")
                        or {}
                    )
                    found_calls.append(
                        ToolCall(
                            id="fallback_tc_1",
                            name=name,
                            arguments=_parse_json_arguments(args),
                        )
                    )
                    return "", found_calls
        except json.JSONDecodeError:
            pass

    return content, []


class OpenAICompatibleProvider(BaseLLMProvider):
    """Connecteur OpenAI-compatible (Hyperfluid vLLM, Ollama, vLLM local, OpenAI, Mistral...)."""

    def __init__(
        self,
        provider_name: str,
        model: str,
        endpoint_url: str,
        api_key: str,
        timeout: float = 120.0,
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout

    @staticmethod
    def _to_openai_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description") or "",
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        api_messages: list[dict[str, Any]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }
        openai_tools = self._to_openai_tools(tools)
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.endpoint_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"Erreur HTTP {exc.response.status_code} ({self.model}) : {exc.response.text[:500]}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Erreur réseau ({self.model}) : {exc}") from exc
        except ValueError as exc:
            raise LLMError(f"Réponse JSON invalide ({self.model}) : {exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"Réponse vide du modèle {self.model} : {data}")

        msg = choices[0].get("message") or {}
        text = _clean_model_text(msg.get("content") or "")
        raw_tool_calls = msg.get("tool_calls") or []

        tool_calls: list[ToolCall] = []
        for idx, tc in enumerate(raw_tool_calls, start=1):
            fn = tc.get("function") or {}
            tc_name = fn.get("name", "")
            if not tc_name:
                continue
            tool_calls.append(
                ToolCall(
                    id=str(tc.get("id") or f"tc_{idx}"),
                    name=tc_name,
                    arguments=_parse_json_arguments(fn.get("arguments")),
                )
            )

        if not tool_calls and text and tools:
            known_names = {t["name"] for t in tools}
            text, tool_calls = _extract_fallback_tool_calls(text, known_names)

        return LLMResponse(text=text.strip(), tool_calls=tool_calls, raw=data)

    def add_user_message(self, messages: list[dict[str, Any]], content: str) -> None:
        messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self, messages: list[dict[str, Any]], response: LLMResponse
    ) -> None:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": response.text or None,
        }
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response.tool_calls
            ]
        elif msg["content"] is None:
            msg["content"] = ""
        messages.append(msg)

    def add_tool_results(
        self, messages: list[dict[str, Any]], results: list[ToolResult]
    ) -> None:
        for res in results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": res.id,
                    "name": res.name,
                    "content": res.content,
                }
            )


class AnthropicClaudeProvider(BaseLLMProvider):
    """Connecteur Anthropic Messages API via httpx (si LLM_PROVIDER=claude)."""

    def __init__(self, model: str, endpoint_url: str, api_key: str, timeout: float = 120.0) -> None:
        self.provider_name = "claude"
        self.model = model
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description") or "",
                    "input_schema": t.get("input_schema") or {"type": "object", "properties": {}},
                }
                for t in tools
            ]

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.endpoint_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"Erreur Anthropic ({self.model}) : {exc}") from exc

        blocks = data.get("content") or []
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        calls = [
            ToolCall(
                id=b["id"],
                name=b["name"],
                arguments=b.get("input") or {},
            )
            for b in blocks
            if b.get("type") == "tool_use"
        ]
        return LLMResponse(text="\n".join(texts).strip(), tool_calls=calls, raw=blocks)

    def add_user_message(self, messages: list[dict[str, Any]], content: str) -> None:
        messages.append({"role": "user", "content": content})

    def add_assistant_message(self, messages: list[dict[str, Any]], response: LLMResponse) -> None:
        content = response.raw if isinstance(response.raw, list) else response.text
        messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, messages: list[dict[str, Any]], results: list[ToolResult]) -> None:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.id,
                        "content": r.content,
                        "is_error": r.is_error,
                    }
                    for r in results
                ],
            }
        )


class GoogleGeminiProvider(BaseLLMProvider):
    """Connecteur Google Gemini REST API via httpx (si LLM_PROVIDER=google)."""

    def __init__(self, model: str, endpoint_url: str, api_key: str, timeout: float = 120.0) -> None:
        self.provider_name = "google"
        self.model = model
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "contents": messages,
            "generationConfig": {"temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t["name"],
                            "description": t.get("description") or "",
                            "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                        }
                        for t in tools
                    ]
                }
            ]

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    self.endpoint_url,
                    params={"key": self.api_key},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"Erreur Google Gemini ({self.model}) : {exc}") from exc

        candidates = data.get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts", []) if candidates else []
        texts = [p["text"] for p in parts if "text" in p]
        calls = [
            ToolCall(
                id=f"gemini_tc_{idx}",
                name=p["functionCall"]["name"],
                arguments=p["functionCall"].get("args") or {},
            )
            for idx, p in enumerate(parts, start=1)
            if "functionCall" in p
        ]
        return LLMResponse(text="\n".join(texts).strip(), tool_calls=calls, raw=parts)

    def add_user_message(self, messages: list[dict[str, Any]], content: str) -> None:
        messages.append({"role": "user", "parts": [{"text": content}]})

    def add_assistant_message(self, messages: list[dict[str, Any]], response: LLMResponse) -> None:
        parts = response.raw if isinstance(response.raw, list) else [{"text": response.text}]
        messages.append({"role": "model", "parts": parts})

    def add_tool_results(self, messages: list[dict[str, Any]], results: list[ToolResult]) -> None:
        messages.append(
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": r.name,
                            "response": {"content": r.content},
                        }
                    }
                    for r in results
                ],
            }
        )


def create_llm_provider(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> BaseLLMProvider:
    """Instancie le connecteur LLM configuré dans .env (ou surchargé par argument)."""
    cfg = resolve_llm_config(provider=provider, model=model, base_url=base_url, api_key=api_key)
    eff_provider = cfg["provider"]

    if eff_provider in {"hyperfluid", "openai_compatible"}:
        return OpenAICompatibleProvider(
            provider_name=eff_provider,
            model=cfg["model"],
            endpoint_url=cfg["base_url"],
            api_key=cfg["api_key"],
        )
    if eff_provider == "claude":
        return AnthropicClaudeProvider(
            model=cfg["model"],
            endpoint_url=cfg["base_url"],
            api_key=cfg["api_key"],
        )
    if eff_provider == "google":
        return GoogleGeminiProvider(
            model=cfg["model"],
            endpoint_url=cfg["base_url"],
            api_key=cfg["api_key"],
        )
    raise ValueError(f"Fournisseur LLM non supporté : {eff_provider}")
