from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from django.conf import settings


class DeepSeekError(Exception):
    """Raised when the local DeepSeek/Ollama backend fails."""


@dataclass
class ChatMessage:
    role: str
    content: str


class DeepSeekClient:
    """Client for local DeepSeek models served by Ollama."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.DEEPSEEK_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

    def is_available(self) -> bool:
        try:
            self._request("GET", "/api/tags")
            return True
        except DeepSeekError:
            return False

    def list_models(self) -> list[str]:
        payload = self._request("GET", "/api/tags")
        models = payload.get("models", [])
        return [model["name"] for model in models]

    def generate(self, prompt: str, system: str | None = None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            body["system"] = system

        payload = self._request("POST", "/api/generate", body)
        response = payload.get("response", "")
        if not response:
            raise DeepSeekError("Model returned an empty response.")
        return response

    def chat(self, messages: list[ChatMessage]) -> str:
        payload = self._request(
            "POST",
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                "stream": False,
            },
        )
        message = payload.get("message", {})
        content = message.get("content", "")
        if not content:
            raise DeepSeekError("Model returned an empty chat response.")
        return content

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DeepSeekError(
                f"Ollama request failed ({exc.code}): {detail or exc.reason}"
            ) from exc
        except URLError as exc:
            raise DeepSeekError(
                "Cannot reach Ollama. Start it with: brew services start ollama"
            ) from exc
        except json.JSONDecodeError as exc:
            raise DeepSeekError("Ollama returned invalid JSON.") from exc
