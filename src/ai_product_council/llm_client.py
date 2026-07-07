from __future__ import annotations

import httpx

from ai_product_council.config import Settings


class LLMClientError(RuntimeError):
    pass


class LMStudioClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def chat(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.settings.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": False,
        }
        try:
            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.settings.timeout_seconds,
                trust_env=False,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            if len(detail) > 1200:
                detail = detail[:1200] + "...[truncated]"
            raise LLMClientError(f"LM Studio API error: {exc}. Response body: {detail}") from exc
        except httpx.HTTPError as exc:
            raise LLMClientError(f"LM Studio API error: {exc}") from exc

        data = response.json()
        try:
            message = data["choices"][0]["message"]
            return message.get("content") or message.get("reasoning_content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Unexpected LM Studio response: {data}") from exc
