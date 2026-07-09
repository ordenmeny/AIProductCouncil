from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI
from openai import APIConnectionError, APIStatusError

from backend.app.core.config import Settings


class LLMClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
        )

    async def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self._settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_tokens,
        }
        if self._settings.llm_use_response_format:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = await self._client.chat.completions.create(**payload)
        except APIStatusError as exc:
            raise RuntimeError(_format_openai_status_error(exc)) from exc
        except APIConnectionError as exc:
            raise RuntimeError(f"Cannot connect to LM Studio at {self._settings.openai_base_url}: {exc}") from exc
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned an empty response")
        return content

    async def healthcheck(self) -> dict[str, Any]:
        models = await self.list_models()
        response = await self._client.chat.completions.create(
            model=self._settings.openai_model,
            messages=[
                {"role": "user", "content": "Return only this JSON: {\"status\":\"ok\"}"},
            ],
            temperature=0,
            max_tokens=20,
        )
        return {
            "status": "ok",
            "configured_model": self._settings.openai_model,
            "available_models": models,
            "base_url": self._settings.openai_base_url,
            "content": response.choices[0].message.content,
        }

    async def list_models(self) -> list[str]:
        try:
            response = await self._client.models.list()
        except APIStatusError as exc:
            raise RuntimeError(_format_openai_status_error(exc)) from exc
        except APIConnectionError as exc:
            raise RuntimeError(f"Cannot connect to LM Studio at {self._settings.openai_base_url}: {exc}") from exc
        return [model.id for model in response.data]

    async def repair_json(self, invalid_payload: str, validation_error: str) -> str:
        system_prompt = (
            "You repair invalid JSON for a backend parser. Return only a valid JSON object. "
            "Do not add Markdown, comments, or explanatory text."
        )
        user_prompt = json.dumps(
            {
                "task": "Repair this payload so it is a single valid JSON object matching the requested schema.",
                "invalid_payload": invalid_payload,
                "validation_error": validation_error,
            },
            ensure_ascii=False,
        )
        return await self.complete_json(system_prompt, user_prompt)

def _format_openai_status_error(exc: APIStatusError) -> str:
    body = exc.response.text if exc.response is not None else ""
    body = body[:1000] if body else ""
    return f"LM Studio returned HTTP {exc.status_code}. Body: {body or exc.message}"
