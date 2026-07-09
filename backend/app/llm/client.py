from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

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
        response = await self._client.chat.completions.create(
            model=self._settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned an empty response")
        return content

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


def json_schema_hint(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, indent=2)
