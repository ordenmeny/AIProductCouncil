from __future__ import annotations

import json
import re


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict:
    match = JSON_BLOCK_RE.search(text)
    candidate = match.group(1) if match else text.strip()

    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("JSON object not found")
        candidate = candidate[start : end + 1]

    return json.loads(candidate)
