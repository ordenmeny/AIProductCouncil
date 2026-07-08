from __future__ import annotations

import json
import re
from json import JSONDecodeError


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict:
    match = JSON_BLOCK_RE.search(text)
    candidate = match.group(1) if match else text.strip()

    decoder = json.JSONDecoder()

    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    raise ValueError("JSON object not found")
