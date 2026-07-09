from __future__ import annotations

import json
import re
from json import JSONDecodeError


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
MARKDOWN_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff]")
REASONING_MARKERS = (
    "okay",
    "i'm",
    "i am",
    "i need",
    "i should",
    "let me",
    "step by step",
    "reasoning",
    "thinking process",
    "analyze the request",
    "role:",
    "constraints:",
    "return only json",
    "schema",
    "private context",
    "project:",
    "phase:",
    "task:",
)


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


def clean_llm_text(text: str, max_chars: int = 900) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = THINK_BLOCK_RE.sub(" ", cleaned)
    cleaned = MARKDOWN_FENCE_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("```json", " ").replace("```", " ")

    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        line_lower = line.lower().lstrip("*-0123456789. ")
        if any(marker in line_lower for marker in REASONING_MARKERS):
            continue
        if line_lower in {"json", "answer", "response"}:
            continue
        lines.append(line)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("` ")
    if cleaned in {"...", "…", "-", "—", "not json", "still not json"}:
        return ""
    if cleaned.count("{") > 1 and cleaned.count("}") > 1:
        return ""
    if not _looks_user_facing(cleaned):
        return ""
    return cleaned[:max_chars].rstrip(" ,;:")


def extract_question_from_text(text: str) -> str:
    cleaned = clean_llm_text(text, max_chars=320)
    if not cleaned:
        return ""
    if "{" in cleaned[:20] or "}" in cleaned:
        return ""
    question_end = cleaned.find("?")
    if question_end != -1:
        candidate = cleaned[: question_end + 1].strip()
    else:
        sentences = re.split(r"(?<=[.!])\s+", cleaned)
        candidate = sentences[0].strip()
        if candidate and not candidate.endswith("?"):
            candidate = candidate.rstrip(".!:;") + "?"
    if len(candidate) < 12 or _has_reasoning_marker(candidate):
        return ""
    return candidate[:260]


def is_russian_user_facing_text(text: str) -> bool:
    return _looks_user_facing(text)


def _looks_user_facing(text: str) -> bool:
    if not text or _has_reasoning_marker(text):
        return False
    if "{" in text[:20] or "}" in text:
        return False
    if CJK_RE.search(text):
        return False
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 12:
        return False
    cyrillic_letters = [char for char in letters if "а" <= char.lower() <= "я" or char.lower() == "ё"]
    return len(cyrillic_letters) / len(letters) >= 0.55


def _has_reasoning_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REASONING_MARKERS)
