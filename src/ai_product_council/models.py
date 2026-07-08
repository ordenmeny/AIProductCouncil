from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ai_product_council.json_utils import clean_llm_text


ProjectMode = Literal["new_service", "feature_in_existing_product"]

PhaseName = Literal[
    "clarifying_questions",
    "analysis",
    "debate",
    "mvp_proposal",
    "mvp_vote",
]

Decision = Literal[
    "go",
    "go_after_clarification",
    "no_go",
    "pivot_or_narrow_mvp",
    "unknown",
]

ResponseStatus = Literal["llm", "repaired", "text", "failed", "fallback"]
FallbackReason = Literal["", "json", "text", "deterministic", "model_unavailable"]


class AgentRole(BaseModel):
    name: str
    slug: str
    description: str
    system_prompt: str
    private_context_path: Path


class AgentPayload(BaseModel):
    summary: str = ""
    question: str = ""
    arguments: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    mvp_features: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    roadmap_items: list[str] = Field(default_factory=list)
    decision: Decision = "unknown"
    next_step: str = ""
    confidence: int = Field(default=3, ge=1, le=5)

    @field_validator("summary", "question", "next_step", mode="before")
    @classmethod
    def clean_placeholder_string(cls, value):
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if normalized in {"...", "…", "-", "—"}:
            return ""
        if _contains_reasoning_marker(normalized):
            return clean_llm_text(normalized)
        return normalized

    @field_validator(
        "arguments",
        "risks",
        "mvp_features",
        "out_of_scope",
        "open_questions",
        "insights",
        "roadmap_items",
        mode="before",
    )
    @classmethod
    def clean_placeholder_list(cls, value):
        if not isinstance(value, list):
            return value
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if _contains_reasoning_marker(normalized):
                normalized = clean_llm_text(normalized)
            if normalized and normalized not in {"...", "…", "-", "—"}:
                result.append(normalized)
        return result


def _contains_reasoning_marker(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in ("thinking process", "analyze the request", "return only json", "schema keys", "<think>")
    )


class ClarifyingQuestion(BaseModel):
    agent: str
    role: str
    question: str
    status: ResponseStatus = "llm"
    fallback_reason: FallbackReason = ""
    error: str | None = None
    raw_text: str = ""


class UserAnswer(BaseModel):
    text: str = ""
    answered_at: datetime = Field(default_factory=datetime.now)


class MeetingTurn(BaseModel):
    agent: str
    role: str
    phase: PhaseName
    status: ResponseStatus
    fallback_reason: FallbackReason = ""
    payload: AgentPayload = Field(default_factory=AgentPayload)
    raw_text: str = ""
    error: str | None = None


class MeetingTranscript(BaseModel):
    turns: list[MeetingTurn] = Field(default_factory=list)

    def add(self, turn: MeetingTurn) -> None:
        self.turns.append(turn)


class MeetingState(BaseModel):
    idea: str
    project_mode: ProjectMode = "new_service"
    constraints: str = ""
    desired_result: str = ""
    started_at: datetime = Field(default_factory=datetime.now)
    questions: list[ClarifyingQuestion] = Field(default_factory=list)
    user_answer: UserAnswer = Field(default_factory=UserAnswer)
    transcript: MeetingTranscript = Field(default_factory=MeetingTranscript)
    transcript_markdown: str = ""
    final_plan_markdown: str = ""

    @property
    def votes(self) -> list[MeetingTurn]:
        return [turn for turn in self.transcript.turns if turn.phase == "mvp_vote"]

    def to_export_dict(self) -> dict:
        return self.model_dump(mode="json")
