from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


ProjectMode = Literal["new_saas", "feature_in_existing_product"]

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

ResponseStatus = Literal["llm", "repaired", "failed"]


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
    decision: Decision = "unknown"
    next_step: str = ""
    confidence: int = Field(default=3, ge=1, le=5)


class ClarifyingQuestion(BaseModel):
    agent: str
    role: str
    question: str
    status: ResponseStatus = "llm"
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
    payload: AgentPayload = Field(default_factory=AgentPayload)
    raw_text: str = ""
    error: str | None = None


class MeetingTranscript(BaseModel):
    turns: list[MeetingTurn] = Field(default_factory=list)

    def add(self, turn: MeetingTurn) -> None:
        self.turns.append(turn)


class MeetingState(BaseModel):
    idea: str
    project_mode: ProjectMode = "new_saas"
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
