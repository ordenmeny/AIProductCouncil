from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


PhaseName = Literal[
    "questions",
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


class AgentRole(BaseModel):
    name: str
    slug: str
    description: str
    system_prompt: str
    private_context_path: Path


class AgentResponse(BaseModel):
    agent: str
    phase: PhaseName
    summary: str = Field(default="")
    questions: list[str] = Field(default_factory=list)
    arguments: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    mvp_features: list[str] = Field(default_factory=list)
    decision: Decision = "unknown"
    next_step: str = ""
    confidence: int = Field(default=3, ge=1, le=5)
    raw_text: str = ""
    is_fallback: bool = False
    error: str | None = None


class MeetingState(BaseModel):
    idea: str
    started_at: datetime = Field(default_factory=datetime.now)
    phases: dict[PhaseName, list[AgentResponse]] = Field(default_factory=dict)
    final_report: str = ""

    def add_response(self, response: AgentResponse) -> None:
        self.phases.setdefault(response.phase, []).append(response)

    @property
    def votes(self) -> list[AgentResponse]:
        return self.phases.get("mvp_vote", [])

    def to_export_dict(self) -> dict:
        return self.model_dump(mode="json")
