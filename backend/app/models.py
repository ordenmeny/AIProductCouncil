from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentId(StrEnum):
    PRODUCT = "product_manager"
    TECH = "tech_lead"
    UX = "ux_researcher"
    SECURITY = "security_data_expert"
    SKEPTIC = "skeptic_risk_officer"


class MeetingPhase(StrEnum):
    INTAKE = "intake"
    CLARIFYING_QUESTIONS = "clarifying_questions"
    WAITING_USER_ANSWERS = "waiting_user_answers"
    INDIVIDUAL_ANALYSIS = "individual_analysis"
    DEBATE = "debate"
    MVP_PROPOSALS = "mvp_proposals"
    VOTE = "vote"
    FINAL_REPORT = "final_report"
    COMPLETED = "completed"


class AgentPhase(StrEnum):
    CLARIFYING_QUESTION = "clarifying_question"
    INDIVIDUAL_ANALYSIS = "individual_analysis"
    DEBATE = "debate"
    MVP_PROPOSAL = "mvp_proposal"
    VOTE = "vote"


class VoteDecision(StrEnum):
    GO = "go"
    GO_AFTER_CLARIFICATION = "go_after_clarification"
    NO_GO = "no_go"
    PIVOT_OR_NARROW_MVP = "pivot_or_narrow_mvp"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class CreateMeetingRequest(BaseModel):
    idea: str = Field(min_length=10, max_length=8000)


class CreateMeetingResponse(BaseModel):
    meeting: "MeetingState"


class AgentRole(BaseModel):
    id: AgentId
    name: str
    short_name: str
    public_focus: str


class ClarifyingQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: AgentId
    agent_name: str
    question: str
    reason: str = ""


class UserAnswer(BaseModel):
    question_id: str
    answer: str = Field(min_length=1, max_length=4000)


class SubmitAnswersRequest(BaseModel):
    answers: list[UserAnswer]


class AgentStructuredResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent: str
    agent_id: AgentId
    phase: AgentPhase
    summary: str = Field(default="", max_length=2500)
    mvp_priority: list[str] = Field(default_factory=list)
    roadmap_items: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    main_risk: str = ""
    decision: VoteDecision | None = None
    next_step: str = ""
    reason: str = ""


class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=now_utc)
    agent_id: AgentId
    agent_name: str
    phase: AgentPhase
    content: str
    structured: AgentStructuredResponse | None = None
    raw_response: str | None = None
    validation_error: str | None = None


class VoteSummary(BaseModel):
    decisions: dict[VoteDecision, int] = Field(default_factory=dict)
    final_decision: VoteDecision | None = None
    key_mvp_features: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    main_next_step: str = ""


class FinalDocuments(BaseModel):
    protocol_md: str = ""
    final_plan_md: str = ""


class MeetingState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)
    idea: str
    phase: MeetingPhase = MeetingPhase.INTAKE
    questions: list[ClarifyingQuestion] = Field(default_factory=list)
    user_answers: list[UserAnswer] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    vote_summary: VoteSummary | None = None
    final_documents: FinalDocuments | None = None
    errors: list[str] = Field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = now_utc()


class AdvanceMeetingResponse(BaseModel):
    meeting: MeetingState
    advanced_to: MeetingPhase


class ApiError(BaseModel):
    detail: str
    code: str = "bad_request"


MeetingState.model_rebuild()
CreateMeetingResponse.model_rebuild()
