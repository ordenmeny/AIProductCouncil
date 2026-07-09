from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.app.agents.roles import list_public_roles
from backend.app.core.config import Settings, get_settings
from backend.app.exporters import build_documents
from backend.app.llm.client import LLMClient
from backend.app.models import (
    AdvanceMeetingResponse,
    CreateMeetingRequest,
    CreateMeetingResponse,
    MeetingPhase,
    MeetingState,
    SubmitAnswersRequest,
)
from backend.app.orchestrator import MeetingOrchestrator
from backend.app.storage import MeetingStorage


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="AI Product Council", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()


def get_storage(settings: Settings = Depends(get_settings)) -> MeetingStorage:
    return MeetingStorage(settings.meeting_storage_dir)


def get_orchestrator(settings: Settings = Depends(get_settings)) -> MeetingOrchestrator:
    return MeetingOrchestrator(LLMClient(settings), settings)


def load_meeting_or_404(storage: MeetingStorage, meeting_id: str) -> MeetingState:
    try:
        return storage.get(meeting_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Meeting not found") from exc


@app.get("/api/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model": settings.openai_model,
        "base_url": settings.openai_base_url,
        "response_format_json": settings.llm_use_response_format,
    }


@app.get("/api/llm/health")
async def llm_health(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    try:
        return await LLMClient(settings).healthcheck()
    except Exception as exc:  # noqa: BLE001 - endpoint is diagnostic by design.
        raise HTTPException(
            status_code=502,
            detail=f"LLM Studio healthcheck failed: {exc}",
        ) from exc


@app.get("/api/agents")
def agents():
    return {"agents": list_public_roles()}


@app.post("/api/meetings", response_model=CreateMeetingResponse)
async def create_meeting(
    request: CreateMeetingRequest,
    storage: MeetingStorage = Depends(get_storage),
    orchestrator: MeetingOrchestrator = Depends(get_orchestrator),
) -> CreateMeetingResponse:
    meeting = await orchestrator.create_meeting(request.idea)
    storage.save(meeting)
    return CreateMeetingResponse(meeting=meeting)


@app.get("/api/meetings/{meeting_id}", response_model=MeetingState)
def get_meeting(meeting_id: str, storage: MeetingStorage = Depends(get_storage)) -> MeetingState:
    return load_meeting_or_404(storage, meeting_id)


@app.post("/api/meetings/{meeting_id}/answers", response_model=MeetingState)
def submit_answers(
    meeting_id: str,
    request: SubmitAnswersRequest,
    storage: MeetingStorage = Depends(get_storage),
) -> MeetingState:
    meeting = load_meeting_or_404(storage, meeting_id)
    if meeting.phase != MeetingPhase.WAITING_USER_ANSWERS:
        raise HTTPException(status_code=409, detail=f"Meeting is not waiting for answers, current phase: {meeting.phase}")
    known_question_ids = {question.id for question in meeting.questions}
    unknown = [answer.question_id for answer in request.answers if answer.question_id not in known_question_ids]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown question ids: {unknown}")
    meeting.user_answers = request.answers
    storage.save(meeting)
    return meeting


@app.post("/api/meetings/{meeting_id}/advance", response_model=AdvanceMeetingResponse)
async def advance_meeting(
    meeting_id: str,
    storage: MeetingStorage = Depends(get_storage),
    orchestrator: MeetingOrchestrator = Depends(get_orchestrator),
) -> AdvanceMeetingResponse:
    meeting = load_meeting_or_404(storage, meeting_id)
    if meeting.phase == MeetingPhase.WAITING_USER_ANSWERS and not meeting.user_answers:
        raise HTTPException(status_code=409, detail="Submit answers before advancing the meeting")
    meeting = await orchestrator.advance(meeting)
    storage.save(meeting)
    return AdvanceMeetingResponse(meeting=meeting, advanced_to=meeting.phase)


@app.get("/api/meetings/{meeting_id}/export/protocol.md")
def export_protocol(meeting_id: str, storage: MeetingStorage = Depends(get_storage)) -> Response:
    meeting = load_meeting_or_404(storage, meeting_id)
    docs = meeting.final_documents or build_documents(meeting)
    return Response(
        content=docs.protocol_md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{meeting_id}-protocol.md"'},
    )


@app.get("/api/meetings/{meeting_id}/export/final-plan.md")
def export_final_plan(meeting_id: str, storage: MeetingStorage = Depends(get_storage)) -> Response:
    meeting = load_meeting_or_404(storage, meeting_id)
    docs = meeting.final_documents or build_documents(meeting)
    return Response(
        content=docs.final_plan_md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{meeting_id}-final-plan.md"'},
    )
