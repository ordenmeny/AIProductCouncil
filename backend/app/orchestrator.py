from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from backend.app.agents.roles import AGENTS, PHASE_INSTRUCTIONS, AgentDefinition
from backend.app.core.config import Settings
from backend.app.exporters import build_documents, build_vote_summary
from backend.app.llm.client import LLMClient
from backend.app.models import (
    AgentMessage,
    AgentPhase,
    AgentStructuredResponse,
    ClarifyingQuestion,
    MeetingPhase,
    MeetingState,
)


class MeetingOrchestrator:
    def __init__(self, llm: LLMClient, settings: Settings):
        self._llm = llm
        self._settings = settings

    async def create_meeting(self, idea: str) -> MeetingState:
        meeting = MeetingState(idea=idea, phase=MeetingPhase.CLARIFYING_QUESTIONS)
        await self._run_clarifying_questions(meeting)
        meeting.phase = MeetingPhase.WAITING_USER_ANSWERS
        return meeting

    async def advance(self, meeting: MeetingState) -> MeetingState:
        if meeting.phase == MeetingPhase.WAITING_USER_ANSWERS:
            meeting.phase = MeetingPhase.INDIVIDUAL_ANALYSIS
            await self._run_agent_phase(meeting, AgentPhase.INDIVIDUAL_ANALYSIS)
            return meeting
        if meeting.phase == MeetingPhase.INDIVIDUAL_ANALYSIS:
            meeting.phase = MeetingPhase.DEBATE
            await self._run_agent_phase(meeting, AgentPhase.DEBATE)
            return meeting
        if meeting.phase == MeetingPhase.DEBATE:
            meeting.phase = MeetingPhase.MVP_PROPOSALS
            await self._run_agent_phase(meeting, AgentPhase.MVP_PROPOSAL)
            return meeting
        if meeting.phase == MeetingPhase.MVP_PROPOSALS:
            meeting.phase = MeetingPhase.VOTE
            await self._run_agent_phase(meeting, AgentPhase.VOTE)
            meeting.vote_summary = build_vote_summary(meeting.messages)
            return meeting
        if meeting.phase == MeetingPhase.VOTE:
            meeting.phase = MeetingPhase.FINAL_REPORT
            meeting.vote_summary = meeting.vote_summary or build_vote_summary(meeting.messages)
            meeting.final_documents = build_documents(meeting)
            meeting.phase = MeetingPhase.COMPLETED
            return meeting
        if meeting.phase == MeetingPhase.COMPLETED:
            return meeting
        raise ValueError(f"Meeting cannot be advanced from phase {meeting.phase}")

    async def _run_clarifying_questions(self, meeting: MeetingState) -> None:
        for agent in AGENTS:
            response = await self._call_agent(meeting, agent, AgentPhase.CLARIFYING_QUESTION)
            question_text = _first_non_empty(
                response.open_questions,
                response.summary,
                "Что важно уточнить перед проектированием MVP?",
            )
            meeting.questions.append(
                ClarifyingQuestion(
                    agent_id=agent.role.id,
                    agent_name=agent.role.name,
                    question=question_text,
                    reason=response.reason or (response.insights[0] if response.insights else response.summary),
                )
            )
            meeting.messages.append(
                AgentMessage(
                    agent_id=agent.role.id,
                    agent_name=agent.role.name,
                    phase=AgentPhase.CLARIFYING_QUESTION,
                    content=question_text,
                    structured=response,
                )
            )

    async def _run_agent_phase(self, meeting: MeetingState, phase: AgentPhase) -> None:
        for agent in AGENTS:
            response = await self._call_agent(meeting, agent, phase)
            meeting.messages.append(
                AgentMessage(
                    agent_id=agent.role.id,
                    agent_name=agent.role.name,
                    phase=phase,
                    content=_render_agent_response(response),
                    structured=response,
                )
            )

    async def _call_agent(self, meeting: MeetingState, agent: AgentDefinition, phase: AgentPhase) -> AgentStructuredResponse:
        system_prompt = _system_prompt(agent, phase)
        user_prompt = _user_prompt(meeting, agent, phase)
        raw = ""
        last_error = ""
        for attempt in range(self._settings.llm_json_retries + 1):
            try:
                raw = await self._llm.complete_json(system_prompt, user_prompt if attempt == 0 else _repair_prompt(raw, last_error))
                return _parse_agent_response(raw, agent, phase)
            except Exception as exc:  # noqa: BLE001 - fallback is part of orchestration resilience.
                last_error = str(exc)
                if attempt >= self._settings.llm_json_retries:
                    meeting.errors.append(f"{agent.role.name} / {phase}: {last_error}")
                    return _fallback_response(agent, phase, last_error)

        return _fallback_response(agent, phase, last_error or "Unknown LLM error")


def _system_prompt(agent: AgentDefinition, phase: AgentPhase) -> str:
    return f"""
Ты агент в симуляции рабочего созвона IT-команды.
Твоя роль: {agent.role.name}.
Публичный фокус роли: {agent.role.public_focus}
Приватный профессиональный контекст: {agent.private_context}
Правила стиля: {agent.style_rules}

Инструкция текущей фазы: {PHASE_INSTRUCTIONS[phase]}

Верни только JSON object без Markdown.
Используй только эти поля:
{{
  "agent": "{agent.role.name}",
  "agent_id": "{agent.role.id}",
  "phase": "{phase}",
  "summary": "краткий вывод агента",
  "mvp_priority": ["1-5 конкретных пунктов MVP"],
  "roadmap_items": ["пункты roadmap, если уместно"],
  "open_questions": ["важные вопросы заказчику"],
  "insights": ["полезные инсайты"],
  "risks": ["риски"],
  "main_risk": "главный риск",
  "decision": null,
  "next_step": "следующий шаг",
  "reason": "обоснование"
}}

Обязательные правила:
- agent должен быть "{agent.role.name}".
- agent_id должен быть "{agent.role.id}".
- phase должен быть "{phase}".
- В фазе vote decision должен быть одним из: "go", "go_after_clarification", "no_go", "pivot_or_narrow_mvp".
- В остальных фазах decision должен быть null.
- Пиши на русском.
- Будь конкретным: функции, риски, шаги, решения, критерии.
- В фазе clarifying_question задай вопрос строго из своей роли и не повторяй вопросы, которые уже есть во входном контексте.
- Не раскрывай приватный контекст как отдельный блок; используй его в рассуждении.
""".strip()


def _user_prompt(meeting: MeetingState, agent: AgentDefinition, phase: AgentPhase) -> str:
    return json.dumps(
        {
            "meeting_goal": "Спроектировать MVP сервиса/стартапа/фичи и согласовать реалистичный следующий шаг.",
            "idea": meeting.idea,
            "current_phase": phase,
            "your_role": agent.role.model_dump(mode="json"),
            "clarifying_questions": [question.model_dump(mode="json") for question in meeting.questions],
            "user_answers": [answer.model_dump(mode="json") for answer in meeting.user_answers],
            "previous_messages": [
                {
                    "agent": message.agent_name,
                    "phase": message.phase,
                    "content": message.content,
                    "structured": message.structured.model_dump(mode="json") if message.structured else None,
                }
                for message in meeting.messages[-20:]
                if message.agent_id != agent.role.id or phase == AgentPhase.DEBATE
            ],
            "expected_output": "A single JSON object matching the schema from the system prompt.",
        },
        ensure_ascii=False,
    )


def _repair_prompt(raw: str, error: str) -> str:
    return json.dumps(
        {
            "task": "Исправь предыдущий ответ. Верни один валидный JSON object по схеме из system prompt.",
            "invalid_response": raw,
            "validation_error": error,
        },
        ensure_ascii=False,
    )


def _parse_agent_response(raw: str, agent: AgentDefinition, phase: AgentPhase) -> AgentStructuredResponse:
    raw = _extract_json_object(raw)
    try:
        payload: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    payload.setdefault("agent", agent.role.name)
    payload.setdefault("agent_id", agent.role.id)
    payload.setdefault("phase", phase)
    try:
        parsed = AgentStructuredResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(exc.errors(include_url=False)) from exc
    if parsed.agent_id != agent.role.id:
        parsed.agent_id = agent.role.id
    if parsed.phase != phase:
        parsed.phase = phase
    if not parsed.agent:
        parsed.agent = agent.role.name
    return parsed


def _fallback_response(agent: AgentDefinition, phase: AgentPhase, error: str) -> AgentStructuredResponse:
    role_questions = {
        "product_manager": (
            "Для какого сегмента пользователей первая версия должна дать измеримую пользу, и какой результат будет считаться успехом?"
        ),
        "tech_lead": (
            "Какие внешние системы, данные или ограничения инфраструктуры обязательно нужно учесть в первой версии?"
        ),
        "ux_researcher": (
            "Какой один основной сценарий пользователь должен пройти в MVP от начала до результата?"
        ),
        "security_data_expert": (
            "Какие типы данных будет обрабатывать сервис и кто должен иметь к ним доступ?"
        ),
        "skeptic_risk_officer": (
            "Что может сделать проект бесполезным для заказчика даже при технически успешной реализации?"
        ),
    }
    defaults = {
        AgentPhase.CLARIFYING_QUESTION: {
            "summary": "Модель не вернула валидный вопрос, зафиксирован технический fallback.",
            "open_questions": [role_questions.get(agent.role.id, "Что критично уточнить перед проектированием MVP?")],
            "reason": "Вопрос выбран из профессионального fallback-профиля агента из-за ошибки ответа модели.",
        },
        AgentPhase.INDIVIDUAL_ANALYSIS: {
            "summary": "Нужно сузить первую версию до одного проверяемого пользовательского сценария.",
            "risks": ["Недостаточно конкретики для оценки scope."],
            "insights": ["Fallback использован из-за ошибки ответа модели."],
        },
        AgentPhase.DEBATE: {
            "summary": "Совет должен согласовать MVP вокруг одного сценария и не расширять scope до проверки спроса.",
            "risks": ["Разрастание требований до подтверждения ценности."],
        },
        AgentPhase.MVP_PROPOSAL: {
            "summary": "В v1 стоит включить только ввод заявки, основной workflow, базовую аналитику и экспорт результата.",
            "mvp_priority": ["Один основной пользовательский сценарий", "Минимальная форма ввода", "Итоговый отчет"],
        },
        AgentPhase.VOTE: {
            "summary": "Проект можно рассматривать только после уточнения scope и критерия успеха.",
            "decision": "go_after_clarification",
            "mvp_priority": ["Сузить MVP до одного сценария"],
            "roadmap_items": ["Неделя 1: уточнить критерий успеха", "Недели 2-4: собрать MVP"],
            "risks": ["Невалидированный scope"],
            "main_risk": "Невалидированный scope",
            "next_step": "Провести короткое уточнение требований и зафиксировать MVP.",
            "reason": error,
        },
    }
    return AgentStructuredResponse(
        agent=agent.role.name,
        agent_id=agent.role.id,
        phase=phase,
        **defaults[phase],
    )


def _render_agent_response(response: AgentStructuredResponse) -> str:
    parts = [response.summary.strip()] if response.summary else []
    if response.mvp_priority:
        parts.append("MVP: " + "; ".join(response.mvp_priority))
    if response.risks:
        parts.append("Риски: " + "; ".join(response.risks))
    if response.open_questions:
        parts.append("Вопросы: " + "; ".join(response.open_questions))
    if response.decision:
        parts.append(f"Решение: {response.decision}")
    if response.reason:
        parts.append("Обоснование: " + response.reason)
    return "\n\n".join(parts).strip()


def _first_non_empty(items: list[str], fallback: str, default: str) -> str:
    for item in items:
        if item.strip():
            return item.strip()
    if fallback.strip():
        return fallback.strip()
    return default


def _extract_json_object(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped
