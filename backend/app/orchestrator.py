from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from backend.app.agents.roles import AGENTS, PHASE_INSTRUCTIONS, AgentDefinition
from backend.app.core.config import Settings
from backend.app.exporters import build_documents, build_vote_summary
from backend.app.llm.client import LLMClient
from backend.app.models import (
    AgentId,
    AgentMessage,
    AgentPhase,
    AgentStructuredResponse,
    ClarifyingQuestion,
    MeetingPhase,
    MeetingState,
)

ROLE_SPECIFIC_FIELDS = {
    "mvp_priority",
    "roadmap_items",
    "risks",
    "target_audience",
    "user_problem",
    "core_mvp_features",
    "tech_stack",
    "tech_stack_reasoning",
    "user_scenario",
    "user_screens",
    "processed_data",
    "data_sensitivity",
    "security_measures",
    "risk_mitigations",
}

ALLOWED_ROLE_FIELDS: dict[AgentId, set[str]] = {
    AgentId.PRODUCT: {"target_audience", "user_problem", "core_mvp_features", "mvp_priority"},
    AgentId.TECH: {"tech_stack", "tech_stack_reasoning", "roadmap_items"},
    AgentId.UX: {"user_scenario", "user_screens"},
    AgentId.SECURITY: {"processed_data", "data_sensitivity", "security_measures", "risks"},
    AgentId.SKEPTIC: {"risks", "risk_mitigations"},
}


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
    role_requirements = _role_requirements(agent)
    json_contract = _json_contract(agent, phase)
    return f"""
Ты агент в симуляции рабочего созвона IT-команды.
Твоя роль: {agent.role.name}.
Публичный фокус роли: {agent.role.public_focus}
Приватный профессиональный контекст: {agent.private_context}
Правила стиля: {agent.style_rules}

Инструкция текущей фазы: {PHASE_INSTRUCTIONS[phase]}

Верни только JSON object без Markdown.
Используй только общие поля и поля своей роли из этого контракта:
{json_contract}

Обязательные поля для твоей роли:
{role_requirements}

Обязательные правила:
- agent должен быть "{agent.role.name}".
- agent_id должен быть "{agent.role.id}".
- phase должен быть "{phase}".
- В фазе vote decision должен быть одним из: "go", "go_after_clarification", "no_go", "pivot_or_narrow_mvp".
- В остальных фазах decision должен быть null.
- Пиши на русском.
- Отвечай только в своей профессиональной зоне. Учитывай чужие аргументы, но не заполняй и не переопределяй чужие разделы.
- Product не выбирает стек; Tech Lead не формулирует ЦА; UX не пишет security policy; Security не проектирует фичи; Skeptic не заменяет остальных.
- Будь максимально конкретным: функции, риски, шаги, технологии, данные, экраны, решения, критерии.
- Минимум воды: короткие пункты, 1 мысль = 1 пункт, без общих фраз вроде "улучшить UX" без объяснения как.
- Все пункты должны быть привязаны к идее пользователя, а не к абстрактному стартапу.
- Не возвращай больше 4 пунктов в каждом списке. Каждый пункт максимум 18 слов.
- summary, reason и next_step: по одному короткому предложению.
- В фазе clarifying_question задай вопрос строго из своей роли и не повторяй вопросы, которые уже есть во входном контексте.
- Не раскрывай приватный контекст как отдельный блок; используй его в рассуждении.
""".strip()


def _json_contract(agent: AgentDefinition, phase: AgentPhase) -> str:
    contract: dict[str, Any] = {
        "agent": agent.role.name,
        "agent_id": str(agent.role.id),
        "phase": str(phase),
        "summary": "краткий вывод агента",
        "open_questions": ["важные вопросы заказчику"],
        "insights": ["полезные инсайты"],
        "main_risk": "главный риск, если он относится к твоей зоне",
        "decision": None if phase != AgentPhase.VOTE else "go_after_clarification",
        "next_step": "следующий шаг из твоей роли",
        "reason": "обоснование",
    }
    role_fields: dict[AgentId, dict[str, Any]] = {
        AgentId.PRODUCT: {
            "target_audience": ["конкретные сегменты пользователей"],
            "user_problem": ["проблемы пользователей, которые решает продукт"],
            "core_mvp_features": ["полезные и необходимые функции MVP"],
            "mvp_priority": ["приоритеты MVP из продуктовой зоны"],
        },
        AgentId.TECH: {
            "tech_stack": ["технологии MVP"],
            "tech_stack_reasoning": ["почему выбрана каждая технология"],
            "roadmap_items": ["технические шаги реализации"],
        },
        AgentId.UX: {
            "user_scenario": ["шаги пользовательского сценария"],
            "user_screens": ["экраны, страницы или модули MVP"],
        },
        AgentId.SECURITY: {
            "processed_data": ["обрабатываемые данные"],
            "data_sensitivity": ["чувствительность данных"],
            "security_measures": ["меры защиты данных"],
            "risks": ["уязвимости и abuse cases"],
        },
        AgentId.SKEPTIC: {
            "risks": ["конкретные риски проекта"],
            "risk_mitigations": ["способы снижения рисков"],
        },
    }
    contract.update(role_fields.get(agent.role.id, {}))
    return json.dumps(contract, ensure_ascii=False, indent=2)


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
            "expected_output": "A single JSON object matching the system prompt. Fill only your role-specific fields.",
        },
        ensure_ascii=False,
    )


def _role_requirements(agent: AgentDefinition) -> str:
    requirements = {
        "product_manager": (
            "- target_audience: 2-4 конкретных сегмента.\n"
            "- user_problem: 2-5 конкретных пользовательских проблем.\n"
            "- core_mvp_features: 3-7 полезных и необходимых функций MVP.\n"
            "- mvp_priority должен дублировать или уточнять core_mvp_features.\n"
            "- Не выбирай технологии, экраны, security-меры и mitigation чужих рисков."
        ),
        "tech_lead": (
            "- tech_stack: список конкретных технологий MVP: frontend, backend, storage, LLM/API, deploy/test минимум.\n"
            "- tech_stack_reasoning: кратко почему выбран этот стек и почему он выгоден для MVP.\n"
            "- roadmap_items должен отражать техническую последовательность реализации.\n"
            "- Не формулируй ЦА, пользовательские проблемы, UX-экраны и security policy."
        ),
        "ux_researcher": (
            "- user_scenario: 4-8 шагов пользовательского сценария MVP.\n"
            "- user_screens: список экранов/страниц/модулей, которые нужно реализовать.\n"
            "- insights должен включать UX-инсайты и потенциальные точки трения.\n"
            "- Не выбирай стек, не определяй защиту данных и не пересобирай продуктовый MVP."
        ),
        "security_data_expert": (
            "- processed_data: список данных, которые сервис будет обрабатывать.\n"
            "- data_sensitivity: чувствительность этих данных: низкая/средняя/высокая и почему.\n"
            "- security_measures: конкретные меры защиты данных и доступа.\n"
            "- risks должен включать уязвимости и abuse cases.\n"
            "- Не проектируй пользовательские фичи, ЦА, UX-экраны и стек."
        ),
        "skeptic_risk_officer": (
            "- risks: конкретные риски проекта, не общие опасения.\n"
            "- risk_mitigations: практические способы снижения каждого ключевого риска.\n"
            "- main_risk должен быть самым критичным риском для MVP.\n"
            "- Не заменяй Product, Tech, UX и Security: только риски, последствия и способы снижения."
        ),
    }
    return requirements.get(agent.role.id, "- Заполни поля, релевантные своей роли.")


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
        repaired = _repair_truncated_json(raw)
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError as repair_exc:
            raise ValueError(f"Invalid JSON: {exc}") from repair_exc
    payload = _normalize_payload(payload)
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
    return _sanitize_response_for_role(parsed)


def _sanitize_response_for_role(response: AgentStructuredResponse) -> AgentStructuredResponse:
    allowed = ALLOWED_ROLE_FIELDS.get(response.agent_id, set())
    data = response.model_dump()
    for field in ROLE_SPECIFIC_FIELDS - allowed:
        if field in data:
            data[field] = []
    return AgentStructuredResponse.model_validate(data)


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    list_string_fields = [
        "mvp_priority",
        "roadmap_items",
        "open_questions",
        "insights",
        "risks",
        "target_audience",
        "user_problem",
        "core_mvp_features",
        "tech_stack",
        "tech_stack_reasoning",
        "user_scenario",
        "user_screens",
        "processed_data",
        "data_sensitivity",
        "security_measures",
        "risk_mitigations",
    ]
    for field in list_string_fields:
        if field in normalized:
            normalized[field] = _normalize_string_list(normalized[field])

    risk_mitigations = list(normalized.get("risk_mitigations") or [])
    for item in _as_list(payload.get("risks")):
        if isinstance(item, dict):
            mitigation = _first_dict_value(item, ["mitigation", "mitigation_action", "solution", "fix", "recommendation"])
            if mitigation:
                risk_mitigations.append(str(mitigation))
    if risk_mitigations:
        normalized["risk_mitigations"] = _dedupe_strings(risk_mitigations)
    return normalized


def _repair_truncated_json(raw: str) -> str:
    text = raw.strip()
    if not text:
        return text

    in_string = False
    escaped = False
    stack: list[str] = []
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()

    if escaped:
        text = text[:-1]
    if in_string:
        text += '"'

    text = text.rstrip()
    while text and text[-1] in ",:":
        text = text[:-1].rstrip()

    for closer in reversed(stack):
        text += closer
    return text


def _normalize_string_list(value: Any) -> list[str]:
    result: list[str] = []
    for item in _as_list(value):
        text = _stringify_item(item)
        if text:
            result.append(text)
    return _dedupe_strings(result)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _stringify_item(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, (int, float, bool)):
        return str(item)
    if isinstance(item, dict):
        return _stringify_dict_item(item)
    return str(item).strip()


def _stringify_dict_item(item: dict[str, Any]) -> str:
    preferred_keys = [
        "risk",
        "title",
        "name",
        "item",
        "feature",
        "technology",
        "screen",
        "data",
        "measure",
        "problem",
        "audience",
        "step",
    ]
    main = _first_dict_value(item, preferred_keys)
    parts = [str(main).strip()] if main else []

    consequence = _first_dict_value(item, ["consequence", "impact", "effect"])
    if consequence:
        parts.append(f"последствие: {consequence}")

    reason = _first_dict_value(item, ["reason", "why", "rationale"])
    if reason:
        parts.append(f"почему: {reason}")

    mitigation = _first_dict_value(item, ["mitigation", "mitigation_action", "solution", "fix", "recommendation"])
    if mitigation:
        parts.append(f"снижение: {mitigation}")

    if parts:
        return "; ".join(parts)

    fallback_parts = [f"{key}: {value}" for key, value in item.items() if value not in (None, "", [])]
    return "; ".join(fallback_parts).strip()


def _first_dict_value(item: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, "", []):
            return value
    return None


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


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
    role_defaults = _role_fallback_fields(agent.role.id)
    return AgentStructuredResponse(
        agent=agent.role.name,
        agent_id=agent.role.id,
        phase=phase,
        **defaults[phase],
        **role_defaults,
    )


def _role_fallback_fields(agent_id: str) -> dict[str, list[str]]:
    fields = {
        "product_manager": {
            "target_audience": ["Основной сегмент пользователей первой версии", "Владелец процесса или заказчик MVP"],
            "user_problem": ["Пользовательская проблема и критерий успеха пока недостаточно уточнены"],
            "core_mvp_features": ["Один основной сценарий", "Минимальная форма ввода", "Итоговый отчет или результат действия"],
        },
        "tech_lead": {
            "tech_stack": ["Python + FastAPI", "React + Vite", "JSON/file storage для прототипа", "LM Studio OpenAI-compatible API", "pytest для проверки логики"],
            "tech_stack_reasoning": [
                "FastAPI быстро дает типизированный API для MVP",
                "React + Vite ускоряют сборку простого интерфейса",
                "Файловое JSON-хранилище достаточно для прототипа без лишней инфраструктуры",
                "LM Studio позволяет тестировать локальные модели без облака",
            ],
        },
        "ux_researcher": {
            "user_scenario": [
                "Пользователь открывает сервис",
                "Вводит исходную идею или задачу",
                "Отвечает на уточняющие вопросы",
                "Просматривает результаты анализа и итоговый план",
            ],
            "user_screens": ["Стартовая форма", "Экран вопросов", "Лента обсуждения", "Итоговый план", "Экспорт документов"],
        },
        "security_data_expert": {
            "processed_data": ["Текст идеи", "Ответы пользователя", "История обсуждения", "Итоговые документы"],
            "data_sensitivity": ["Средняя: может содержать бизнес-идеи, внутренние процессы или персональные данные"],
            "security_measures": ["Локальное хранение", "Ограничение доступа к файлам состояния", "Не логировать секреты", "Минимизировать срок хранения"],
        },
        "skeptic_risk_officer": {
            "risk_mitigations": [
                "Сузить MVP до одного сценария",
                "Заранее определить критерий успеха",
                "Проверить спрос на коротком пилоте",
                "Не добавлять интеграции до подтверждения ценности",
            ],
        },
    }
    return fields.get(agent_id, {})


def _render_agent_response(response: AgentStructuredResponse) -> str:
    parts = [response.summary.strip()] if response.summary else []
    allowed = ALLOWED_ROLE_FIELDS.get(response.agent_id, set())
    if "target_audience" in allowed and response.target_audience:
        parts.append("ЦА: " + "; ".join(response.target_audience))
    if "user_problem" in allowed and response.user_problem:
        parts.append("Проблемы: " + "; ".join(response.user_problem))
    if "core_mvp_features" in allowed and response.core_mvp_features:
        parts.append("Функции MVP: " + "; ".join(response.core_mvp_features))
    if "mvp_priority" in allowed and response.mvp_priority:
        parts.append("MVP: " + "; ".join(response.mvp_priority))
    if "tech_stack" in allowed and response.tech_stack:
        parts.append("Стек: " + "; ".join(response.tech_stack))
    if "tech_stack_reasoning" in allowed and response.tech_stack_reasoning:
        parts.append("Почему стек: " + "; ".join(response.tech_stack_reasoning))
    if "roadmap_items" in allowed and response.roadmap_items:
        parts.append("Roadmap: " + "; ".join(response.roadmap_items))
    if "user_scenario" in allowed and response.user_scenario:
        parts.append("Сценарий: " + "; ".join(response.user_scenario))
    if "user_screens" in allowed and response.user_screens:
        parts.append("Экраны: " + "; ".join(response.user_screens))
    if "processed_data" in allowed and response.processed_data:
        parts.append("Данные: " + "; ".join(response.processed_data))
    if "data_sensitivity" in allowed and response.data_sensitivity:
        parts.append("Чувствительность: " + "; ".join(response.data_sensitivity))
    if "security_measures" in allowed and response.security_measures:
        parts.append("Защита: " + "; ".join(response.security_measures))
    if "risks" in allowed and response.risks:
        parts.append("Риски: " + "; ".join(response.risks))
    if "risk_mitigations" in allowed and response.risk_mitigations:
        parts.append("Снижение рисков: " + "; ".join(response.risk_mitigations))
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
