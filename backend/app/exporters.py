from __future__ import annotations

from collections import Counter

from backend.app.models import AgentId, AgentMessage, AgentPhase, AgentStructuredResponse, FinalDocuments, MeetingState, VoteDecision, VoteSummary


def _bullet(items: list[str]) -> str:
    if not items:
        return "- Не зафиксировано."
    return "\n".join(f"- {item}" for item in items)


def _phase_title(phase: AgentPhase) -> str:
    return {
        AgentPhase.CLARIFYING_QUESTION: "Уточняющие вопросы",
        AgentPhase.INDIVIDUAL_ANALYSIS: "Индивидуальный анализ",
        AgentPhase.DEBATE: "Обсуждение и спор",
        AgentPhase.MVP_PROPOSAL: "Предложения по MVP",
        AgentPhase.VOTE: "Голосование",
    }[phase]


def build_vote_summary(messages: list[AgentMessage]) -> VoteSummary:
    vote_messages = [
        message.structured
        for message in messages
        if message.phase == AgentPhase.VOTE and message.structured and message.structured.decision
    ]
    decisions_counter = Counter(response.decision for response in vote_messages if response.decision)
    final_decision = None
    if decisions_counter:
        final_decision = decisions_counter.most_common(1)[0][0]

    mvp_features: list[str] = []
    risks: list[str] = []
    questions: list[str] = []
    insights: list[str] = []
    next_steps: list[str] = []
    for response in vote_messages:
        mvp_features.extend(response.mvp_priority)
        risks.extend(response.risks or ([response.main_risk] if response.main_risk else []))
        questions.extend(response.open_questions)
        insights.extend(response.insights)
        if response.next_step:
            next_steps.append(response.next_step)

    return VoteSummary(
        decisions={decision: count for decision, count in decisions_counter.items()},
        final_decision=final_decision,
        key_mvp_features=_dedupe(mvp_features)[:10],
        key_risks=_dedupe(risks)[:10],
        open_questions=_dedupe(questions)[:10],
        insights=_dedupe(insights)[:10],
        main_next_step=next_steps[0] if next_steps else "Сузить MVP и согласовать владельца следующего шага.",
    )


def build_documents(meeting: MeetingState) -> FinalDocuments:
    summary = meeting.vote_summary or build_vote_summary(meeting.messages)
    protocol = build_protocol(meeting)
    final_plan = build_final_plan(meeting, summary)
    return FinalDocuments(protocol_md=protocol, final_plan_md=final_plan)


def build_protocol(meeting: MeetingState) -> str:
    lines = [
        "# Протокол AI Product Council",
        "",
        f"**Идея:** {meeting.idea}",
        "",
        "## Вопросы агентов и ответы пользователя",
        "",
    ]
    answer_by_question = {answer.question_id: answer.answer for answer in meeting.user_answers}
    if not meeting.questions:
        lines.append("- Вопросы не были сформированы.")
    for question in meeting.questions:
        lines.extend(
            [
                f"### {question.agent_name}",
                "",
                f"**Вопрос:** {question.question}",
                "",
                f"**Почему важно:** {question.reason or 'Не указано.'}",
                "",
                f"**Ответ пользователя:** {answer_by_question.get(question.id, 'Нет ответа.')}",
                "",
            ]
        )

    for phase in [
        AgentPhase.INDIVIDUAL_ANALYSIS,
        AgentPhase.DEBATE,
        AgentPhase.MVP_PROPOSAL,
        AgentPhase.VOTE,
    ]:
        phase_messages = [message for message in meeting.messages if message.phase == phase]
        if not phase_messages:
            continue
        lines.extend(["", f"## {_phase_title(phase)}", ""])
        for message in phase_messages:
            lines.extend([f"### {message.agent_name}", "", message.content.strip() or "Нет содержимого.", ""])
            if message.validation_error:
                lines.extend([f"Ошибка валидации ответа модели: `{message.validation_error}`", ""])

    return "\n".join(lines).strip() + "\n"


def build_final_plan(meeting: MeetingState, summary: VoteSummary) -> str:
    vote_label = _decision_label(summary.final_decision)
    roadmap = _collect_roadmap(meeting.messages)
    agent_briefs = _build_agent_briefs(meeting.messages)
    return "\n".join(
        [
            "# Итоговый план проекта",
            "",
            "## Суть задачи",
            "",
            meeting.idea,
            "",
            "## Решение совета",
            "",
            f"**Вердикт:** {vote_label}",
            "",
            "## MVP",
            "",
            _bullet(summary.key_mvp_features),
            "",
            "## Roadmap на 4-6 недель",
            "",
            _bullet(roadmap),
            "",
            "## Выжимки агентов",
            "",
            agent_briefs,
            "",
            "## Ключевые риски",
            "",
            _bullet(summary.key_risks),
            "",
            "## Открытые вопросы",
            "",
            _bullet(summary.open_questions),
            "",
            "## Инсайты",
            "",
            _bullet(summary.insights),
            "",
            "## Главный следующий шаг",
            "",
            summary.main_next_step,
            "",
        ]
    )


def _build_agent_briefs(messages: list[AgentMessage]) -> str:
    responses = _latest_agent_responses(messages)
    sections = [
        _product_brief(responses.get(AgentId.PRODUCT)),
        _tech_brief(responses.get(AgentId.TECH)),
        _ux_brief(responses.get(AgentId.UX)),
        _security_brief(responses.get(AgentId.SECURITY)),
        _skeptic_brief(responses.get(AgentId.SKEPTIC)),
    ]
    return "\n\n".join(section for section in sections if section.strip())


def _latest_agent_responses(messages: list[AgentMessage]) -> dict[AgentId, AgentStructuredResponse | None]:
    result: dict[AgentId, AgentStructuredResponse | None] = {}
    for agent_id in AgentId:
        agent_messages = [message for message in messages if message.agent_id == agent_id and message.structured]
        vote_response = next(
            (message.structured for message in reversed(agent_messages) if message.phase == AgentPhase.VOTE and message.structured),
            None,
        )
        result[agent_id] = vote_response or _merge_latest_non_empty(agent_messages)
    return result


def _merge_latest_non_empty(messages: list[AgentMessage]) -> AgentStructuredResponse | None:
    if not messages:
        return None
    base = messages[-1].structured
    if not base:
        return None
    data = base.model_dump()
    list_fields = [
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
        "risks",
        "risk_mitigations",
    ]
    for field in list_fields:
        if data.get(field):
            continue
        for message in reversed(messages):
            response = message.structured
            values = getattr(response, field, []) if response else []
            if values:
                data[field] = values
                break
    return AgentStructuredResponse.model_validate(data)


def _product_brief(response: AgentStructuredResponse | None) -> str:
    return "\n".join(
        [
            "### Product/Business Manager",
            "",
            "**Целевая аудитория:**",
            _bullet(response.target_audience if response else []),
            "",
            "**Проблема пользователя:**",
            _bullet(response.user_problem if response else []),
            "",
            "**Основные функции MVP:**",
            _bullet((response.core_mvp_features or response.mvp_priority) if response else []),
        ]
    )


def _tech_brief(response: AgentStructuredResponse | None) -> str:
    return "\n".join(
        [
            "### Tech Lead / Architect",
            "",
            "**Технологии MVP:**",
            _bullet(response.tech_stack if response else []),
            "",
            "**Почему выбран этот стек:**",
            _bullet(response.tech_stack_reasoning if response else []),
        ]
    )


def _ux_brief(response: AgentStructuredResponse | None) -> str:
    return "\n".join(
        [
            "### UX Researcher / Designer",
            "",
            "**Пользовательский сценарий MVP:**",
            _bullet(response.user_scenario if response else []),
            "",
            "**Экраны / страницы / модули:**",
            _bullet(response.user_screens if response else []),
        ]
    )


def _security_brief(response: AgentStructuredResponse | None) -> str:
    return "\n".join(
        [
            "### Security / Data Expert",
            "",
            "**Обрабатываемые данные:**",
            _bullet(response.processed_data if response else []),
            "",
            "**Чувствительность данных:**",
            _bullet(response.data_sensitivity if response else []),
            "",
            "**Меры защиты данных:**",
            _bullet(response.security_measures if response else []),
        ]
    )


def _skeptic_brief(response: AgentStructuredResponse | None) -> str:
    return "\n".join(
        [
            "### Skeptic / Risk Officer",
            "",
            "**Основные риски:**",
            _bullet(response.risks if response else []),
            "",
            "**Пути снижения рисков:**",
            _bullet(response.risk_mitigations if response else []),
        ]
    )


def _collect_roadmap(messages: list[AgentMessage]) -> list[str]:
    items: list[str] = []
    for message in messages:
        if message.structured:
            items.extend(message.structured.roadmap_items)
    if items:
        return _dedupe(items)[:8]
    return [
        "Неделя 1: уточнить владельца, целевой сценарий и критерий успеха.",
        "Неделя 2: собрать кликабельный или технический прототип ключевого сценария.",
        "Недели 3-4: реализовать минимальный рабочий MVP.",
        "Недели 5-6: провести пилот, собрать метрики и решить, расширять ли scope.",
    ]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _decision_label(decision: VoteDecision | None) -> str:
    labels = {
        VoteDecision.GO: "запускать проект",
        VoteDecision.GO_AFTER_CLARIFICATION: "запускать после уточнений",
        VoteDecision.NO_GO: "не запускать",
        VoteDecision.PIVOT_OR_NARROW_MVP: "изменить идею или сузить MVP",
    }
    return labels.get(decision, "не определено")
