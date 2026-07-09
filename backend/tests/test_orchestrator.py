from backend.app.agents.roles import AGENTS
from backend.app.core.config import Settings
from backend.app.models import AgentId
from backend.app.models import AgentPhase, MeetingPhase, UserAnswer
from backend.app.orchestrator import MeetingOrchestrator, _parse_agent_response
from backend.app.agents.roles import AGENTS_BY_ID


class FakeLLM:
    async def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        agent_id = _agent_id_from_prompt(system_prompt)
        role_fields = _role_fields(agent_id)
        if f'"{AgentPhase.CLARIFYING_QUESTION}"' in system_prompt:
            phase = AgentPhase.CLARIFYING_QUESTION
            return (
                '{"agent":"stub","agent_id":"'
                + agent_id
                + '","phase":"clarifying_question",'
                '"summary":"Уточнение","open_questions":["Какой пользовательский сценарий главный?"],'
                '"reason":"Это сузит MVP.",'
                + role_fields
                + "}"
            )
        if f'"{AgentPhase.VOTE}"' in system_prompt:
            return (
                '{"agent":"stub","agent_id":"'
                + agent_id
                + '","phase":"vote","summary":"Голос",'
                '"decision":"go_after_clarification","mvp_priority":["Один сценарий"],'
                '"roadmap_items":["Неделя 1: scope"],"risks":["Размытый scope"],'
                '"open_questions":["Кто владелец?"],"insights":["MVP надо сузить"],'
                '"main_risk":"Размытый scope","next_step":"Зафиксировать MVP","reason":"Реализуемо",'
                + role_fields
                + "}"
            )
        phase = "individual_analysis"
        if f'"{AgentPhase.DEBATE}"' in system_prompt:
            phase = "debate"
        if f'"{AgentPhase.MVP_PROPOSAL}"' in system_prompt:
            phase = "mvp_proposal"
        return (
            '{"agent":"stub","agent_id":"'
            + agent_id
            + '","phase":"'
            + phase
            + '","summary":"Содержательный ответ","mvp_priority":["Фича 1"],"risks":["Риск 1"],'
            + role_fields
            + "}"
        )


def test_orchestrator_reaches_completed_phase():
    settings = Settings(LLM_JSON_RETRIES=0)
    orchestrator = MeetingOrchestrator(FakeLLM(), settings)

    import asyncio

    meeting = asyncio.run(orchestrator.create_meeting("Сервис для согласования заявок внутри компании"))
    assert meeting.phase == MeetingPhase.WAITING_USER_ANSWERS
    assert len(meeting.questions) == len(AGENTS)

    meeting.user_answers = [
        UserAnswer(question_id=question.id, answer="Главный сценарий: быстро согласовать заявку.")
        for question in meeting.questions
    ]
    for _ in range(5):
        meeting = asyncio.run(orchestrator.advance(meeting))

    assert meeting.phase == MeetingPhase.COMPLETED
    assert meeting.vote_summary is not None
    assert meeting.final_documents is not None
    final_plan = meeting.final_documents.final_plan_md
    assert "## Выжимки агентов" in final_plan
    assert "### Product/Business Manager" in final_plan
    assert "### Tech Lead / Architect" in final_plan
    assert "### UX Researcher / Designer" in final_plan
    assert "### Security / Data Expert" in final_plan
    assert "### Skeptic / Risk Officer" in final_plan


def test_parser_normalizes_object_risks_from_llm():
    raw = """
    {
      "agent": "Skeptic / Risk Officer",
      "agent_id": "skeptic_risk_officer",
      "phase": "clarifying_question",
      "summary": "Есть риски",
      "risks": [
        {
          "risk": "Недостаточная узнаваемость бренда GlyphForge",
          "consequence": "Малая вероятность привлечения целевой аудитории.",
          "mitigation": "Разработать стратегию маркетинга и PR."
        }
      ]
    }
    """
    parsed = _parse_agent_response(raw, AGENTS_BY_ID[AgentId.SKEPTIC], AgentPhase.CLARIFYING_QUESTION)

    assert parsed.risks == [
        "Недостаточная узнаваемость бренда GlyphForge; последствие: Малая вероятность привлечения целевой аудитории.; снижение: Разработать стратегию маркетинга и PR."
    ]
    assert parsed.risk_mitigations == ["Разработать стратегию маркетинга и PR."]


def _agent_id_from_prompt(system_prompt: str) -> str:
    for agent_id in AgentId:
        if f'"{agent_id}"' in system_prompt:
            return str(agent_id)
    return str(AgentId.PRODUCT)


def _role_fields(agent_id: str) -> str:
    common = {
        str(AgentId.PRODUCT): (
            '"target_audience":["Операционные менеджеры"],'
            '"user_problem":["Долго согласуют заявки"],'
            '"core_mvp_features":["Форма заявки","Статусы","Экспорт решения"]'
        ),
        str(AgentId.TECH): (
            '"tech_stack":["FastAPI","React","JSON storage","LM Studio"],'
            '"tech_stack_reasoning":["Быстро собрать API","Достаточно для MVP","Минимум инфраструктуры"]'
        ),
        str(AgentId.UX): (
            '"user_scenario":["Создать заявку","Ответить на вопросы","Получить план"],'
            '"user_screens":["Стартовая форма","Вопросы","Лента обсуждения","Итоговый план"]'
        ),
        str(AgentId.SECURITY): (
            '"processed_data":["Идея","Ответы пользователя","История встречи"],'
            '"data_sensitivity":["Средняя: может содержать бизнес-данные"],'
            '"security_measures":["Локальное хранение","Ограничить доступ","Не логировать секреты"]'
        ),
        str(AgentId.SKEPTIC): (
            '"risk_mitigations":["Сузить MVP","Проверить спрос","Назначить владельца"]'
        ),
    }
    return common.get(agent_id, common[str(AgentId.PRODUCT)])
