from backend.app.agents.roles import AGENTS
from backend.app.core.config import Settings
from backend.app.models import AgentPhase, MeetingPhase, UserAnswer
from backend.app.orchestrator import MeetingOrchestrator


class FakeLLM:
    async def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        if f'"{AgentPhase.CLARIFYING_QUESTION}"' in system_prompt:
            phase = AgentPhase.CLARIFYING_QUESTION
            return (
                '{"agent":"stub","agent_id":"product_manager","phase":"clarifying_question",'
                '"summary":"Уточнение","open_questions":["Какой пользовательский сценарий главный?"],'
                '"reason":"Это сузит MVP."}'
            )
        if f'"{AgentPhase.VOTE}"' in system_prompt:
            return (
                '{"agent":"stub","agent_id":"product_manager","phase":"vote","summary":"Голос",'
                '"decision":"go_after_clarification","mvp_priority":["Один сценарий"],'
                '"roadmap_items":["Неделя 1: scope"],"risks":["Размытый scope"],'
                '"open_questions":["Кто владелец?"],"insights":["MVP надо сузить"],'
                '"main_risk":"Размытый scope","next_step":"Зафиксировать MVP","reason":"Реализуемо"}'
            )
        phase = "individual_analysis"
        if f'"{AgentPhase.DEBATE}"' in system_prompt:
            phase = "debate"
        if f'"{AgentPhase.MVP_PROPOSAL}"' in system_prompt:
            phase = "mvp_proposal"
        return (
            '{"agent":"stub","agent_id":"product_manager","phase":"'
            + phase
            + '","summary":"Содержательный ответ","mvp_priority":["Фича 1"],"risks":["Риск 1"]}'
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
