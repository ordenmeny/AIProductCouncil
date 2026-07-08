from pathlib import Path

from ai_product_council.config import Settings
from ai_product_council.models import AgentRole, MeetingState, MeetingTurn
from ai_product_council.orchestrator import CouncilOrchestrator


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(messages)
        if isinstance(self.responses[0], Exception):
            raise self.responses.pop(0)
        return self.responses.pop(0)


def make_agent(name="Product Manager", slug="product_manager", context_path=Path("missing.md")):
    return AgentRole(
        name=name,
        slug=slug,
        description="test",
        system_prompt="system",
        private_context_path=context_path,
    )


def test_parse_valid_agent_response():
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]), agents=[])
    turn = orchestrator.parse_agent_response(
        '{"summary":"ok","confidence":4,"decision":"go","arguments":["a"],"insights":["i"]}',
        "Product Manager",
        "analysis",
    )

    assert turn.agent == "Product Manager"
    assert turn.phase == "analysis"
    assert turn.payload.summary == "ok"
    assert turn.payload.arguments == ["a"]
    assert turn.payload.insights == ["i"]


def test_repair_invalid_json_object_once():
    client = FakeClient(
        [
            '{"summary": 123}',
            '{"summary":"fixed","confidence":3,"decision":"unknown"}',
        ]
    )
    agent = make_agent()
    settings = Settings(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="test-model",
        enable_repair=True,
    )
    orchestrator = CouncilOrchestrator(llm_client=client, settings=settings, agents=[agent])
    state = MeetingState(idea="test")

    turn = orchestrator.ask_agent_turn(agent, "analysis", state)

    assert turn.status == "repaired"
    assert turn.payload.summary == "fixed"
    assert len(client.messages) == 2


def test_failed_response_is_not_replaced_with_content():
    client = FakeClient(["not json", "still not json"])
    agent = make_agent()
    orchestrator = CouncilOrchestrator(llm_client=client, agents=[agent])
    state = MeetingState(idea="test")

    turn = orchestrator.ask_agent_turn(agent, "analysis", state)

    assert turn.status == "failed"
    assert turn.payload.summary == ""
    assert turn.error


def test_aggregate_votes_top_items():
    state = MeetingState(idea="test")
    state.transcript.add(
        MeetingTurn(
            agent="a",
            role="product_manager",
            phase="mvp_vote",
            status="llm",
            payload={
                "summary": "go",
                "decision": "go",
                "mvp_features": ["auth", "dashboard"],
                "risks": ["security"],
                "roadmap_items": ["week 1"],
                "open_questions": ["who owns it?"],
                "insights": ["manual workaround exists"],
                "next_step": "interviews",
            },
        )
    )
    state.transcript.add(
        MeetingTurn(
            agent="b",
            role="tech_lead",
            phase="mvp_vote",
            status="llm",
            payload={
                "summary": "scope",
                "decision": "go_after_clarification",
                "mvp_features": ["auth", "csv"],
                "risks": ["security", "pricing"],
                "roadmap_items": ["week 2"],
                "open_questions": ["what is success?"],
            },
        )
    )

    result = CouncilOrchestrator(llm_client=FakeClient([]), agents=[]).aggregate_votes(state)

    assert result["top_features"][0] == "auth"
    assert result["top_risks"][0] == "security"
    assert result["roadmap_items"] == ["week 1", "week 2"]
    assert result["open_questions"] == ["who owns it?", "what is success?"]
    assert result["insights"] == ["manual workaround exists"]
    assert result["next_step"] == "interviews"


def test_private_context_is_not_mixed_between_agents(tmp_path: Path):
    pm_context = tmp_path / "pm.md"
    tech_context = tmp_path / "tech.md"
    pm_context.write_text("PM_SECRET", encoding="utf-8")
    tech_context.write_text("TECH_SECRET", encoding="utf-8")

    pm = make_agent("PM", "product_manager", pm_context)
    tech = make_agent("Tech", "tech_lead", tech_context)
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]), agents=[pm, tech])
    state = MeetingState(idea="test")

    pm_messages = orchestrator.build_messages(pm, "analysis", state)
    tech_messages = orchestrator.build_messages(tech, "analysis", state)

    assert "PM_SECRET" in pm_messages[1]["content"]
    assert "TECH_SECRET" not in pm_messages[1]["content"]
    assert "TECH_SECRET" in tech_messages[1]["content"]
    assert "PM_SECRET" not in tech_messages[1]["content"]


def test_markdown_outputs_include_transcript_and_final_plan():
    state = MeetingState(
        idea="Сервис для подготовки заявок",
        constraints="4 недели",
        desired_result="MVP plan",
    )
    state.transcript.add(
        MeetingTurn(
            agent="Product Manager",
            role="product_manager",
            phase="analysis",
            status="llm",
            payload={
                "summary": "Есть ценность",
                "risks": ["широкий scope"],
                "open_questions": ["Кто согласует заявку?"],
                "insights": ["Главная польза может быть в снижении ошибок."],
            },
        )
    )
    state.transcript.add(
        MeetingTurn(
            agent="Tech Lead",
            role="tech_lead",
            phase="mvp_vote",
            status="llm",
            payload={
                "summary": "Можно запускать",
                "decision": "go",
                "mvp_features": ["CSV import"],
                "roadmap_items": ["Неделя 1: уточнить форму заявки"],
                "next_step": "pilot",
            },
        )
    )
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]), agents=[])

    transcript = orchestrator.build_transcript_markdown(state)
    plan = orchestrator.build_final_plan_markdown(state)

    assert "# Протокол AI Product Council" in transcript
    assert "Сервис для подготовки заявок" in transcript
    assert "# Итоговый план IT-созвона" in plan
    assert "CSV import" in plan
    assert "## Roadmap на 4-6 недель" in plan
    assert "## Риски" in plan
    assert "## Вопросы к заказчику" in plan
    assert "## Инсайты" in plan


def test_default_prompts_do_not_force_b2b_saas():
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]))
    state = MeetingState(idea="Новый сервис для заявок")

    all_messages = [
        message["content"]
        for agent in orchestrator.agents
        for message in orchestrator.build_messages(agent, "analysis", state)
    ]

    assert not any("B2B SaaS" in message for message in all_messages)
    assert not any("enterprise" in message.lower() for message in all_messages)
