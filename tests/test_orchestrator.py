from pathlib import Path

from ai_product_council.models import AgentRole, AgentResponse, MeetingState
from ai_product_council.orchestrator import CouncilOrchestrator


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        return self.responses.pop(0)


def make_agent(name, slug, context_path):
    return AgentRole(
        name=name,
        slug=slug,
        description="test",
        system_prompt="system",
        private_context_path=context_path,
    )


def test_parse_valid_agent_response():
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]), agents=[])
    response = orchestrator.parse_agent_response(
        '{"summary":"ok","confidence":4,"decision":"go","arguments":["a"]}',
        "Product Manager",
        "analysis",
    )

    assert response.agent == "Product Manager"
    assert response.phase == "analysis"
    assert response.summary == "ok"
    assert response.arguments == ["a"]


def test_parse_invalid_agent_response_returns_fallback():
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]), agents=[])
    response = orchestrator.parse_agent_response("not json", "Tech Lead", "analysis")

    assert response.is_fallback is True
    assert response.agent == "Tech Lead"
    assert response.error


def test_aggregate_votes_top_items():
    state = MeetingState(idea="test")
    state.add_response(
        AgentResponse(
            agent="a",
            phase="mvp_vote",
            decision="go",
            mvp_features=["auth", "dashboard"],
            risks=["security"],
            next_step="interviews",
        )
    )
    state.add_response(
        AgentResponse(
            agent="b",
            phase="mvp_vote",
            decision="go_after_clarification",
            mvp_features=["auth", "csv"],
            risks=["security", "pricing"],
        )
    )

    result = CouncilOrchestrator(llm_client=FakeClient([]), agents=[]).aggregate_votes(state)

    assert result["top_features"][0] == "auth"
    assert result["top_risks"][0] == "security"
    assert result["next_step"] == "interviews"


def test_private_context_is_not_mixed_between_agents(tmp_path: Path):
    pm_context = tmp_path / "pm.md"
    tech_context = tmp_path / "tech.md"
    pm_context.write_text("PM_SECRET", encoding="utf-8")
    tech_context.write_text("TECH_SECRET", encoding="utf-8")

    pm = make_agent("PM", "pm", pm_context)
    tech = make_agent("Tech", "tech", tech_context)
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]), agents=[pm, tech])
    state = MeetingState(idea="test")

    pm_messages = orchestrator.build_messages(pm, "questions", state)
    tech_messages = orchestrator.build_messages(tech, "questions", state)

    assert "PM_SECRET" in pm_messages[0]["content"]
    assert "TECH_SECRET" not in pm_messages[0]["content"]
    assert "TECH_SECRET" in tech_messages[0]["content"]
    assert "PM_SECRET" not in tech_messages[0]["content"]


def test_ask_agent_skips_retry_when_response_has_no_json_object():
    client = FakeClient(["not json"])
    agent = make_agent("PM", "pm", Path("missing.md"))
    orchestrator = CouncilOrchestrator(llm_client=client, agents=[agent])

    response = orchestrator.ask_agent(agent, "questions", MeetingState(idea="test"))

    assert response.is_fallback is True
    assert len(client.messages) == 1


def test_ask_agent_retries_invalid_json_object_once():
    client = FakeClient(
        [
            '{"summary": 123}',
            '{"summary":"fixed","confidence":3,"decision":"unknown"}',
        ]
    )
    agent = make_agent("PM", "pm", Path("missing.md"))
    orchestrator = CouncilOrchestrator(llm_client=client, agents=[agent])

    response = orchestrator.ask_agent(agent, "questions", MeetingState(idea="test"))

    assert response.is_fallback is False
    assert response.summary == "fixed"
    assert len(client.messages) == 2
