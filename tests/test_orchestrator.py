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


def test_placeholder_values_are_removed_from_payload():
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]), agents=[])
    turn = orchestrator.parse_agent_response(
        '{"summary":"...","arguments":["...","реальный аргумент"],"risks":["…"],"confidence":3}',
        "Product Manager",
        "analysis",
    )

    assert turn.payload.summary == ""
    assert turn.payload.arguments == ["реальный аргумент"]
    assert turn.payload.risks == []


def test_reasoning_inside_valid_json_is_removed():
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]), agents=[])
    turn = orchestrator.parse_agent_response(
        '{"summary":"Thinking Process: Analyze the Request. '
        'Для MVP нужен каталог шрифтов и покупка лицензии.",'
        '"arguments":["Return only JSON. Schema keys. Проверить оплату."],"confidence":3}',
        "Product Manager",
        "analysis",
    )

    assert "Thinking Process" not in turn.payload.summary
    assert "Analyze the Request" not in turn.payload.summary
    assert "Return only JSON" not in " ".join(turn.payload.arguments)


def test_empty_question_uses_fallback_question():
    client = FakeClient(['{"question":"...","summary":"..."}'])
    agent = make_agent()
    orchestrator = CouncilOrchestrator(llm_client=client, agents=[agent])
    state = MeetingState(idea="test")

    question = orchestrator.ask_clarifying_question(agent, state)

    assert question.status == "fallback"
    assert question.question
    assert "сценарий" in question.question


def test_invalid_json_question_keeps_useful_raw_text():
    client = FakeClient(["Кто будет покупать шрифт и как он поймёт условия лицензии?"])
    agent = make_agent()
    orchestrator = CouncilOrchestrator(llm_client=client, agents=[agent])
    state = MeetingState(idea="test")

    question = orchestrator.ask_clarifying_question(agent, state)

    assert question.status == "fallback"
    assert question.question == "Кто будет покупать шрифт и как он поймёт условия лицензии?"


def test_deepseek_question_uses_text_mode_for_clean_question():
    client = FakeClient(["Кто будет первым покупателем шрифта и какая лицензия ему нужна?"])
    agent = make_agent()
    settings = Settings(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="deepseek-r1-distill-qwen-7b-q4-k-m",
    )
    orchestrator = CouncilOrchestrator(llm_client=client, settings=settings, agents=[agent])
    state = MeetingState(idea="Сайт для продажи шрифтов")

    question = orchestrator.ask_clarifying_question(agent, state)

    assert question.status == "text"
    assert question.question == "Кто будет первым покупателем шрифта и какая лицензия ему нужна?"


def test_reasoning_question_uses_domain_fallback_instead_of_raw_reasoning():
    raw = "Thinking Process: Role: Product Manager. Analyze the Request: Constraints: MVP?"
    client = FakeClient([raw])
    agent = make_agent()
    settings = Settings(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="deepseek-r1-distill-qwen-7b-q4-k-m",
    )
    orchestrator = CouncilOrchestrator(llm_client=client, settings=settings, agents=[agent])
    state = MeetingState(idea="Сайт для продажи шрифтов")

    question = orchestrator.ask_clarifying_question(agent, state)

    assert question.status == "fallback"
    assert question.fallback_reason == "deterministic"
    assert "Thinking Process" not in question.question
    assert "шрифт" in question.question
    assert "Okay" not in question.question


def test_failed_turn_uses_fallback_payload():
    client = FakeClient(["not json", "still not json"])
    agent = make_agent()
    orchestrator = CouncilOrchestrator(llm_client=client, agents=[agent])
    state = MeetingState(idea="test")

    turn = orchestrator.ask_agent_turn(agent, "mvp_vote", state)

    assert turn.status == "fallback"
    assert turn.payload.summary
    assert turn.payload.mvp_features
    assert turn.payload.decision == "go_after_clarification"


def test_collect_questions_deduplicates_similar_questions_by_role():
    agents = [
        make_agent("Product Manager", "product_manager"),
        make_agent("Tech Lead", "tech_lead"),
    ]
    client = FakeClient(
        [
            "Какие функции доставки и оплаты критически важны для запуска MVP?",
            "Какие именно функции доставки и оплаты являются критически важными для запуска MVP?",
        ]
    )
    settings = Settings(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="deepseek-r1-distill-qwen-7b-q4-k-m",
    )
    orchestrator = CouncilOrchestrator(llm_client=client, settings=settings, agents=agents)
    state = MeetingState(idea="Сервис доставки ульев за 15 минут в полночь")

    questions = orchestrator.collect_questions(state)

    assert len(questions) == 2
    assert questions[0].question != questions[1].question
    assert questions[1].status == "fallback"
    assert "данные" in questions[1].question.lower() or "api" in questions[1].question.lower()


def test_fallback_questions_are_role_specific():
    agents = [
        make_agent("Product Manager", "product_manager"),
        make_agent("Security", "security"),
        make_agent("Skeptic", "skeptic"),
    ]
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]), agents=agents)
    state = MeetingState(idea="Сервис доставки ульев за 15 минут в полночь")

    questions = [orchestrator._fallback_question(agent, state, "", "failed").question for agent in agents]

    assert len(set(questions)) == len(questions)
    assert any("данные" in question.lower() or "платеж" in question.lower() for question in questions)
    assert any("допущ" in question.lower() for question in questions)


def test_fallback_turns_are_role_specific_in_same_phase():
    agents = [
        make_agent("Product Manager", "product_manager"),
        make_agent("Tech Lead", "tech_lead"),
        make_agent("UX Researcher", "ux_researcher"),
    ]
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]), agents=agents)
    state = MeetingState(idea="Сервис доставки ульев за 15 минут в полночь")

    turns = [orchestrator._fallback_turn(agent, "analysis", state, "", "failed") for agent in agents]

    assert len({turn.payload.summary for turn in turns}) == len(turns)
    assert len({tuple(turn.payload.arguments) for turn in turns}) == len(turns)
    assert len({tuple(turn.payload.risks) for turn in turns}) == len(turns)


def test_invalid_json_turn_keeps_useful_raw_text_as_summary():
    raw = "Для сайта шрифтов MVP должен включать каталог, страницу шрифта, покупку лицензии и простую оплату."
    client = FakeClient([raw, "still not json"])
    agent = make_agent()
    orchestrator = CouncilOrchestrator(llm_client=client, agents=[agent])
    state = MeetingState(idea="Сайт для продажи шрифтов")

    turn = orchestrator.ask_agent_turn(agent, "analysis", state)

    assert turn.status == "fallback"
    assert turn.payload.summary == raw


def test_reasoning_turn_is_not_used_as_summary():
    raw = "Thinking Process: Role: Tech Lead. Analyze the Request: Return only JSON. Schema keys."
    client = FakeClient([raw, "still not json"])
    agent = make_agent("Tech Lead", "tech_lead")
    orchestrator = CouncilOrchestrator(llm_client=client, agents=[agent])
    state = MeetingState(idea="Сайт для продажи шрифтов")

    turn = orchestrator.ask_agent_turn(agent, "analysis", state)

    assert turn.status == "fallback"
    assert turn.fallback_reason == "deterministic"
    assert "Thinking Process" not in turn.payload.summary
    assert "сайт по продаже шрифтов" in turn.payload.summary


def test_deepseek_reasoning_turn_uses_deterministic_summary():
    raw = "Okay, so I'm trying to figure out how to create an MVP for this typography license website."
    client = FakeClient([raw])
    agent = make_agent("Tech Lead", "tech_lead")
    settings = Settings(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="deepseek-r1-distill-qwen-7b-q4-k-m",
    )
    orchestrator = CouncilOrchestrator(llm_client=client, settings=settings, agents=[agent])
    state = MeetingState(idea="Сайт для продажи шрифтов")

    turn = orchestrator.ask_agent_turn(agent, "analysis", state)

    assert turn.status == "fallback"
    assert turn.fallback_reason == "deterministic"
    assert "Okay" not in turn.payload.summary
    assert "I'm" not in turn.payload.summary
    assert "сайт по продаже шрифтов" in turn.payload.summary


def test_valid_json_with_english_reasoning_falls_back_to_russian_payload():
    raw = '{"summary":"Okay, I need to define the MVP step by step.","confidence":3}'
    client = FakeClient([raw])
    agent = make_agent("Product Manager", "product_manager")
    settings = Settings(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="deepseek-r1-distill-qwen-7b-q4-k-m",
    )
    orchestrator = CouncilOrchestrator(llm_client=client, settings=settings, agents=[agent])
    state = MeetingState(idea="Сайт для продажи шрифтов")

    turn = orchestrator.ask_agent_turn(agent, "analysis", state)

    assert turn.status == "fallback"
    assert "Okay" not in turn.payload.summary
    assert "сайт продажи шрифтов" not in turn.payload.summary


def test_font_fallback_templates_are_clean_russian():
    agents = [
        make_agent("Product Manager", "product_manager"),
        make_agent("Tech Lead", "tech_lead"),
        make_agent("UX Researcher", "ux_researcher"),
        make_agent("Security", "security"),
        make_agent("Skeptic", "skeptic"),
    ]
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]), agents=agents)
    state = MeetingState(idea="Сайт для продажи шрифтов")

    texts = []
    for agent in agents:
        texts.append(orchestrator._fallback_question(agent, state, "", "failed").question)
        turn = orchestrator._fallback_turn(agent, "analysis", state, "", "failed")
        texts.append(turn.payload.summary)
        texts.extend(turn.payload.open_questions)

    joined = "\n".join(texts)
    banned = ["Okay", "I'm", "Let me", "step by step", "сайт продажи шрифтов", "для выбор шрифта"]
    assert not any(item in joined for item in banned)
    assert "сайт по продаже шрифтов" in joined
    assert "сценарий выбора шрифта" in joined


def test_repair_invalid_json_object_once():
    client = FakeClient(
        [
            '{"summary": 123}',
            '{"summary":"Ответ исправлен","confidence":3,"decision":"unknown"}',
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
    assert turn.payload.summary == "Ответ исправлен"
    assert len(client.messages) == 2


def test_failed_response_is_not_replaced_with_content():
    client = FakeClient(["not json", "still not json"])
    agent = make_agent()
    orchestrator = CouncilOrchestrator(llm_client=client, agents=[agent])
    state = MeetingState(idea="test")

    turn = orchestrator.ask_agent_turn(agent, "analysis", state)

    assert turn.status == "fallback"
    assert turn.payload.summary
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


def test_aggregate_roadmap_removes_duplicate_week_prefixes():
    state = MeetingState(idea="test")
    state.transcript.add(
        MeetingTurn(
            agent="a",
            role="product_manager",
            phase="mvp_vote",
            status="fallback",
            payload={
                "summary": "go",
                "roadmap_items": ["Неделя 1: уточнить MVP", "Неделя 1: повтор", "Неделя 2: прототип"],
            },
        )
    )

    result = CouncilOrchestrator(llm_client=FakeClient([]), agents=[]).aggregate_votes(state)

    assert result["roadmap_items"] == ["Неделя 1: уточнить MVP", "Неделя 2: прототип"]


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
    pm_content = "\n".join(message["content"] for message in pm_messages)
    tech_content = "\n".join(message["content"] for message in tech_messages)

    assert "PM_SECRET" in pm_content
    assert "TECH_SECRET" not in pm_content
    assert "TECH_SECRET" in tech_content
    assert "PM_SECRET" not in tech_content


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


def test_markdown_outputs_do_not_include_reasoning_or_placeholders():
    state = MeetingState(idea="Сайт для продажи шрифтов")
    state.transcript.add(
        MeetingTurn(
            agent="Tech Lead",
            role="tech_lead",
            phase="analysis",
            status="fallback",
            fallback_reason="deterministic",
            payload={
                "summary": "Нужно выбрать простую реализацию для сайта продажи шрифтов.",
                "risks": ["Размытый scope"],
                "insights": ["Проверить покупку лицензии."],
            },
        )
    )
    orchestrator = CouncilOrchestrator(llm_client=FakeClient([]), agents=[])

    markdown = orchestrator.build_final_plan_markdown(state)

    banned = ["Thinking Process", "Analyze the Request", "Return only JSON", "Schema", "..."]
    assert not any(item in markdown for item in banned)


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
