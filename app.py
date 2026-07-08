from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from ai_product_council.agents import get_default_agents
from ai_product_council.config import load_settings
from ai_product_council.llm_client import LLMClientError
from ai_product_council.models import MeetingState, ProjectMode
from ai_product_council.orchestrator import CouncilOrchestrator, DISCUSSION_PHASES, PHASE_LABELS


OUTPUT_DIR = Path("outputs")


class DemoFailedClient:
    def chat(self, messages):
        raise LLMClientError("Демо без LLM: запрос намеренно помечен как failed.")


def save_outputs(state: MeetingState) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {
        "json": OUTPUT_DIR / f"meeting_{stamp}.json",
        "transcript": OUTPUT_DIR / f"meeting_transcript_{stamp}.md",
        "final_plan": OUTPUT_DIR / f"final_plan_{stamp}.md",
    }
    paths["json"].write_text(
        json.dumps(state.to_export_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["transcript"].write_text(state.transcript_markdown, encoding="utf-8")
    paths["final_plan"].write_text(state.final_plan_markdown, encoding="utf-8")
    return paths


def make_orchestrator(use_demo_without_llm: bool) -> CouncilOrchestrator:
    client = DemoFailedClient() if use_demo_without_llm else None
    return CouncilOrchestrator(settings=load_settings(), llm_client=client)


def ensure_model_ready(orchestrator: CouncilOrchestrator, use_demo_without_llm: bool) -> bool:
    if use_demo_without_llm:
        return True
    try:
        models = orchestrator.llm_client.list_models()
    except LLMClientError as exc:
        st.error(f"LM Studio недоступен: {exc}")
        return False
    if orchestrator.settings.model not in models:
        st.error(
            f"Модель `{orchestrator.settings.model}` не загружена в LM Studio. "
            f"Сейчас доступны: {', '.join(models) or 'нет моделей'}."
        )
        return False
    return True


def warn_if_fallback_dominates(state: MeetingState, context: str) -> None:
    orchestrator = make_orchestrator(demo_without_llm)
    stats = orchestrator.response_stats(state)
    useful = stats["llm"] + stats["repaired"]
    fallback = stats["fallback"]
    if fallback and useful == 0:
        st.warning(
            f"{context}: все ответы получены через fallback. "
            "Это аварийный режим: LM Studio доступен, но модель не вернула валидный JSON. "
            "Для нормальной демонстрации проверьте модель, identifier и попробуйте Qwen/Gemma вместо reasoning-модели."
        )
    elif fallback > useful:
        st.warning(
            f"{context}: fallback сработал чаще, чем реальные ответы LLM. "
            "Результат можно показать как отказоустойчивость, но содержательно лучше перезапустить с более стабильной моделью."
        )


st.set_page_config(page_title="AI Product Council", layout="wide")

st.title("AI Product Council")
st.caption("Рабочий созвон ИИ-агентов для MVP нового сервиса или фичи")

settings = load_settings()

with st.sidebar:
    st.header("LM Studio")
    st.text_input("Base URL", value=settings.base_url, disabled=True)
    st.text_input("Model", value=settings.model, disabled=True)
    st.text_input("Timeout, sec", value=str(settings.timeout_seconds), disabled=True)
    st.text_input("Max tokens", value=str(settings.max_tokens), disabled=True)
    demo_without_llm = st.checkbox(
        "Демо без LLM",
        value=False,
        help="Только для аварийной демонстрации UI. Настоящий созвон требует выключенного режима и запущенного LM Studio.",
    )

    st.header("Агенты")
    for agent in get_default_agents():
        st.markdown(f"**{agent.name}**")
        st.caption(agent.description)

project_mode_label = st.radio(
    "Тип задачи",
    ["Новый сервис или внутренний инструмент", "Новая фича в существующем продукте"],
    horizontal=True,
)
project_mode: ProjectMode = (
    "new_service"
    if project_mode_label == "Новый сервис или внутренний инструмент"
    else "feature_in_existing_product"
)

default_idea = (
    "Внутренний сервис для подготовки заявок на закупку оборудования: "
    "сотрудник описывает потребность, система помогает собрать требования, "
    "проверяет обязательные поля и формирует черновик заявки для согласования."
)

idea = st.text_area("Идея продукта или фичи", value=default_idea, height=150)
constraints = st.text_area(
    "Ограничения",
    value="MVP должен быть реализуем за 4-6 недель небольшой командой. Сложные интеграции лучше отложить.",
    height=90,
)
desired_result = st.text_area(
    "Какой результат нужен от совета",
    value="Определить MVP, roadmap на 4-6 недель, риски, вопросы к заказчику и следующий практический шаг.",
    height=80,
)

col_questions, col_reset = st.columns([1, 4])

if col_questions.button("1. Получить вопросы агентов", type="primary"):
    if not idea.strip():
        st.error("Введите идею продукта или фичи.")
    else:
        orchestrator = make_orchestrator(demo_without_llm)
        if not ensure_model_ready(orchestrator, demo_without_llm):
            st.stop()
        state = orchestrator.create_state(
            idea=idea,
            project_mode=project_mode,
            constraints=constraints,
            desired_result=desired_result,
        )
        with st.spinner("Агенты формируют уточняющие вопросы..."):
            orchestrator.collect_questions(state)
        st.session_state["meeting_state"] = state
        warn_if_fallback_dominates(state, "Уточняющие вопросы")
        st.success("Вопросы собраны. Теперь ответьте на них ниже.")

if col_reset.button("Сбросить"):
    st.session_state.pop("meeting_state", None)
    st.rerun()

state: MeetingState | None = st.session_state.get("meeting_state")

if state:
    st.divider()
    st.header("Уточняющие вопросы")
    for question in state.questions:
        status = question.status
        if question.question:
            st.markdown(f"**{question.agent}** · `{status}`")
            st.write(question.question)
        else:
            st.markdown(f"**{question.agent}** · `failed`")
            st.warning(question.error or "Не удалось получить вопрос.")

    user_answer = st.text_area(
        "Ответы пользователя на вопросы агентов",
        value=state.user_answer.text,
        height=180,
        placeholder="Можно ответить одним текстом: что известно, что неизвестно, какие ограничения важны.",
    )

    if st.button("2. Провести созвон"):
        orchestrator = make_orchestrator(demo_without_llm)
        if not ensure_model_ready(orchestrator, demo_without_llm):
            st.stop()
        orchestrator.set_user_answer(state, user_answer)
        progress = st.progress(0)
        total_steps = len(orchestrator.agents) * len(DISCUSSION_PHASES)
        completed = 0
        with st.spinner("Агенты проводят созвон..."):
            state.transcript.turns.clear()
            for phase in DISCUSSION_PHASES:
                for agent in orchestrator.agents:
                    state.transcript.add(orchestrator.ask_agent_turn(agent, phase, state))
                    completed += 1
                    progress.progress(completed / total_steps)
            state.transcript_markdown = orchestrator.build_transcript_markdown(state)
            state.final_plan_markdown = orchestrator.build_final_plan_markdown(state)
        st.session_state["meeting_state"] = state
        warn_if_fallback_dominates(state, "Созвон")
        st.success("Созвон завершён. Ниже протокол и итоговый план.")

if state and state.transcript.turns:
    st.divider()
    orchestrator = make_orchestrator(demo_without_llm)
    stats = orchestrator.response_stats(state)

    col_llm, col_repaired, col_failed, col_fallback = st.columns(4)
    col_llm.metric("LLM", stats["llm"])
    col_repaired.metric("Repaired", stats["repaired"])
    col_failed.metric("Failed", stats["failed"])
    col_fallback.metric("Fallback", stats["fallback"])
    warn_if_fallback_dominates(state, "Текущий результат")

    st.header("Ход созвона")
    for phase in DISCUSSION_PHASES:
        with st.expander(PHASE_LABELS[phase], expanded=phase == "mvp_vote"):
            for turn in [item for item in state.transcript.turns if item.phase == phase]:
                st.subheader(f"{turn.agent} · {turn.status}")
                if turn.status == "failed":
                    st.warning(turn.error or "Ответ агента не получен.")
                    continue
                st.write(turn.payload.summary)
                st.json(turn.payload.model_dump(mode="json"))

    tab_transcript, tab_plan = st.tabs(["Протокол созвона", "Итоговый план"])
    with tab_transcript:
        st.markdown(state.transcript_markdown)
        st.download_button(
            "Скачать meeting_transcript.md",
            data=state.transcript_markdown,
            file_name="meeting_transcript.md",
            mime="text/markdown",
        )
    with tab_plan:
        st.markdown(state.final_plan_markdown)
        st.download_button(
            "Скачать final_plan.md",
            data=state.final_plan_markdown,
            file_name="final_plan.md",
            mime="text/markdown",
        )

    if st.button("Сохранить Markdown и JSON в outputs"):
        saved = save_outputs(state)
        st.success(
            "Сохранено: "
            + ", ".join(str(path) for path in saved.values())
        )
