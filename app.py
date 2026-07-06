from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from ai_product_council.agents import get_default_agents
from ai_product_council.config import load_settings
from ai_product_council.llm_client import LLMClientError
from ai_product_council.orchestrator import CouncilOrchestrator, PHASE_LABELS


OUTPUT_DIR = Path("outputs")


class FastDemoClient:
    def chat(self, messages):
        raise LLMClientError("Быстрый демо-режим: ответ сгенерирован fallback-логикой без ожидания LLM.")


def save_outputs(state) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"meeting_{stamp}.json"
    md_path = OUTPUT_DIR / f"meeting_{stamp}.md"
    json_path.write_text(
        json.dumps(state.to_export_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(state.final_report, encoding="utf-8")
    return json_path, md_path


st.set_page_config(page_title="AI Product Council", layout="wide")

st.title("AI Product Council")
st.caption("MVP мультиагентного продуктового совета для проектирования B2B SaaS")

settings = load_settings()

with st.sidebar:
    st.header("LM Studio")
    st.text_input("Base URL", value=settings.base_url, disabled=True)
    st.text_input("Model", value=settings.model, disabled=True)
    fast_demo = st.checkbox(
        "Быстрый демо-режим",
        value=False,
        help="Не ждёт локальную модель, а сразу использует fallback-ответы ролей. Обычный режим обращается к LM Studio.",
    )
    st.caption("Настройки берутся из `.env` или значений по умолчанию.")

    st.header("Агенты")
    for agent in get_default_agents():
        st.markdown(f"**{agent.name}**")
        st.caption(agent.description)

default_idea = (
    "B2B SaaS для автоматизации подготовки коммерческих предложений: "
    "менеджеры загружают вводные клиента, система собирает КП по шаблонам, "
    "учитывает прайсинг и помогает согласовать документ внутри компании."
)

idea = st.text_area(
    "Опишите идею B2B SaaS",
    value=default_idea,
    height=180,
)

col_run, _ = st.columns([1, 3])
run_clicked = col_run.button("Запустить совет", type="primary", use_container_width=True)

if run_clicked:
    if not idea.strip():
        st.error("Введите описание идеи.")
    else:
        llm_client = FastDemoClient() if fast_demo else None
        orchestrator = CouncilOrchestrator(settings=settings, llm_client=llm_client)
        progress = st.progress(0)
        status = st.empty()
        total_steps = len(orchestrator.agents) * len(PHASE_LABELS)
        completed = 0

        state = None
        try:
            from ai_product_council.models import MeetingState

            state = MeetingState(idea=idea.strip())
            for phase in PHASE_LABELS:
                status.info(f"Фаза: {PHASE_LABELS[phase]}")
                for agent in orchestrator.agents:
                    response = orchestrator.ask_agent(agent=agent, phase=phase, state=state)
                    state.add_response(response)
                    completed += 1
                    progress.progress(completed / total_steps)
            state.final_report = orchestrator.build_final_report(state)
            st.session_state["meeting_state"] = state
            status.success("Совет завершён.")
        except Exception as exc:
            status.error(f"Ошибка запуска встречи: {exc}")

state = st.session_state.get("meeting_state")

if state:
    st.divider()
    st.header("Ход встречи")

    for phase, label in PHASE_LABELS.items():
        responses = state.phases.get(phase, [])
        with st.expander(label, expanded=phase in {"mvp_vote"}):
            for response in responses:
                st.subheader(response.agent)
                if response.is_fallback:
                    st.warning(response.summary)
                    st.code(response.raw_text or response.error or "", language="text")
                else:
                    st.write(response.summary)
                    st.json(response.model_dump(mode="json"))

    st.header("Голосование")
    aggregated = CouncilOrchestrator(settings=settings).aggregate_votes(state)
    st.write(f"Итоговое решение: **{aggregated['final_decision']}**")
    st.write("Топ-3 функции MVP:")
    st.write(aggregated["top_features"] or ["Не определено"])
    st.write("Топ-3 риска:")
    st.write(aggregated["top_risks"] or ["Не определено"])

    st.header("Финальный отчёт")
    st.markdown(state.final_report)

    if st.button("Сохранить результат"):
        json_path, md_path = save_outputs(state)
        st.success(f"Сохранено: {json_path} и {md_path}")
