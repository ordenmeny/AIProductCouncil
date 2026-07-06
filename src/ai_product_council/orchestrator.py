from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from ai_product_council.agents import get_default_agents
from ai_product_council.config import Settings, load_settings
from ai_product_council.json_utils import extract_json_object
from ai_product_council.llm_client import LLMClientError, LMStudioClient
from ai_product_council.models import AgentResponse, AgentRole, MeetingState, PhaseName


PHASE_LABELS: dict[PhaseName, str] = {
    "questions": "Уточняющие вопросы",
    "analysis": "Индивидуальный анализ",
    "debate": "Обсуждение и спор",
    "mvp_proposal": "Предложения по MVP",
    "mvp_vote": "Голосование и приоритизация",
}

PHASE_INSTRUCTIONS: dict[PhaseName, str] = {
    "questions": "Задай 1-2 самых важных уточняющих вопроса пользователю со своей профессиональной позиции.",
    "analysis": "Проанализируй идею со своей роли: ценность, слабые места, ограничения и что важно проверить.",
    "debate": "Отреагируй на аргументы других агентов. Согласись или поспорь, но предложи конструктивное уточнение.",
    "mvp_proposal": "Предложи, что должно войти в первую версию продукта, и что нужно отложить.",
    "mvp_vote": (
        "Проголосуй за решение: go, go_after_clarification, no_go или pivot_or_narrow_mvp. "
        "Выбери MVP-функции, главные риски и следующий шаг."
    ),
}

OUTPUT_SCHEMA = """
Верни строго один JSON-объект без markdown:
{
  "agent": "название агента",
  "phase": "questions | analysis | debate | mvp_proposal | mvp_vote",
  "summary": "краткий вывод",
  "questions": ["вопрос 1"],
  "arguments": ["аргумент 1"],
  "risks": ["риск 1"],
  "mvp_features": ["функция 1"],
  "decision": "go | go_after_clarification | no_go | pivot_or_narrow_mvp | unknown",
  "next_step": "главный следующий шаг",
  "confidence": 1
}
"""


class CouncilOrchestrator:
    def __init__(
        self,
        llm_client: LMStudioClient | None = None,
        settings: Settings | None = None,
        agents: list[AgentRole] | None = None,
    ):
        self.settings = settings or load_settings()
        self.llm_client = llm_client or LMStudioClient(self.settings)
        self.agents = agents or get_default_agents()

    def run_meeting(self, idea: str) -> MeetingState:
        state = MeetingState(idea=idea)
        for phase in PHASE_LABELS:
            for agent in self.agents:
                response = self.ask_agent(agent=agent, phase=phase, state=state)
                state.add_response(response)
        state.final_report = self.build_final_report(state)
        return state

    def ask_agent(self, agent: AgentRole, phase: PhaseName, state: MeetingState) -> AgentResponse:
        messages = self.build_messages(agent=agent, phase=phase, state=state)
        try:
            raw = self.llm_client.chat(messages)
        except LLMClientError as exc:
            return self._fallback_response(agent.name, phase, str(exc))
        parsed = self.parse_agent_response(raw, agent.name, phase)
        if parsed.is_fallback:
            if "{" not in raw or "}" not in raw:
                return self._fallback_response(agent.name, phase, parsed.error or "JSON object not found", raw)
            retry_messages = messages + [
                {
                    "role": "assistant",
                    "content": raw,
                },
                {
                    "role": "user",
                    "content": (
                        "Предыдущий ответ не удалось распарсить как нужный JSON. "
                        "Исправь формат и верни только валидный JSON по схеме."
                    ),
                },
            ]
            try:
                retry_raw = self.llm_client.chat(retry_messages)
            except LLMClientError as exc:
                return self._fallback_response(agent.name, phase, str(exc), raw)
            retry_parsed = self.parse_agent_response(retry_raw, agent.name, phase)
            if retry_parsed.is_fallback:
                return self._fallback_response(agent.name, phase, retry_parsed.error or "Invalid JSON", retry_raw)
            return retry_parsed
        return parsed

    def build_messages(self, agent: AgentRole, phase: PhaseName, state: MeetingState) -> list[dict[str, str]]:
        private_context = self._read_private_context(agent.private_context_path)
        meeting_context = self._summarize_state_for_prompt(state)
        system = (
            f"{agent.system_prompt}\n\n"
            "Ты участвуешь в рабочем созвоне по проектированию B2B SaaS. Пиши кратко. "
            "Не показывай ход рассуждений. Верни только JSON.\n\n"
            f"Твой приватный контекст, недоступный другим агентам:\n{private_context}\n\n"
            f"{OUTPUT_SCHEMA}"
        )
        user = (
            "/no_think\n"
            f"Идея пользователя:\n{state.idea}\n\n"
            f"Фаза: {phase} — {PHASE_LABELS[phase]}.\n"
            f"Задача фазы: {PHASE_INSTRUCTIONS[phase]}\n\n"
            f"Контекст предыдущих фаз:\n{meeting_context}\n\n"
            "Заполни поля JSON применимо к этой фазе. Для нерелевантных списков используй пустые массивы, "
            "для decision используй unknown, если это не фаза голосования. Отвечай кратко."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def parse_agent_response(self, raw: str, agent_name: str, phase: PhaseName) -> AgentResponse:
        try:
            data = extract_json_object(raw)
            data["agent"] = agent_name
            data["phase"] = phase
            data["raw_text"] = raw
            return AgentResponse.model_validate(data)
        except (ValueError, ValidationError, TypeError) as exc:
            return AgentResponse(
                agent=agent_name,
                phase=phase,
                summary="Не удалось получить валидный JSON-ответ от модели.",
                raw_text=raw,
                is_fallback=True,
                error=str(exc),
            )

    def _fallback_response(
        self,
        agent_name: str,
        phase: PhaseName,
        error: str,
        raw_text: str = "",
    ) -> AgentResponse:
        templates = {
            "questions": {
                "summary": "Fallback: нужны ключевые уточнения перед оценкой.",
                "questions": [
                    "Кто основной покупатель и кто ежедневный пользователь продукта?",
                    "Какую ручную или дорогую операцию продукт должен заменить в первом MVP?",
                ],
            },
            "analysis": {
                "summary": "Fallback: идею стоит проверять через узкий ICP и один основной workflow.",
                "arguments": [
                    "MVP должен доказывать ценность на одном повторяемом сценарии.",
                    "Сложные интеграции лучше заменить импортом/экспортом на первой версии.",
                ],
                "risks": ["Слишком широкий scope", "Неясный бюджет покупателя"],
            },
            "debate": {
                "summary": "Fallback: проект можно запускать только с ограничением scope.",
                "arguments": [
                    "Сначала нужна проверка боли и готовности платить.",
                    "Архитектура должна оставлять место для ролей доступа и аудита.",
                ],
                "risks": ["Перегрузка MVP функциями"],
            },
            "mvp_proposal": {
                "summary": "Fallback: первая версия должна закрывать базовый end-to-end сценарий.",
                "mvp_features": [
                    "Ввод и структурирование исходных данных клиента",
                    "Генерация черновика проектного/MVP-плана",
                    "Экспорт итогового отчёта",
                ],
                "risks": ["Низкое качество входных данных"],
            },
            "mvp_vote": {
                "summary": "Fallback: запускать после сужения MVP и проверки ICP.",
                "mvp_features": [
                    "Ввод идеи продукта",
                    "Многоагентный анализ",
                    "Финальный отчёт",
                ],
                "risks": [
                    "Слишком широкий ICP",
                    "Медленная локальная модель",
                    "Невалидные JSON-ответы LLM",
                ],
                "decision": "go_after_clarification",
                "next_step": "Провести 5-7 интервью с потенциальными B2B-клиентами.",
            },
        }
        data = templates[phase]
        return AgentResponse(
            agent=agent_name,
            phase=phase,
            summary=data.get("summary", ""),
            questions=data.get("questions", []),
            arguments=data.get("arguments", []),
            risks=data.get("risks", []),
            mvp_features=data.get("mvp_features", []),
            decision=data.get("decision", "unknown"),
            next_step=data.get("next_step", ""),
            raw_text=raw_text,
            is_fallback=True,
            error=error,
        )

    def aggregate_votes(self, state: MeetingState) -> dict:
        vote_responses = state.votes
        decisions = Counter(response.decision for response in vote_responses)
        features = Counter(
            feature
            for response in vote_responses
            for feature in response.mvp_features
            if feature.strip()
        )
        risks = Counter(
            risk
            for response in vote_responses
            for risk in response.risks
            if risk.strip()
        )
        next_steps = [response.next_step for response in vote_responses if response.next_step.strip()]
        return {
            "decision_counts": dict(decisions),
            "final_decision": decisions.most_common(1)[0][0] if decisions else "unknown",
            "top_features": [item for item, _ in features.most_common(3)],
            "top_risks": [item for item, _ in risks.most_common(3)],
            "next_step": next_steps[0] if next_steps else "Провести интервью с потенциальными клиентами.",
        }

    def build_final_report(self, state: MeetingState) -> str:
        aggregated = self.aggregate_votes(state)
        analyses = state.phases.get("analysis", [])
        proposals = state.phases.get("mvp_proposal", [])

        analysis_lines = "\n".join(
            f"- **{response.agent}:** {response.summary}" for response in analyses if response.summary
        )
        proposal_lines = "\n".join(
            f"- **{response.agent}:** {', '.join(response.mvp_features) or response.summary}"
            for response in proposals
        )
        features = "\n".join(f"- {feature}" for feature in aggregated["top_features"]) or "- Не определено"
        risks = "\n".join(f"- {risk}" for risk in aggregated["top_risks"]) or "- Не определено"

        return f"""# Итоговый проект-план

## Краткое описание продукта

{state.idea}

## Оценка агентов

{analysis_lines or "- Пока нет валидных оценок агентов."}

## MVP

Топ-3 функции MVP по голосованию:

{features}

Предложения агентов:

{proposal_lines or "- Пока нет валидных предложений."}

## Техническая архитектура

- Streamlit frontend для ввода идеи и просмотра хода встречи.
- Python-оркестратор управляет фазами и состоянием.
- LM Studio OpenAI-compatible API генерирует ответы агентов.
- Приватные markdown-контексты разделяют информацию между ролями.
- Результаты сохраняются в JSON и Markdown.

## Риски

{risks}

## План разработки на 4-6 недель

- Неделя 1: уточнить ICP, сценарии, ограничения и критерии успешного MVP.
- Неделя 2: собрать прототип ключевого workflow и базовую архитектуру.
- Неделя 3: реализовать основные MVP-функции и простую авторизацию/роли.
- Неделя 4: провести пилот с 2-3 потенциальными клиентами.
- Неделя 5-6: доработать UX, безопасность, отчётность и подготовить демо продаж.

## Итоговое решение команды

**{aggregated["final_decision"]}**

Главный следующий шаг: {aggregated["next_step"]}
"""

    def _summarize_state_for_prompt(self, state: MeetingState) -> str:
        if not state.phases:
            return "Предыдущих ответов пока нет."
        lines: list[str] = []
        for phase, responses in state.phases.items():
            lines.append(f"### {PHASE_LABELS[phase]}")
            for response in responses:
                bits = [response.summary]
                if response.risks:
                    bits.append("Риски: " + "; ".join(response.risks[:2]))
                if response.mvp_features:
                    bits.append("MVP: " + "; ".join(response.mvp_features[:3]))
                lines.append(f"- {response.agent}: {' | '.join(bit for bit in bits if bit)}")
        return "\n".join(lines)

    @staticmethod
    def _read_private_context(path: Path) -> str:
        if not path.exists():
            return "Приватный контекст не найден."
        return path.read_text(encoding="utf-8").strip()
