from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import ValidationError

from ai_product_council.agents import get_default_agents
from ai_product_council.config import Settings, load_settings
from ai_product_council.json_utils import (
    clean_llm_text,
    extract_json_object,
    extract_question_from_text,
    is_russian_user_facing_text,
)
from ai_product_council.llm_client import LLMClientError, LMStudioClient
from ai_product_council.models import (
    AgentPayload,
    AgentRole,
    ClarifyingQuestion,
    MeetingState,
    MeetingTurn,
    PhaseName,
    ProjectMode,
    ResponseStatus,
    UserAnswer,
)


PHASE_LABELS: dict[PhaseName, str] = {
    "clarifying_questions": "Уточняющие вопросы",
    "analysis": "Индивидуальный анализ",
    "debate": "Обсуждение и спор",
    "mvp_proposal": "MVP / scope первой версии",
    "mvp_vote": "Голосование и решение",
}

DISCUSSION_PHASES: list[PhaseName] = ["analysis", "debate", "mvp_proposal", "mvp_vote"]

QUESTION_SCHEMA = """
Return only JSON:
{
  "question": "concrete Russian question",
  "summary": "why the answer changes the MVP"
}
"""

TURN_SCHEMA = """
Return only JSON:
{
  "summary": "short concrete Russian position",
  "arguments": ["specific argument"],
  "risks": ["specific risk"],
  "mvp_features": ["specific MVP feature"],
  "out_of_scope": ["specific item not for v1"],
  "open_questions": ["specific question for the customer"],
  "insights": ["specific non-obvious useful thought"],
  "roadmap_items": ["specific roadmap step"],
  "decision": "go | go_after_clarification | no_go | pivot_or_narrow_mvp | unknown",
  "next_step": "one practical next step",
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

    def create_state(
        self,
        idea: str,
        project_mode: ProjectMode,
        constraints: str = "",
        desired_result: str = "",
    ) -> MeetingState:
        return MeetingState(
            idea=idea.strip(),
            project_mode=project_mode,
            constraints=constraints.strip(),
            desired_result=desired_result.strip(),
        )

    def collect_questions(self, state: MeetingState) -> list[ClarifyingQuestion]:
        state.questions = []
        for agent in self.agents:
            question = self.ask_clarifying_question(agent, state)
            if self._is_duplicate_question(question.question, state.questions):
                question = self._fallback_question(
                    agent,
                    state,
                    "",
                    "Question was too similar to an earlier agent question",
                )
            state.questions.append(question)
        return state.questions

    def set_user_answer(self, state: MeetingState, answer_text: str) -> None:
        state.user_answer = UserAnswer(text=answer_text.strip())

    def run_discussion(self, state: MeetingState) -> MeetingState:
        for phase in DISCUSSION_PHASES:
            for agent in self.agents:
                state.transcript.add(self.ask_agent_turn(agent=agent, phase=phase, state=state))
        state.transcript_markdown = self.build_transcript_markdown(state)
        state.final_plan_markdown = self.build_final_plan_markdown(state)
        return state

    def run_full_meeting(
        self,
        idea: str,
        project_mode: ProjectMode = "new_service",
        constraints: str = "",
        desired_result: str = "",
        user_answer_text: str = "",
    ) -> MeetingState:
        state = self.create_state(idea, project_mode, constraints, desired_result)
        self.collect_questions(state)
        self.set_user_answer(state, user_answer_text)
        return self.run_discussion(state)

    def ask_clarifying_question(self, agent: AgentRole, state: MeetingState) -> ClarifyingQuestion:
        if self._prefers_text_fallback():
            return self._ask_text_question(agent, state)

        messages = self._build_question_messages(agent, state)
        raw, status, error = self._chat_with_repair(
            messages,
            QUESTION_SCHEMA,
            max_tokens=self.settings.question_max_tokens,
            allow_repair=self.settings.enable_repair,
        )
        if status == "failed":
            return self._fallback_question(agent, state, raw, error)
        payload = self._parse_payload(raw, agent.name, "clarifying_questions")
        if not is_russian_user_facing_text(payload.question):
            return self._fallback_question(agent, state, raw, "LLM returned an empty question")
        return ClarifyingQuestion(
            agent=agent.name,
            role=agent.slug,
            question=payload.question,
            status=status,
            error=error,
            raw_text=raw,
        )

    def ask_agent_turn(self, agent: AgentRole, phase: PhaseName, state: MeetingState) -> MeetingTurn:
        messages = self.build_messages(agent=agent, phase=phase, state=state)
        raw, status, error = self._chat_with_repair(
            messages,
            self._turn_schema(phase),
            max_tokens=self.settings.turn_max_tokens,
            allow_repair=self._allow_repair(),
        )
        if status == "failed":
            return self._fallback_turn(agent, phase, state, raw, error)
        payload = self._parse_payload(raw, agent.name, phase)
        if self._payload_is_empty(payload) or not self._payload_is_russian(payload):
            return self._fallback_turn(agent, phase, state, raw, "LLM returned empty, placeholder, or non-Russian content")
        return MeetingTurn(
            agent=agent.name,
            role=agent.slug,
            phase=phase,
            status=status,
            payload=payload,
            raw_text=raw,
            error=error,
        )

    def build_messages(self, agent: AgentRole, phase: PhaseName, state: MeetingState) -> list[dict[str, str]]:
        private_context = self._read_private_context(agent.private_context_path)
        system = self._agent_system_prompt(
            agent,
            private_context,
            (
                "Return one valid JSON object only. No markdown. No reasoning. "
                "Russian only. No English words or explanations. "
                "If you want to reason, do not output reasoning. "
                "Do not use quotation marks inside JSON string values. "
                "Never write ellipsis, placeholder text, TBD, or empty-looking content. "
                "If a list has no useful items, return an empty array. "
                "Use short Russian values. "
                f"{self._turn_schema(phase)}"
            ),
        )
        previous = self._short_transcript_summary(state)
        focus = self._role_focus(agent.slug)
        user = (
            f"Role: {agent.name}. "
            f"Role focus: {focus}. "
            f"Project: {self._project_mode_label(state.project_mode)}. "
            f"Idea: {state.idea[:500]}. "
            f"Constraints: {(state.constraints or 'none')[:180]}. "
            f"User answers: {(state.user_answer.text or 'none')[:500]}. "
            f"Phase: {phase}. "
            f"Previous: {previous[:500]}. "
            f"Task: {self._phase_instruction(phase)}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def parse_agent_response(self, raw: str, agent_name: str, phase: PhaseName) -> MeetingTurn:
        return MeetingTurn(
            agent=agent_name,
            role="unknown",
            phase=phase,
            status="llm",
            payload=self._parse_payload(raw, agent_name, phase),
            raw_text=raw,
        )

    def aggregate_votes(self, state: MeetingState) -> dict:
        valid_votes = [turn for turn in state.votes if turn.status != "failed"]
        decisions = Counter(turn.payload.decision for turn in valid_votes)
        features = Counter(
            feature
            for turn in valid_votes
            for feature in turn.payload.mvp_features
            if feature.strip()
        )
        risks = Counter(
            risk
            for turn in state.transcript.turns
            for risk in turn.payload.risks
            if turn.status != "failed" and risk.strip()
        )
        next_steps = [
            turn.payload.next_step
            for turn in valid_votes
            if turn.payload.next_step.strip()
        ]
        open_questions = self._collect_unique(
            question
            for turn in state.transcript.turns
            if turn.status != "failed"
            for question in turn.payload.open_questions
        )
        insights = self._collect_unique(
            insight
            for turn in state.transcript.turns
            if turn.status != "failed"
            for insight in turn.payload.insights
        )
        roadmap_items = self._collect_unique(
            item
            for turn in state.transcript.turns
            if turn.status != "failed"
            for item in turn.payload.roadmap_items
        )
        return {
            "decision_counts": dict(decisions),
            "final_decision": decisions.most_common(1)[0][0] if decisions else "unknown",
            "top_features": [item for item, _ in features.most_common(3)],
            "top_risks": [item for item, _ in risks.most_common(3)],
            "open_questions": open_questions[:5],
            "insights": insights[:5],
            "roadmap_items": self._normalize_roadmap(roadmap_items),
            "next_step": next_steps[0] if next_steps else "Уточнить главный сценарий, критерии успеха и scope MVP.",
        }

    def response_stats(self, state: MeetingState) -> dict[str, int]:
        statuses = Counter(question.status for question in state.questions)
        statuses.update(turn.status for turn in state.transcript.turns)
        return {
            "llm": statuses.get("llm", 0),
            "repaired": statuses.get("repaired", 0),
            "text": statuses.get("text", 0),
            "failed": statuses.get("failed", 0),
            "fallback": statuses.get("fallback", 0),
        }

    def build_transcript_markdown(self, state: MeetingState) -> str:
        stats = self.response_stats(state)
        lines = [
            "# Протокол AI Product Council",
            "",
            f"**Режим проекта:** {self._project_mode_label(state.project_mode)}",
            "",
            "## Исходная идея",
            "",
            state.idea,
            "",
            "## Ограничения и желаемый результат",
            "",
            f"- Ограничения: {state.constraints or 'Не указаны'}",
            f"- Желаемый результат: {state.desired_result or 'Не указан'}",
            "",
            "## Уточняющие вопросы",
            "",
        ]
        for question in state.questions:
            value = question.question or f"[failed] {question.error}"
            lines.append(f"- **{question.agent}** ({question.status}): {value}")

        lines.extend(["", "## Ответы пользователя", "", state.user_answer.text or "Не указаны.", ""])

        for phase in DISCUSSION_PHASES:
            lines.extend(["", f"## {PHASE_LABELS[phase]}", ""])
            for turn in [item for item in state.transcript.turns if item.phase == phase]:
                lines.append(f"### {turn.agent} ({turn.status})")
                if turn.status == "failed":
                    lines.extend(["", f"> Failed: {turn.error}", ""])
                    continue
                lines.extend(
                    [
                        "",
                        turn.payload.summary or "Нет краткого вывода.",
                        "",
                    ]
                )
                self._append_list(lines, "Аргументы", turn.payload.arguments)
                self._append_list(lines, "Риски", turn.payload.risks)
                self._append_list(lines, "MVP / scope", turn.payload.mvp_features)
                self._append_list(lines, "Не входит в v1", turn.payload.out_of_scope)
                self._append_list(lines, "Вопросы к заказчику", turn.payload.open_questions)
                self._append_list(lines, "Инсайты", turn.payload.insights)
                self._append_list(lines, "Roadmap", turn.payload.roadmap_items)
                if phase == "mvp_vote":
                    lines.append(f"- Решение: `{turn.payload.decision}`")
                    lines.append(f"- Следующий шаг: {turn.payload.next_step or 'Не указан'}")
                lines.append("")

        lines.extend(
            [
                "## Техническая диагностика",
                "",
                f"- LLM: {stats['llm']}",
                f"- Repaired: {stats['repaired']}",
                f"- Text: {stats['text']}",
                f"- Failed: {stats['failed']}",
                f"- Fallback: {stats['fallback']}",
                "",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    def build_final_plan_markdown(self, state: MeetingState) -> str:
        aggregated = self.aggregate_votes(state)
        analyses = [turn for turn in state.transcript.turns if turn.phase == "analysis" and turn.status != "failed"]
        proposals = [turn for turn in state.transcript.turns if turn.phase == "mvp_proposal" and turn.status != "failed"]
        votes = [turn for turn in state.votes if turn.status != "failed"]
        features = aggregated["top_features"] or self._collect_unique(
            feature for turn in proposals for feature in turn.payload.mvp_features
        )[:3]
        risks = aggregated["top_risks"] or self._collect_unique(
            risk for turn in analyses for risk in turn.payload.risks
        )[:3]
        roadmap = aggregated["roadmap_items"] or self._collect_unique(
            item for turn in proposals + votes for item in turn.payload.roadmap_items
        )[:6]
        open_questions = aggregated["open_questions"] or self._collect_unique(
            question for turn in analyses + proposals + votes for question in turn.payload.open_questions
        )[:5]
        insights = aggregated["insights"] or self._collect_unique(
            insight for turn in analyses + proposals + votes for insight in turn.payload.insights
        )[:5]
        out_of_scope = self._collect_unique(
            item for turn in proposals for item in turn.payload.out_of_scope
        )[:5]

        lines = [
            "# Итоговый план IT-созвона",
            "",
            "## Суть задачи",
            "",
            state.idea,
            "",
            "## Ценность и рабочий сценарий",
            "",
        ]
        if analyses:
            for turn in analyses:
                lines.append(f"- **{turn.agent}:** {turn.payload.summary}")
        else:
            lines.append("- Недостаточно валидных ответов агентов для уверенного вывода.")

        lines.extend(["", "## MVP / Scope первой версии", ""])
        self._append_plain_list(lines, features, "Не определено")

        lines.extend(["", "## Что не входит в первую версию", ""])
        self._append_plain_list(lines, out_of_scope, "Не определено")

        lines.extend(["", "## Roadmap на 4-6 недель", ""])
        if roadmap:
            self._append_plain_list(lines, roadmap, "Не определено")
        else:
            lines.extend(
                [
                    "- Неделя 1: уточнить основной сценарий, пользователя и критерии успеха.",
                    "- Неделя 2: собрать прототип без сложных интеграций.",
                    "- Неделя 3: реализовать ключевые функции MVP и базовое хранение данных.",
                    "- Неделя 4: провести пилот на 2-3 реальных сценариях.",
                    "- Неделя 5-6: доработать UX, безопасность, отчётность и принять решение о следующей версии.",
                ]
            )

        lines.extend(["", "## Технический подход", ""])
        tech_turns = [turn for turn in analyses + proposals if turn.role == "tech_lead"]
        if tech_turns:
            for turn in tech_turns:
                for argument in turn.payload.arguments[:3]:
                    lines.append(f"- {argument}")
                for feature in turn.payload.mvp_features[:3]:
                    lines.append(f"- {feature}")
        else:
            lines.extend(
                [
                    "- Streamlit UI для сценария защиты и демонстрации.",
                    "- Python-оркестратор для фаз созвона и хранения состояния.",
                    "- LM Studio OpenAI-compatible API для локальной LLM.",
                    "- Markdown/JSON экспорт результатов.",
                ]
            )

        lines.extend(["", "## UX / Workflow", ""])
        ux_turns = [turn for turn in analyses + proposals if turn.role == "ux_researcher"]
        if ux_turns:
            for turn in ux_turns:
                lines.append(f"- {turn.payload.summary}")
                for feature in turn.payload.mvp_features[:2]:
                    lines.append(f"- {feature}")
        else:
            lines.append("- Проверить основной пользовательский workflow на 2-3 реальных сценариях.")

        lines.extend(["", "## Внедрение и проверка ценности", ""])
        business_turns = [turn for turn in analyses + proposals if turn.role == "business_value"]
        if business_turns:
            for turn in business_turns:
                lines.append(f"- {turn.payload.summary}")
        else:
            lines.append("- Найти 2-3 будущих пользователя или внутренних заказчика и проверить пользу на пилоте.")

        lines.extend(["", "## Данные и безопасность", ""])
        security_turns = [turn for turn in analyses + proposals if turn.role == "security"]
        if security_turns:
            for turn in security_turns:
                lines.append(f"- {turn.payload.summary}")
                for risk in turn.payload.risks[:2]:
                    lines.append(f"- Риск: {risk}")
        else:
            lines.append("- Зафиксировать требования к данным, доступам и хранению до пилота.")

        lines.extend(["", "## Риски", ""])
        self._append_plain_list(lines, risks, "Не определено")

        lines.extend(["", "## Вопросы к заказчику", ""])
        self._append_plain_list(lines, open_questions, "Не определено")

        lines.extend(["", "## Инсайты", ""])
        self._append_plain_list(lines, insights, "Не определено")

        lines.extend(["", "## Итоговое решение команды", "", f"**{aggregated['final_decision']}**", ""])
        if votes:
            for turn in votes:
                lines.append(f"- **{turn.agent}:** `{turn.payload.decision}` — {turn.payload.summary}")
        lines.extend(["", f"**Следующий шаг:** {aggregated['next_step']}", ""])
        return "\n".join(lines).strip() + "\n"

    def _chat_with_repair(
        self,
        messages: list[dict[str, str]],
        schema: str,
        max_tokens: int | None = None,
        allow_repair: bool = True,
    ) -> tuple[str, ResponseStatus, str | None]:
        try:
            raw = self.llm_client.chat(messages, max_tokens=max_tokens)
        except LLMClientError as exc:
            return "", "failed", str(exc)
        try:
            AgentPayload.model_validate(extract_json_object(raw))
            return raw, "llm", None
        except Exception as exc:
            if not allow_repair:
                return raw, "failed", str(exc)
            repair_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Repair your answer. {schema}"},
            ]
            try:
                repaired = self.llm_client.chat(repair_messages, max_tokens=max_tokens)
                AgentPayload.model_validate(extract_json_object(repaired))
                return repaired, "repaired", str(exc)
            except Exception as repair_exc:
                return raw, "failed", f"{exc}; repair failed: {repair_exc}"

    def _parse_payload(self, raw: str, agent_name: str, phase: PhaseName) -> AgentPayload:
        try:
            data = extract_json_object(raw)
            return AgentPayload.model_validate(data)
        except (ValueError, ValidationError, TypeError) as exc:
            raise ValueError(f"Invalid payload from {agent_name} in {phase}: {exc}") from exc

    def _ask_text_question(self, agent: AgentRole, state: MeetingState) -> ClarifyingQuestion:
        messages = self._build_text_question_messages(agent, state)
        try:
            raw = self.llm_client.chat(messages, max_tokens=self.settings.question_max_tokens)
        except LLMClientError as exc:
            return self._fallback_question(agent, state, "", str(exc))
        question = extract_question_from_text(raw)
        if question:
            return ClarifyingQuestion(
                agent=agent.name,
                role=agent.slug,
                question=question,
                status="text",
                fallback_reason="text",
                raw_text=raw,
            )
        return self._fallback_question(agent, state, raw, "LLM returned reasoning or unusable text")

    def _fallback_question(
        self,
        agent: AgentRole,
        state: MeetingState,
        raw: str,
        error: str | None,
    ) -> ClarifyingQuestion:
        raw_question = extract_question_from_text(raw)
        domain = self._domain_terms(state.idea, state)
        return ClarifyingQuestion(
            agent=agent.name,
            role=agent.slug,
            question=raw_question
            or self._role_question(agent.slug, domain),
            status="fallback",
            fallback_reason="text" if raw_question else "deterministic",
            error=error,
            raw_text=raw,
        )

    def _fallback_turn(
        self,
        agent: AgentRole,
        phase: PhaseName,
        state: MeetingState,
        raw: str,
        error: str | None,
    ) -> MeetingTurn:
        raw_summary = clean_llm_text(raw)
        if self._is_reasoning_model() and not is_russian_user_facing_text(raw_summary):
            raw_summary = ""
        domain = self._domain_terms(state.idea, state)
        role_payload = self._fallback_role_payload(agent.slug, domain)
        payloads = {
            "analysis": AgentPayload(
                summary=raw_summary or role_payload["analysis_summary"],
                arguments=role_payload["arguments"],
                risks=role_payload["risks"],
                open_questions=role_payload["open_questions"],
                insights=role_payload["insights"],
            ),
            "debate": AgentPayload(
                summary=raw_summary or role_payload["debate_summary"],
                arguments=role_payload["debate_arguments"],
                risks=role_payload["risks"][:1],
                insights=role_payload["insights"][:1],
            ),
            "mvp_proposal": AgentPayload(
                summary=raw_summary or role_payload["proposal_summary"],
                mvp_features=role_payload["mvp_features"],
                out_of_scope=role_payload["out_of_scope"],
                risks=role_payload["risks"],
                roadmap_items=role_payload["roadmap_items"],
                open_questions=role_payload["open_questions"],
            ),
            "mvp_vote": AgentPayload(
                summary=raw_summary or role_payload["vote_summary"],
                decision="go_after_clarification",
                mvp_features=role_payload["mvp_features"][:3],
                risks=role_payload["risks"],
                roadmap_items=role_payload["roadmap_items"],
                open_questions=role_payload["open_questions"],
                insights=role_payload["insights"],
                next_step=role_payload["next_step"],
            ),
        }
        fallback_reason = "text" if raw_summary else "deterministic"
        return MeetingTurn(
            agent=agent.name,
            role=agent.slug,
            phase=phase,
            status="fallback",
            fallback_reason=fallback_reason,
            payload=payloads[phase],
            raw_text=raw,
            error=error,
        )

    @staticmethod
    def _raw_text_to_question(raw: str) -> str:
        return extract_question_from_text(raw)

    @staticmethod
    def _raw_text_to_summary(raw: str) -> str:
        return clean_llm_text(raw)

    @staticmethod
    def _clean_raw_text(raw: str) -> str:
        return clean_llm_text(raw)

    def _build_text_question_messages(self, agent: AgentRole, state: MeetingState) -> list[dict[str, str]]:
        private_context = self._read_private_context(agent.private_context_path)
        focus = self._role_focus(agent.slug)
        return [
            {
                "role": "system",
                "content": (
                    "Ты участник рабочего IT-созвона. Верни только один короткий вопрос по-русски. "
                    "Без JSON, markdown, списков, reasoning и объяснений. "
                    "Ответ должен быть только на русском. Английские фразы запрещены. "
                    "Если хочется рассуждать, не выводи рассуждения. "
                    "Не повторяй вопросы других ролей. Спрашивай строго из своей профессиональной роли."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Роль: {agent.name}. Фокус роли: {focus}. "
                    f"Приватный контекст роли: {private_context[:260]}. "
                    f"Идея: {state.idea[:400]}. "
                    f"Ограничения: {(state.constraints or 'не указаны')[:160]}. "
                    f"Уже заданные вопросы: {self._format_questions(state)[:500]}. "
                    "Задай один новый вопрос заказчику, который поможет определить MVP."
                ),
            },
        ]

    def _prefers_text_fallback(self) -> bool:
        return self._is_reasoning_model()

    def _is_reasoning_model(self) -> bool:
        model = self.settings.model.lower()
        return "deepseek" in model or "r1" in model or "qwen3" in model

    @staticmethod
    def _domain_terms(idea: str, state: MeetingState | None) -> dict:
        text = " ".join(
            part for part in [idea, state.constraints if state else "", state.desired_result if state else ""] if part
        ).lower()
        if any(word in text for word in ["шрифт", "font", "лиценз", "eula"]):
            return {
                "product": "сайт по продаже шрифтов",
                "workflow": "сценарий выбора шрифта, проверки начертания, покупки лицензии и получения файла",
                "mvp_features": [
                    "Каталог шрифтов с фильтрами",
                    "Карточка шрифта с live preview",
                    "Корзина и покупка лицензии",
                    "Автоматическая выдача файла и EULA после оплаты",
                ],
                "out_of_scope": [
                    "Подписка на библиотеку шрифтов",
                    "Личный кабинет с расширенной историей",
                    "Маркетплейс сторонних студий",
                ],
                "risks": [
                    "Неясные условия лицензии могут тормозить покупку.",
                    "Платежи и выдача файлов требуют аккуратной проверки.",
                    "Слишком большой каталог усложнит запуск MVP.",
                ],
                "open_questions": [
                    "Какие типы лицензий нужны в первой версии?",
                    "Какие способы оплаты обязательны для первых покупателей?",
                    "Какие 10 шрифтов попадут в стартовый каталог?",
                ],
            }
        return {
            "product": idea.strip()[:120] or "сервис",
            "workflow": "ключевой сценарий заказа и получения результата",
            "mvp_features": ["Ввод исходных данных", "Обработка ключевого сценария", "Экспорт или сохранение результата"],
            "out_of_scope": ["Сложные интеграции", "Расширенная аналитика", "Несколько разных сценариев сразу"],
            "risks": ["Размытый scope", "Неясный владелец процесса"],
            "open_questions": ["Кто принимает итоговое решение по результату пилота?"],
        }

    @staticmethod
    def _role_focus(role: str) -> str:
        return {
            "product_manager": "пользователь, проблема, MVP, приоритеты и критерии успеха",
            "business_value": "измеримая польза, внедрение, спрос и первый пилот",
            "tech_lead": "архитектура, данные, интеграции, сроки и технические ограничения",
            "ux_researcher": "первый пользователь, workflow, трение, onboarding и доверие",
            "security": "данные, доступы, хранение, платежи, файлы и минимальные меры защиты",
            "skeptic": "ложные допущения, причины провала, scope creep и способы сузить риск",
        }.get(role, "MVP и практические ограничения")

    @staticmethod
    def _role_question(role: str, domain: dict) -> str:
        return {
            "product_manager": f"Какой один результат пользователь должен получить в первой версии через сценарий {domain['workflow']}?",
            "business_value": f"Какой измеримый признак покажет, что {domain['product']} действительно полезен заказчику?",
            "tech_lead": f"Какие данные, API или внешние сервисы обязательны, чтобы реализовать {domain['workflow']} за 4-6 недель?",
            "ux_researcher": f"Кто первый пользователь и где сейчас самое болезненное действие в сценарии {domain['workflow']}?",
            "security": f"Какие персональные данные, платежи, файлы или роли доступа появятся в первой версии {domain['product']}?",
            "skeptic": f"Какое самое опасное допущение о пользователях, сроках или спросе может сорвать запуск {domain['product']}?",
        }.get(role, "Какое главное ограничение нужно учесть перед выбором MVP?")

    @staticmethod
    def _is_duplicate_question(question: str, previous_questions: list[ClarifyingQuestion]) -> bool:
        normalized = CouncilOrchestrator._normalize_similarity_text(question)
        if not normalized:
            return False
        for previous in previous_questions:
            previous_normalized = CouncilOrchestrator._normalize_similarity_text(previous.question)
            if not previous_normalized:
                continue
            if normalized == previous_normalized:
                return True
            if SequenceMatcher(None, normalized, previous_normalized).ratio() >= 0.72:
                return True
        return False

    @staticmethod
    def _normalize_similarity_text(value: str) -> str:
        normalized = "".join(char.lower() if char.isalnum() else " " for char in value)
        stop_words = {
            "какие",
            "какой",
            "какая",
            "какое",
            "именно",
            "нужно",
            "нужны",
            "можно",
            "для",
            "чтобы",
            "mvp",
            "минимально",
            "жизнеспособного",
            "продукта",
            "первой",
            "версии",
        }
        words = [word for word in normalized.split() if word not in stop_words]
        return " ".join(words)

    @staticmethod
    def _fallback_role_payload(role: str, domain: dict) -> dict[str, list[str] | str]:
        product = domain["product"]
        workflow = domain["workflow"]
        base = {
            "analysis_summary": f"Нужно сузить {product} до сценария {workflow}.",
            "debate_summary": "Поддерживаю сужение scope до проверяемого результата.",
            "proposal_summary": f"Первая версия должна закрывать сценарий {workflow}.",
            "vote_summary": "Запускать после уточнения scope и критериев успеха.",
            "arguments": [f"Для задачи {product} первая версия должна быть проверяемой и небольшой."],
            "debate_arguments": ["Команде важно сначала доказать полезность одного процесса, а не строить полную систему."],
            "risks": domain["risks"],
            "mvp_features": domain["mvp_features"],
            "out_of_scope": domain["out_of_scope"],
            "open_questions": domain["open_questions"],
            "insights": [f"Главная ценность MVP — довести сценарий {workflow} до понятного результата."],
            "roadmap_items": [
                f"Неделя 1: уточнить сценарий {workflow} и критерии успеха",
                "Неделя 2-3: собрать прототип и основную логику",
                "Неделя 4-6: провести пилот и доработать решение",
            ],
            "next_step": "Провести короткое уточнение требований и зафиксировать MVP на один сценарий.",
        }
        role_specific = {
            "product_manager": {
                "analysis_summary": f"Нужно выбрать один пользовательский результат для {workflow} и не расширять MVP раньше пилота.",
                "debate_summary": "Продуктово спор полезен только если приводит к явному приоритету первого сценария.",
                "proposal_summary": "MVP должен включать путь от ввода потребности до понятного результата для пользователя.",
                "vote_summary": "Запускать после фиксации пользователя, результата и критерия успеха.",
                "arguments": ["Без одного приоритетного сценария команда будет обсуждать набор функций, а не ценность."],
                "risks": ["Размытый пользователь и размытый критерий успеха сделают MVP непроверяемым."],
                "open_questions": [f"Кто первый пользователь {product} и какой результат для него считается успехом?"],
                "insights": ["Первую версию стоит оценивать по завершённому сценарию, а не по количеству функций."],
                "next_step": "Зафиксировать первого пользователя, его задачу и критерий успешного пилота.",
            },
            "business_value": {
                "analysis_summary": f"Нужно доказать, что {product} даёт измеримую пользу до расширения функциональности.",
                "debate_summary": "Бизнес-решение стоит принимать через пилот с метрикой пользы, а не через список пожеланий.",
                "proposal_summary": "MVP должен включать минимальный пилот и способ измерить эффект внедрения.",
                "vote_summary": "Запускать после выбора метрики пользы и владельца пилота.",
                "arguments": ["Даже технически готовый сервис не взлетит без понятного владельца внедрения."],
                "risks": ["Пользователи могут не перейти на новый процесс, если польза не видна в первые дни."],
                "open_questions": [f"Какая метрика покажет, что {product} экономит время, деньги или снижает ошибки?"],
                "insights": ["Ранний пилот должен проверять готовность пользоваться сервисом, а не только качество интерфейса."],
                "next_step": "Выбрать владельца пилота и одну метрику ценности.",
            },
            "tech_lead": {
                "analysis_summary": f"Нужно проверить данные, API и внешние зависимости для сценария {workflow}.",
                "debate_summary": "Технический риск лучше снижать отказом от необязательных интеграций в первой версии.",
                "proposal_summary": "MVP должен использовать простую архитектуру и минимум внешних зависимостей.",
                "vote_summary": "Запускать после проверки обязательных интеграций и формата данных.",
                "arguments": ["Срок 4-6 недель реалистичен только при ясных данных и ограниченном числе интеграций."],
                "risks": ["Платежи, внешние API или неясная модель данных могут стать главным узким местом."],
                "mvp_features": ["Форма ввода данных", "Основная бизнес-логика", "Минимальное хранение результата"],
                "out_of_scope": ["Сложные интеграции", "Автоматизация всех исключений", "Масштабирование под высокую нагрузку"],
                "open_questions": [f"Какие API, данные и сервисы обязательны для первой версии {product}?"],
                "insights": ["Технический MVP должен сначала подтвердить поток данных, а затем расширять автоматизацию."],
                "next_step": "Составить список обязательных данных и интеграций для первой версии.",
            },
            "ux_researcher": {
                "analysis_summary": f"Нужно проверить, где пользователь теряет время или доверие в сценарии {workflow}.",
                "debate_summary": "UX-риск не в красоте интерфейса, а в непонятном первом действии и результате.",
                "proposal_summary": "MVP должен включать простой workflow, понятные состояния и объяснимый результат.",
                "vote_summary": "Запускать после проверки сценария на 2-3 будущих пользователях.",
                "arguments": ["Если пользователь не понимает первый шаг, полезность сервиса не будет проверена."],
                "risks": ["Слишком сложный onboarding может скрыть реальную ценность продукта."],
                "mvp_features": ["Понятный стартовый экран", "Пошаговый ввод данных", "Ясный экран результата"],
                "out_of_scope": ["Сложная персонализация", "Несколько альтернативных workflows", "Расширенная визуальная настройка"],
                "open_questions": [f"Какое действие в сценарии {workflow} сейчас самое непонятное для первого пользователя?"],
                "insights": ["Для защиты проекта важнее показать сквозной сценарий, чем широкий набор экранов."],
                "next_step": "Проверить прототип сценария на нескольких будущих пользователях.",
            },
            "security": {
                "analysis_summary": f"Нужно заранее ограничить данные, роли доступа и хранение в {product}.",
                "debate_summary": "Безопасность MVP должна быть минимальной, но явной: кто что видит и где это хранится.",
                "proposal_summary": "MVP должен включать базовые роли, минимизацию данных и понятное хранение результата.",
                "vote_summary": "Запускать после фиксации данных, доступов и платежных или файловых рисков.",
                "arguments": ["Даже прототип может обрабатывать чувствительные данные, если есть файлы, платежи или заявки."],
                "risks": ["Неясные права доступа и хранение файлов могут заблокировать пилот."],
                "mvp_features": ["Минимальные роли доступа", "Сохранение только нужных данных", "Журнал ключевых действий"],
                "out_of_scope": ["Сложная IAM-модель", "Полный аудит безопасности", "Хранение лишних персональных данных"],
                "open_questions": [f"Какие данные, файлы, платежи и роли доступа есть в первой версии {product}?"],
                "insights": ["Минимизация данных часто дешевле и быстрее сложных защитных механизмов."],
                "next_step": "Описать типы данных, роли доступа и срок хранения для пилота.",
            },
            "skeptic": {
                "analysis_summary": f"Главный риск {product} — строить слишком широкий сервис без проверки спроса.",
                "debate_summary": "Я бы не расширял scope, пока не доказано, что пользователям нужен именно этот сценарий.",
                "proposal_summary": "MVP должен специально проверить самое рискованное допущение, а не имитировать полный продукт.",
                "vote_summary": "Запускать только после сужения MVP и явного условия остановки пилота.",
                "arguments": ["Слабое место идеи может быть не в реализации, а в предположении, что пользователи изменят привычный процесс."],
                "risks": ["Команда может принять техническую готовность за доказанную ценность."],
                "mvp_features": ["Пилот на одном сценарии", "Проверка критического допущения", "Критерий остановки или продолжения"],
                "out_of_scope": ["Все дополнительные сценарии", "Функции без проверки спроса", "Доработки для гипотетических пользователей"],
                "open_questions": [f"Какое допущение о {product} окажется самым дорогим, если оно неверно?"],
                "insights": ["Хороший MVP должен уметь быстро показать, что идею нужно изменить или сузить."],
                "next_step": "Назвать одно критическое допущение и способ проверить его за неделю.",
            },
        }
        base.update(role_specific.get(role, {}))
        return base

    @staticmethod
    def _payload_is_empty(payload: AgentPayload) -> bool:
        return not any(
            [
                payload.summary.strip(),
                payload.question.strip(),
                payload.arguments,
                payload.risks,
                payload.mvp_features,
                payload.out_of_scope,
                payload.open_questions,
                payload.insights,
                payload.roadmap_items,
                payload.next_step.strip(),
            ]
        )

    @staticmethod
    def _payload_is_russian(payload: AgentPayload) -> bool:
        values = [
            payload.summary,
            payload.question,
            payload.next_step,
            *payload.arguments,
            *payload.risks,
            *payload.mvp_features,
            *payload.out_of_scope,
            *payload.open_questions,
            *payload.insights,
            *payload.roadmap_items,
        ]
        useful_values = [value for value in values if value.strip()]
        return bool(useful_values) and all(is_russian_user_facing_text(value) for value in useful_values)

    def _build_question_messages(self, agent: AgentRole, state: MeetingState) -> list[dict[str, str]]:
        private_context = self._read_private_context(agent.private_context_path)
        focus = self._role_focus(agent.slug)
        system = (
            "Return only JSON. No markdown. No reasoning. "
            "Russian only. No English words or explanations. "
            "If you want to reason, do not output reasoning. "
            "Do not use quotation marks inside JSON string values. "
            "Never write ellipsis, placeholder text, TBD, or empty-looking content. "
            'Schema keys: question string, summary string.'
        )
        user = (
            f"Role: {agent.name}. "
            f"Role focus: {focus}. "
            f"Role private context: {private_context[:260]}. "
            f"Project: {self._project_mode_label(state.project_mode)}. "
            f"Idea: {state.idea[:500]}. "
            f"Constraints: {(state.constraints or 'none')[:220]}. "
            f"Already asked: {self._format_questions(state)[:500]}. "
            "Ask one new Russian question for your role only."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _agent_system_prompt(self, agent: AgentRole, private_context: str, schema: str) -> str:
        return (
            f"{agent.system_prompt}\n"
            "Ты участник рабочего созвона IT-команды. Отвечай кратко, конкретно, по-русски. "
            "Не показывай ход рассуждений. Не повторяй других агентов. "
            "Ответ должен быть только на русском. Английские фразы и объяснение процесса ответа запрещены. "
            "Используй только свою роль и приватный контекст.\n\n"
            f"Приватный контекст роли:\n{private_context}\n\n"
            f"{schema}"
        )

    def _phase_instruction(self, phase: PhaseName) -> str:
        return {
            "analysis": "Дай позицию своей роли: ценность, ограничения, риски, вопросы к заказчику и неочевидный инсайт.",
            "debate": "Отреагируй на предыдущие реплики: согласись, поспорь или уточни риск, добавь полезный инсайт.",
            "mvp_proposal": "Предложи scope первой версии: что входит, что не входит, и шаги roadmap.",
            "mvp_vote": "Проголосуй и назови топ-функции, риски, вопросы к заказчику, инсайты и следующий шаг.",
            "clarifying_questions": "Задай один важный вопрос.",
        }[phase]

    def _allow_repair(self) -> bool:
        if not self.settings.enable_repair:
            return False
        is_real_lm_studio = isinstance(self.llm_client, LMStudioClient)
        is_gemma = "gemma" in self.settings.model.lower()
        return not (is_real_lm_studio and is_gemma)

    def _turn_schema(self, phase: PhaseName) -> str:
        schemas = {
            "analysis": (
                "Schema keys: summary string, arguments string array, risks string array, "
                "open_questions string array, insights string array, confidence integer."
            ),
            "debate": (
                "Schema keys: summary string, arguments string array, risks string array, "
                "insights string array, confidence integer."
            ),
            "mvp_proposal": (
                "Schema keys: summary string, mvp_features string array, out_of_scope string array, "
                "risks string array, roadmap_items string array, open_questions string array, confidence integer."
            ),
            "mvp_vote": (
                "Schema keys: summary string, decision one of go go_after_clarification no_go "
                "pivot_or_narrow_mvp unknown, mvp_features string array, risks string array, "
                "roadmap_items string array, open_questions string array, insights string array, "
                "next_step string, confidence integer."
            ),
            "clarifying_questions": QUESTION_SCHEMA,
        }
        return schemas[phase]

    def _format_questions(self, state: MeetingState) -> str:
        if not state.questions:
            return "Вопросы ещё не заданы."
        return "\n".join(
            f"- {question.agent}: {question.question or '[failed]'}"
            for question in state.questions
        )

    def _short_transcript_summary(self, state: MeetingState) -> str:
        valid_turns = [turn for turn in state.transcript.turns if turn.status != "failed"]
        if not valid_turns:
            return "Пока нет реплик."
        recent = valid_turns[-8:]
        return "\n".join(
            f"- {turn.agent} / {PHASE_LABELS[turn.phase]}: {turn.payload.summary}"
            for turn in recent
        )

    @staticmethod
    def _read_private_context(path: Path) -> str:
        if not path.exists():
            return "Приватный контекст не найден."
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _project_mode_label(mode: ProjectMode) -> str:
        return {
            "new_service": "Новый сервис или внутренний инструмент",
            "feature_in_existing_product": "Новая фича в существующем продукте",
        }[mode]

    @staticmethod
    def _append_list(lines: list[str], title: str, values: list[str]) -> None:
        if values:
            lines.append(f"**{title}:**")
            for value in values:
                lines.append(f"- {value}")
            lines.append("")

    @staticmethod
    def _append_plain_list(lines: list[str], values: list[str], empty: str) -> None:
        if values:
            for value in values:
                lines.append(f"- {value}")
        else:
            lines.append(f"- {empty}")

    @staticmethod
    def _collect_unique(values) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if normalized and normalized not in seen:
                result.append(normalized)
                seen.add(normalized)
        return result

    @staticmethod
    def _normalize_roadmap(values: list[str]) -> list[str]:
        if not values:
            return []
        result: list[str] = []
        seen_prefixes: set[str] = set()
        for value in values:
            normalized = value.strip()
            if not normalized:
                continue
            prefix = normalized.split(":", 1)[0].lower()
            if prefix in seen_prefixes:
                continue
            result.append(normalized)
            seen_prefixes.add(prefix)
            if len(result) >= 5:
                break
        return result
