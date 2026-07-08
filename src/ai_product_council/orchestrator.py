from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from ai_product_council.agents import get_default_agents
from ai_product_council.config import Settings, load_settings
from ai_product_council.json_utils import clean_llm_text, extract_json_object, extract_question_from_text
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
        state.questions = [self.ask_clarifying_question(agent, state) for agent in self.agents]
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
        if not payload.question.strip():
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
        if self._payload_is_empty(payload):
            return self._fallback_turn(agent, phase, state, raw, "LLM returned empty or placeholder content")
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
        system = (
            "Return one valid JSON object only. No markdown. No reasoning. "
            "Do not use quotation marks inside JSON string values. "
            "Never write ellipsis, placeholder text, TBD, or empty-looking content. "
            "If a list has no useful items, return an empty array. "
            "Use short Russian values. "
            f"{self._turn_schema(phase)}"
        )
        previous = self._short_transcript_summary(state)
        user = (
            f"Role: {agent.name}. "
            f"Role hint: {private_context[:260]}. "
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
        questions = {
            "product_manager": f"Какой один сценарий в первой версии должен быть доведён до результата: {domain['workflow']}?",
            "business_value": f"По какому признаку будет понятно, что {domain['product']} действительно полезен заказчику?",
            "tech_lead": f"Какие данные и внешние сервисы нужны для MVP, чтобы реализовать {domain['workflow']} за 4-6 недель?",
            "ux_researcher": f"Кто первый пользователь и какое действие в сценарии {domain['workflow']} должно стать проще?",
            "security": f"Какие пользовательские данные, платежи или файлы будут обрабатываться в {domain['product']}?",
            "skeptic": f"Какой самый рискованный допуск может сорвать запуск {domain['product']}?",
        }
        return ClarifyingQuestion(
            agent=agent.name,
            role=agent.slug,
            question=raw_question
            or questions.get(agent.slug, "Какое главное ограничение нужно учесть перед выбором MVP?"),
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
        domain = self._domain_terms(state.idea, state)
        role_focus = {
            "product_manager": f"сфокусировать MVP на сценарии {domain['workflow']}",
            "business_value": f"проверить, что {domain['product']} даёт измеримую пользу",
            "tech_lead": f"выбрать простую реализацию для {domain['product']}: {domain['workflow']} без лишних интеграций",
            "ux_researcher": f"проверить понятность выбора и получения результата в {domain['product']}",
            "security": f"минимизировать данные, платежные риски и доступы в {domain['product']}",
            "skeptic": f"сузить scope {domain['product']} до проверяемого запуска",
        }.get(agent.slug, "сфокусировать первую версию")
        payloads = {
            "analysis": AgentPayload(
                summary=raw_summary or f"Нужно {role_focus}.",
                arguments=[f"Для задачи {domain['product']} первая версия должна быть проверяемой и небольшой."],
                risks=["Слишком широкий scope может сорвать MVP за 4-6 недель."],
                open_questions=[f"Какой результат пользователь должен получить после сценария {domain['workflow']}?"],
                insights=[f"Главная ценность MVP — довести сценарий {domain['workflow']} до понятного результата, а не собрать полный продукт."],
            ),
            "debate": AgentPayload(
                summary=raw_summary or f"Поддерживаю необходимость сузить решение: нужно {role_focus}.",
                arguments=["Команде важно сначала доказать полезность одного процесса, а не строить полную систему."],
                risks=["Без ясного критерия успеха обсуждение уйдёт в список желаемых функций."],
                insights=["Спор агентов полезен, если приводит к ограничению первой версии."],
            ),
            "mvp_proposal": AgentPayload(
                summary=raw_summary or f"Первая версия должна закрывать сценарий {domain['workflow']}.",
                mvp_features=domain["mvp_features"],
                out_of_scope=domain["out_of_scope"],
                risks=domain["risks"],
                roadmap_items=[
                    f"Неделя 1: уточнить сценарий {domain['workflow']} и критерии успеха",
                    "Неделя 2-3: собрать прототип и основную логику",
                    "Неделя 4: провести пилот и собрать обратную связь",
                ],
                open_questions=domain["open_questions"],
            ),
            "mvp_vote": AgentPayload(
                summary=raw_summary or "Запускать после уточнения scope и критериев успеха.",
                decision="go_after_clarification",
                mvp_features=domain["mvp_features"][:3],
                risks=domain["risks"],
                roadmap_items=[
                    "Неделя 1: зафиксировать MVP",
                    "Неделя 2-3: реализовать первую версию",
                    "Неделя 4-6: пилот и доработка",
                ],
                open_questions=domain["open_questions"],
                insights=[f"Лучший следующий шаг — не расширять идею, а проверить сценарий {domain['workflow']}."],
                next_step="Провести короткое уточнение требований и зафиксировать MVP на один сценарий.",
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
        return [
            {
                "role": "system",
                "content": (
                    "Ты участник рабочего IT-созвона. Верни только один короткий вопрос по-русски. "
                    "Без JSON, markdown, списков, reasoning и объяснений."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Роль: {agent.name}. Идея: {state.idea[:400]}. "
                    f"Ограничения: {(state.constraints or 'не указаны')[:160]}. "
                    "Задай один вопрос заказчику, который поможет определить MVP."
                ),
            },
        ]

    def _prefers_text_fallback(self) -> bool:
        model = self.settings.model.lower()
        return "qwen" in model or "deepseek" in model or "r1" in model

    @staticmethod
    def _domain_terms(idea: str, state: MeetingState | None) -> dict:
        text = " ".join(
            part for part in [idea, state.constraints if state else "", state.desired_result if state else ""] if part
        ).lower()
        if any(word in text for word in ["шрифт", "font", "лиценз", "eula"]):
            return {
                "product": "сайт продажи шрифтов",
                "workflow": "выбор шрифта, проверка начертания, покупка лицензии и получение файла",
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
            "workflow": "один основной пользовательский сценарий",
            "mvp_features": ["Ввод исходных данных", "Обработка основного сценария", "Экспорт или сохранение результата"],
            "out_of_scope": ["Сложные интеграции", "Расширенная аналитика", "Несколько разных сценариев сразу"],
            "risks": ["Размытый scope", "Неясный владелец процесса"],
            "open_questions": ["Кто принимает итоговое решение по результату пилота?"],
        }

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

    def _build_question_messages(self, agent: AgentRole, state: MeetingState) -> list[dict[str, str]]:
        system = (
            "Return only JSON. No markdown. No reasoning. "
            "Do not use quotation marks inside JSON string values. "
            "Never write ellipsis, placeholder text, TBD, or empty-looking content. "
            'Schema keys: question string, summary string.'
        )
        user = (
            f"Role: {agent.name}. "
            f"Project: {self._project_mode_label(state.project_mode)}. "
            f"Idea: {state.idea[:500]}. "
            f"Constraints: {(state.constraints or 'none')[:220]}. "
            "Ask one important Russian question for deciding MVP scope, implementation risks, or adoption."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _agent_system_prompt(self, agent: AgentRole, private_context: str, schema: str) -> str:
        return (
            f"{agent.system_prompt}\n"
            "Ты участник рабочего созвона IT-команды. Отвечай кратко, конкретно, по-русски. "
            "Не показывай ход рассуждений. Не повторяй других агентов. "
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
