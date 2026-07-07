from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from ai_product_council.agents import get_default_agents
from ai_product_council.config import Settings, load_settings
from ai_product_council.json_utils import extract_json_object
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
  "question": "one important Russian question",
  "summary": "why this question matters"
}
"""

TURN_SCHEMA = """
Return only JSON:
{
  "summary": "short Russian position",
  "arguments": ["1-3 arguments"],
  "risks": ["0-3 risks"],
  "mvp_features": ["0-3 features"],
  "out_of_scope": ["0-3 items not for v1"],
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
        project_mode: ProjectMode = "new_saas",
        constraints: str = "",
        desired_result: str = "",
        user_answer_text: str = "",
    ) -> MeetingState:
        state = self.create_state(idea, project_mode, constraints, desired_result)
        self.collect_questions(state)
        self.set_user_answer(state, user_answer_text)
        return self.run_discussion(state)

    def ask_clarifying_question(self, agent: AgentRole, state: MeetingState) -> ClarifyingQuestion:
        messages = self._build_question_messages(agent, state)
        raw, status, error = self._chat_with_repair(
            messages,
            QUESTION_SCHEMA,
            max_tokens=self.settings.question_max_tokens,
            allow_repair=self.settings.enable_repair,
        )
        if status == "failed":
            return ClarifyingQuestion(
                agent=agent.name,
                role=agent.slug,
                question="",
                status=status,
                error=error,
                raw_text=raw,
            )
        payload = self._parse_payload(raw, agent.name, "clarifying_questions")
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
            return MeetingTurn(
                agent=agent.name,
                role=agent.slug,
                phase=phase,
                status=status,
                raw_text=raw,
                error=error,
            )
        return MeetingTurn(
            agent=agent.name,
            role=agent.slug,
            phase=phase,
            status=status,
            payload=self._parse_payload(raw, agent.name, phase),
            raw_text=raw,
            error=error,
        )

    def build_messages(self, agent: AgentRole, phase: PhaseName, state: MeetingState) -> list[dict[str, str]]:
        private_context = self._read_private_context(agent.private_context_path)
        system = (
            "Return only JSON. No markdown. No reasoning. "
            "Do not use quotation marks inside JSON string values. "
            "Do not copy dots or placeholders. Replace all placeholders with real content. "
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
        return {
            "decision_counts": dict(decisions),
            "final_decision": decisions.most_common(1)[0][0] if decisions else "unknown",
            "top_features": [item for item, _ in features.most_common(3)],
            "top_risks": [item for item, _ in risks.most_common(3)],
            "next_step": next_steps[0] if next_steps else "Провести проблемные интервью и уточнить scope MVP.",
        }

    def response_stats(self, state: MeetingState) -> dict[str, int]:
        statuses = Counter(question.status for question in state.questions)
        statuses.update(turn.status for turn in state.transcript.turns)
        return {
            "llm": statuses.get("llm", 0),
            "repaired": statuses.get("repaired", 0),
            "failed": statuses.get("failed", 0),
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
                f"- Failed: {stats['failed']}",
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
        out_of_scope = self._collect_unique(
            item for turn in proposals for item in turn.payload.out_of_scope
        )[:5]

        lines = [
            "# Итоговый MVP / Feature Plan",
            "",
            "## Краткое описание",
            "",
            state.idea,
            "",
            "## Целевой клиент и ценность",
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

        lines.extend(["", "## Архитектура MVP", ""])
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

        lines.extend(["", "## GTM / Rollout", ""])
        sales_turns = [turn for turn in analyses + proposals if turn.role == "sales_gtm"]
        if sales_turns:
            for turn in sales_turns:
                lines.append(f"- {turn.payload.summary}")
        else:
            lines.append("- Найти 2-3 дизайн-партнёра и проверить готовность к пилоту.")

        lines.extend(["", "## Security / Compliance", ""])
        security_turns = [turn for turn in analyses + proposals if turn.role == "security"]
        if security_turns:
            for turn in security_turns:
                lines.append(f"- {turn.payload.summary}")
                for risk in turn.payload.risks[:2]:
                    lines.append(f"- Риск: {risk}")
        else:
            lines.append("- Зафиксировать требования к данным, доступам и хранению до пилота.")

        lines.extend(["", "## Топ-3 риска", ""])
        self._append_plain_list(lines, risks, "Не определено")

        lines.extend(
            [
                "",
                "## План на 4-6 недель",
                "",
                "- Неделя 1: уточнить ICP, пользовательский workflow и критерии успеха пилота.",
                "- Неделя 2: собрать прототип основного сценария без сложных интеграций.",
                "- Неделя 3: реализовать MVP-функции, базовое хранение данных и экспорт результата.",
                "- Неделя 4: провести пилот с 2-3 пользователями или командами.",
                "- Неделя 5-6: доработать UX, безопасность, отчётность и подготовить решение go/no-go.",
                "",
                "## Итоговое решение команды",
                "",
                f"**{aggregated['final_decision']}**",
                "",
            ]
        )
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

    def _build_question_messages(self, agent: AgentRole, state: MeetingState) -> list[dict[str, str]]:
        system = (
            "Return only JSON. No markdown. No reasoning. "
            "Do not use quotation marks inside JSON string values. "
            "Do not copy dots or placeholders. Replace all placeholders with real content. "
            'Schema: {"question":"...","summary":"..."}'
        )
        user = (
            f"Role: {agent.name}. "
            f"Project: {self._project_mode_label(state.project_mode)}. "
            f"Idea: {state.idea[:500]}. "
            f"Constraints: {(state.constraints or 'none')[:220]}. "
            "Ask one important Russian question for deciding MVP scope."
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
            "analysis": "Дай позицию своей роли: ценность, ограничения, риски и что проверить.",
            "debate": "Отреагируй на предыдущие реплики: согласись, поспорь или уточни риск.",
            "mvp_proposal": "Предложи scope первой версии: что входит, что не входит.",
            "mvp_vote": "Проголосуй и назови топ-функции, риски и следующий шаг.",
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
                'Schema: {"summary":"...","arguments":["..."],'
                '"risks":["..."],"confidence":3}'
            ),
            "debate": (
                'Schema: {"summary":"...","arguments":["..."],'
                '"risks":["..."],"confidence":3}'
            ),
            "mvp_proposal": (
                'Schema: {"summary":"...","mvp_features":["..."],'
                '"out_of_scope":["..."],"risks":["..."],"confidence":3}'
            ),
            "mvp_vote": (
                'Schema: {"summary":"...","decision":"go_after_clarification",'
                '"mvp_features":["..."],"risks":["..."],"next_step":"...","confidence":3}'
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
            "new_saas": "Новый B2B SaaS",
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
