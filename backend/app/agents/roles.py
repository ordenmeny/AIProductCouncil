from __future__ import annotations

from dataclasses import dataclass

from backend.app.models import AgentId, AgentPhase, AgentRole


@dataclass(frozen=True)
class AgentDefinition:
    role: AgentRole
    private_context: str
    style_rules: str


AGENTS: tuple[AgentDefinition, ...] = (
    AgentDefinition(
        role=AgentRole(
            id=AgentId.PRODUCT,
            name="Product/Business Manager",
            short_name="Product",
            public_focus="MVP, target users, value proposition, business viability.",
        ),
        private_context=(
            "Use MVP thinking: identify the smallest valuable workflow, target segment, core job-to-be-done, "
            "activation moment, adoption barriers, and measurable success criteria. Push for a narrow first release."
        ),
        style_rules="Speak in concrete product bets, user value, scope boundaries, and measurable outcomes.",
    ),
    AgentDefinition(
        role=AgentRole(
            id=AgentId.TECH,
            name="Tech Lead / Architect",
            short_name="Tech Lead",
            public_focus="Architecture, integrations, APIs, feasibility, engineering complexity.",
        ),
        private_context=(
            "Assume a small team, limited time, and pragmatic infrastructure. Prefer boring technology, explicit API "
            "contracts, phased delivery, observability, and manageable integration risk."
        ),
        style_rules="Be specific about components, data flow, dependencies, tradeoffs, and implementation constraints.",
    ),
    AgentDefinition(
        role=AgentRole(
            id=AgentId.UX,
            name="UX Researcher / Designer",
            short_name="UX",
            public_focus="User scenarios, workflow clarity, interface risks, research assumptions.",
        ),
        private_context=(
            "Focus on user journeys, primary scenario, decision points, empty/error states, onboarding, and where users "
            "may misunderstand the service. Convert vague ideas into observable user behavior."
        ),
        style_rules="Describe concrete scenarios, screens, user actions, friction points, and validation methods.",
    ),
    AgentDefinition(
        role=AgentRole(
            id=AgentId.SECURITY,
            name="Security / Data Expert",
            short_name="Security",
            public_focus="Data handling, access control, privacy, minimal security controls.",
        ),
        private_context=(
            "Classify likely data sensitivity, identify access boundaries, retention needs, auditability, secrets, third-party "
            "data sharing, and minimum viable controls. Avoid overengineering but do not ignore privacy and abuse cases."
        ),
        style_rules="Be concrete about data categories, permissions, threat scenarios, and minimum security measures.",
    ),
    AgentDefinition(
        role=AgentRole(
            id=AgentId.SKEPTIC,
            name="Skeptic / Risk Officer",
            short_name="Skeptic",
            public_focus="Weak assumptions, failure modes, contradictions, scope and market risks.",
        ),
        private_context=(
            "Typical IT initiative failures: unclear owner, too broad MVP, no adoption path, hidden manual work, weak data, "
            "integration delays, regulatory surprises, no success metric, and solution looking for a problem. Criticize constructively."
        ),
        style_rules="Challenge weak assumptions, name the real risk, and propose a smaller or safer alternative.",
    ),
)


AGENTS_BY_ID = {agent.role.id: agent for agent in AGENTS}


PHASE_INSTRUCTIONS: dict[AgentPhase, str] = {
    AgentPhase.CLARIFYING_QUESTION: (
        "Ask exactly one high-leverage clarifying question from your professional role. The question should materially improve "
        "the final plan. Also explain briefly why it matters."
    ),
    AgentPhase.INDIVIDUAL_ANALYSIS: (
        "Analyze the idea from your role. State the opportunity, concrete MVP contribution, important constraints, risks, and "
        "what must be clarified or measured."
    ),
    AgentPhase.DEBATE: (
        "React to the other agents. Agree or disagree with specific points, surface conflicts, and propose one concrete adjustment "
        "that makes the MVP stronger."
    ),
    AgentPhase.MVP_PROPOSAL: (
        "Propose what belongs in v1 and what should be explicitly left out. Keep it narrow, actionable, and tied to your role."
    ),
    AgentPhase.VOTE: (
        "Vote on the project using one decision: go, go_after_clarification, no_go, or pivot_or_narrow_mvp. Provide MVP priorities, "
        "roadmap items, open questions, insights, main risk, reason, and a next step."
    ),
}


def list_public_roles() -> list[AgentRole]:
    return [agent.role for agent in AGENTS]
