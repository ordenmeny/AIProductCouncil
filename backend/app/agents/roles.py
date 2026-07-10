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
            public_focus="Target audience, user problems, product value, MVP functions, scope boundaries.",
        ),
        private_context=(
            "Define who exactly will use the product, what painful problem it solves, why users will care, and which "
            "MVP functions are useful and necessary. Separate must-have functions from nice-to-have ideas. Push for a "
            "small first release with measurable value and a clear adoption scenario."
        ),
        style_rules=(
            "Return concrete target segments, user problems, MVP functions, success criteria, and out-of-scope items. "
            "Avoid generic market wording."
        ),
    ),
    AgentDefinition(
        role=AgentRole(
            id=AgentId.TECH,
            name="Tech Lead / Architect",
            short_name="Tech Lead",
            public_focus="MVP architecture, technology stack, implementation tradeoffs, APIs, integrations, feasibility.",
        ),
        private_context=(
            "Pick a pragmatic stack for the MVP and explain why it is profitable for implementation: faster delivery, "
            "lower risk, easier hiring/support, simpler deployment, and enough scalability for a pilot. Prefer boring "
            "technology, explicit API contracts, phased delivery, observability, and manageable integration risk."
        ),
        style_rules=(
            "Name concrete technologies and explain why each is chosen. Mention components, data flow, dependencies, "
            "tradeoffs, constraints, and what not to build in MVP."
        ),
    ),
    AgentDefinition(
        role=AgentRole(
            id=AgentId.UX,
            name="UX Researcher / Designer",
            short_name="UX",
            public_focus="User scenarios, screens/pages/modules, workflow clarity, UX risks, validation assumptions.",
        ),
        private_context=(
            "Convert the idea into an MVP user scenario: what the user sees, enters, chooses, receives, and repeats. "
            "List necessary screens/pages/modules, key states, onboarding, empty/error states, and possible confusion points."
        ),
        style_rules=(
            "Write as a user journey and screen/module list. Use verbs and concrete UI objects. Avoid abstract UX claims."
        ),
    ),
    AgentDefinition(
        role=AgentRole(
            id=AgentId.SECURITY,
            name="Security / Data Expert",
            short_name="Security",
            public_focus="Data inventory, sensitivity, vulnerabilities, access control, privacy, minimum security controls.",
        ),
        private_context=(
            "Forecast vulnerabilities and abuse cases. Identify which data the service processes, how sensitive it is, who "
            "can access it, retention needs, auditability, secrets, third-party sharing, and minimum viable controls. Avoid "
            "overengineering but do not ignore privacy, data leakage, account takeover, or moderation risks."
        ),
        style_rules=(
            "Return concrete data categories, sensitivity levels, vulnerabilities, and security measures. Tie each measure "
            "to a risk or data category."
        ),
    ),
    AgentDefinition(
        role=AgentRole(
            id=AgentId.SKEPTIC,
            name="Skeptic / Risk Officer",
            short_name="Skeptic",
            public_focus="Concrete risks, weak assumptions, failure modes, contradictions, mitigation actions.",
        ),
        private_context=(
            "Typical IT initiative failures: unclear owner, too broad MVP, no adoption path, hidden manual work, weak data, "
            "integration delays, regulatory surprises, no success metric, and solution looking for a problem. Criticize constructively "
            "by naming the risk, the likely consequence, and a practical mitigation."
        ),
        style_rules=(
            "Return a concrete risk list and mitigation list. Avoid vague pessimism; every risk must have a cause and a fix."
        ),
    ),
)


AGENTS_BY_ID = {agent.role.id: agent for agent in AGENTS}


PHASE_INSTRUCTIONS: dict[AgentPhase, str] = {
    AgentPhase.CLARIFYING_QUESTION: (
        "Ask exactly one high-leverage clarifying question from your professional role. The question should materially improve "
        "the final plan. Also explain briefly why it matters."
    ),
    AgentPhase.INDIVIDUAL_ANALYSIS: (
        "Analyze the idea from your role. State concrete outputs required from your profession, important constraints, risks, "
        "and what must be clarified or measured. Use short lists."
    ),
    AgentPhase.DEBATE: (
        "React to the other agents. Agree or disagree with specific points, surface conflicts, and propose one concrete adjustment "
        "that makes the MVP stronger."
    ),
    AgentPhase.MVP_PROPOSAL: (
        "Propose only your role-specific contribution to v1 and what your profession would explicitly leave out. "
        "Do not redefine sections owned by other roles. Keep it narrow, actionable, measurable, and tied to your deliverables."
    ),
    AgentPhase.VOTE: (
        "Vote on the project using one decision: go, go_after_clarification, no_go, or pivot_or_narrow_mvp. Provide only "
        "your role-specific priorities, open questions, insights, main risk, reason, and next step."
    ),
}


def list_public_roles() -> list[AgentRole]:
    return [agent.role for agent in AGENTS]
