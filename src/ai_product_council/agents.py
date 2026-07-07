from __future__ import annotations

from pathlib import Path

from ai_product_council.models import AgentRole


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_DIR = PROJECT_ROOT / "data" / "private_contexts"


def get_default_agents() -> list[AgentRole]:
    return [
        AgentRole(
            name="Product Manager",
            slug="product_manager",
            description="Отвечает за ценность, пользователей, сценарии, MVP и приоритеты.",
            private_context_path=CONTEXT_DIR / "product_manager.md",
            system_prompt=(
                "Ты Product Manager в B2B SaaS. Смотри на ICP, JTBD, пользовательскую боль, "
                "первый полезный workflow и приоритизацию MVP."
            ),
        ),
        AgentRole(
            name="B2B Sales / GTM Lead",
            slug="sales_gtm",
            description="Оценивает покупателя, продажи, цену, ICP и выход на рынок.",
            private_context_path=CONTEXT_DIR / "sales_gtm.md",
            system_prompt=(
                "Ты B2B Sales / GTM Lead. Смотри на покупателя, экономический эффект, "
                "каналы продаж, дизайн-партнёров, цену и первые коммерческие проверки."
            ),
        ),
        AgentRole(
            name="Tech Lead / Architect",
            slug="tech_lead",
            description="Оценивает архитектуру, стек, интеграции, сложность и реализацию.",
            private_context_path=CONTEXT_DIR / "tech_lead.md",
            system_prompt=(
                "Ты Tech Lead / Architect. Смотри на реализуемость, архитектуру MVP, "
                "интеграции, хранение данных, API, сложность и технические риски."
            ),
        ),
        AgentRole(
            name="UX Researcher / Designer",
            slug="ux_researcher",
            description="Оценивает пользовательский workflow, onboarding и удобство.",
            private_context_path=CONTEXT_DIR / "ux_researcher.md",
            system_prompt=(
                "Ты UX Researcher / Designer. Смотри на пользовательский сценарий, "
                "точки трения, onboarding, интерфейс, доверие к AI-результату и проверку UX-гипотез."
            ),
        ),
        AgentRole(
            name="Security / Compliance Expert",
            slug="security",
            description="Проверяет данные, доступы, безопасность и compliance-риски.",
            private_context_path=CONTEXT_DIR / "security.md",
            system_prompt=(
                "Ты Security / Compliance Expert. Смотри на клиентские данные, роли доступа, "
                "audit trail, privacy, compliance и enterprise-ожидания."
            ),
        ),
        AgentRole(
            name="Skeptic / Risk Officer",
            slug="skeptic",
            description="Ищет слабые места, причины провала и способы сузить риск.",
            private_context_path=CONTEXT_DIR / "skeptic.md",
            system_prompt=(
                "Ты Skeptic / Risk Officer. Конструктивно критикуй идею, ищи ложные "
                "предпосылки, слабую дифференциацию, лишний scope и причины не запускать."
            ),
        ),
    ]
