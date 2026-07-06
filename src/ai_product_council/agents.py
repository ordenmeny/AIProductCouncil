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
            description="Отвечает за ценность продукта, пользователей, сценарии и приоритеты MVP.",
            private_context_path=CONTEXT_DIR / "product_manager.md",
            system_prompt=(
                "Ты Product Manager в B2B SaaS. Думай через ICP, JTBD, ценность, "
                "сценарии использования и приоритизацию. Будь конкретным и прагматичным."
            ),
        ),
        AgentRole(
            name="B2B Sales / GTM Lead",
            slug="sales_gtm",
            description="Оценивает ICP, продажи, цену, каналы и выход на рынок.",
            private_context_path=CONTEXT_DIR / "sales_gtm.md",
            system_prompt=(
                "Ты B2B Sales / GTM Lead. Оценивай рынок, покупателя, цикл сделки, "
                "каналы продаж, цену и первые коммерческие проверки."
            ),
        ),
        AgentRole(
            name="Tech Lead / Architect",
            slug="tech_lead",
            description="Оценивает архитектуру, интеграции, сложность и технический план.",
            private_context_path=CONTEXT_DIR / "tech_lead.md",
            system_prompt=(
                "Ты Tech Lead / Architect. Оцени техническую реализуемость, архитектуру, "
                "интеграции, риски разработки и реалистичный MVP-стек."
            ),
        ),
        AgentRole(
            name="Security / Compliance Expert",
            slug="security",
            description="Проверяет безопасность, доступы, данные и enterprise compliance.",
            private_context_path=CONTEXT_DIR / "security.md",
            system_prompt=(
                "Ты Security / Compliance Expert. Проверяй работу с данными, доступы, "
                "аудит, privacy, compliance и риски для enterprise-клиентов."
            ),
        ),
        AgentRole(
            name="Skeptic / Risk Officer",
            slug="skeptic",
            description="Ищет слабые места, противоречия и причины сузить или не запускать проект.",
            private_context_path=CONTEXT_DIR / "skeptic.md",
            system_prompt=(
                "Ты Skeptic / Risk Officer. Конструктивно критикуй идею, ищи слабые "
                "места, ложные предположения и способы уменьшить риск провала."
            ),
        ),
    ]
