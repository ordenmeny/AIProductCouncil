# AI Product Council

MVP мультиагентного симулятора продуктового совета. Пользователь описывает идею B2B SaaS, агенты с разными ролями обсуждают её через локальную LLM в LM Studio, голосуют и формируют итоговый проект-план.

## Запуск

1. Установите зависимости:

```bash
uv sync
```

2. Запустите LM Studio local server с OpenAI-compatible API.

3. Создайте `.env` или используйте значения по умолчанию:

```env
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_MODEL=local-model-name
```

4. Запустите интерфейс:

```bash
uv run streamlit run app.py
```

## Проверки

```bash
uv run python -m compileall src app.py
uv run pytest
```
