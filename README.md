# AI Product Council

MVP мультиагентной системы, которая имитирует рабочий созвон IT-команды по созданию нового сервиса, внутреннего инструмента или внедрению новой фичи в существующий продукт.

Пользователь выступает заказчиком или инициатором задачи: описывает идею, ограничения и ожидаемый результат. Агенты с разными ролями задают уточняющие вопросы, пользователь отвечает, после чего агенты обсуждают решение, предлагают MVP/scope, формируют roadmap, выделяют риски, вопросы к заказчику и полезные инсайты.

Система формирует два Markdown-документа:

- `meeting_transcript.md` — протокол созвона с вопросами и репликами агентов;
- `final_plan.md` — итоговый план: суть задачи, MVP/scope, roadmap, риски, вопросы и инсайты.

## Запуск

1. Установить зависимости:

```bash
uv sync
```

2. Запустить LM Studio local server с OpenAI-compatible API.

3. Настроить `.env`.

Для домашнего компьютера с Gemma:

```bash
cp .env.gemma .env
```

Для аудитории с DeepSeek R1 Distill Qwen:

```bash
cp .env.qwen .env
```

Пример `.env`:

```env
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_MODEL=deepseek-r1-distill-qwen-7b-q4-k-m
LM_STUDIO_TIMEOUT_SECONDS=60
LM_STUDIO_MAX_TOKENS=420
LM_STUDIO_QUESTION_MAX_TOKENS=180
LM_STUDIO_TURN_MAX_TOKENS=420
LM_STUDIO_ENABLE_REPAIR=false
```

4. Запустить интерфейс:

```bash
uv run streamlit run app.py
```

5. Открыть:

```text
http://localhost:8501
```

## Сценарий использования

1. Выбрать тип задачи: новый сервис/внутренний инструмент или новая фича.
2. Ввести идею, ограничения и ожидаемый результат.
3. Нажать `1. Получить вопросы агентов`.
4. Ответить на вопросы агентов в одном текстовом поле.
5. Нажать `2. Провести созвон`.
6. Проверить ход созвона, статусы `llm/repaired/failed`, протокол и итоговый план.
7. Сохранить Markdown и JSON в `outputs/`.

## Роли агентов

- `Product Manager` — ценность, пользовательский сценарий, MVP и приоритеты.
- `Business Value / Adoption Lead` — бизнес-ценность, внедрение и проверка пользы.
- `Tech Lead / Architect` — реализация, архитектура, ограничения и технические риски.
- `UX Researcher / Designer` — сценарии, удобство, onboarding и доверие к результату.
- `Security / Data Expert` — данные, доступы, хранение и безопасность.
- `Skeptic / Risk Officer` — слабые места, сужение scope и неочевидные риски.

## Проверки

```bash
uv run python -m compileall src app.py
uv run pytest
```

## Модели

Домашняя модель:

```text
google/gemma-4-e4b
```

Модель для аудитории:

```text
deepseek-r1-distill-qwen-7b-q4-k-m
```

Через LM Studio CLI:

```powershell
lms server start --port 1234
lms load deepseek-r1-distill-qwen-7b-q4-k-m --identifier deepseek-r1-distill-qwen-7b-q4-k-m --context-length 8192 -y
```

Если LM Studio показывает другой identifier в `lms ps`, укажите в `LM_STUDIO_MODEL` ровно это значение.

Подробнее: `instaction.md`.

## Настройки генерации

- `LM_STUDIO_MAX_TOKENS` — общий fallback-лимит ответа.
- `LM_STUDIO_QUESTION_MAX_TOKENS` — лимит для уточняющих вопросов.
- `LM_STUDIO_TURN_MAX_TOKENS` — лимит для реплик созвона.
- `LM_STUDIO_ENABLE_REPAIR` — делать ли повторный запрос для исправления невалидного JSON.

Для Gemma и DeepSeek R1 repair выключен, потому что reasoning-модели часто тратят второй запрос на рассуждения вместо чистого JSON. Если нужен эксперимент с более строгим JSON, можно включить `LM_STUDIO_ENABLE_REPAIR=true`, но демо будет медленнее.

DeepSeek R1 Distill может отвечать reasoning-текстом на английском. Это нормально для модели, но не для итогового протокола: приложение отбрасывает такие ответы и использует русский fallback. Поэтому в метриках может быть больше `Fallback`, зато вопросы, реплики и итоговый план должны оставаться на русском.

Если в браузере появляется `NotFoundError removeChild`, отключите авто-перевод или расширение переводчика для `localhost:8501`. Переводчик меняет DOM Streamlit/React и может ломать перерисовку страницы.
