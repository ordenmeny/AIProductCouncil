# AI Product Council

Мультиагентный симулятор рабочего созвона IT-команды. Пользователь вводит идею сервиса, стартапа или фичи, а пять агентов с разными ролями задают уточняющие вопросы, анализируют идею, спорят, предлагают MVP, голосуют и формируют протокол и итоговый план.

## Stack

- Backend: Python 3.11+, FastAPI, Pydantic, OpenAI-compatible client
- Package manager: uv
- LLM runtime: LM Studio
- Frontend: React + Vite
- Storage: JSON-файлы в `data/meetings`

## Локальный запуск

1. Запустить LM Studio server с моделью Qwen2.5-7B-Instruct Q4_K_M.
2. Скопировать настройки:

```bash
cp .env.qwen-example .env.qwen
```

3. Установить backend-зависимости и запустить API:

```bash
uv sync
uv run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

4. Установить frontend-зависимости и запустить UI:

```bash
cd frontend
npm install
npm run dev
```

5. Открыть `http://localhost:5173`.

## API

- `POST /api/meetings` — создать созвон по идее.
- `GET /api/meetings/{id}` — получить состояние созвона.
- `POST /api/meetings/{id}/answers` — сохранить ответы пользователя на вопросы агентов.
- `POST /api/meetings/{id}/advance` — перейти к следующей фазе.
- `GET /api/meetings/{id}/export/protocol.md` — скачать протокол.
- `GET /api/meetings/{id}/export/final-plan.md` — скачать итоговый план.

## Фазы

1. Уточняющие вопросы.
2. Ответы пользователя одним пакетом.
3. Индивидуальный анализ.
4. Обсуждение и спор.
5. Предложения по MVP.
6. Голосование.
7. Финальный отчет.

Оркестратор является детерминированным кодом: он управляет фазами, хранит состояние, передает агентам только разрешенный контекст, валидирует JSON-ответы и собирает итоговые Markdown-документы.
