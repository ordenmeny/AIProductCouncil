# Инструкция: переключение моделей LM Studio

В проекте есть два готовых окружения:

- дома: `Gemma 4 E4B` через `.env.gemma`;
- на практике: `Qwen 3.5 9B` через `.env.qwen`.

Код менять не нужно. Меняется только файл `.env`.

## Домашний компьютер: Gemma 4 E4B

Текущий рабочий `.env` уже настроен под домашнюю модель:

```env
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_MODEL=google/gemma-4-e4b
LM_STUDIO_TIMEOUT_SECONDS=45
LM_STUDIO_MAX_TOKENS=350
LM_STUDIO_QUESTION_MAX_TOKENS=160
LM_STUDIO_TURN_MAX_TOKENS=260
LM_STUDIO_ENABLE_REPAIR=false
```

Если нужно восстановить домашнюю конфигурацию:

```powershell
Copy-Item .env.gemma .env -Force
```

Проверить, что модель загружена в LM Studio:

```powershell
lms ps
```

В списке должен быть identifier:

```text
google/gemma-4-e4b
```

Если identifier отличается, нужно поставить его в `LM_STUDIO_MODEL`.

## Аудитория: Qwen 3.5 9B

Для практики подготовлен файл:

```text
.env.qwen
```

Ожидаемая конфигурация:

```env
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_MODEL=qwen3.5-9b
LM_STUDIO_TIMEOUT_SECONDS=60
LM_STUDIO_MAX_TOKENS=450
LM_STUDIO_QUESTION_MAX_TOKENS=220
LM_STUDIO_TURN_MAX_TOKENS=450
LM_STUDIO_ENABLE_REPAIR=true
```

На компьютере в аудитории:

```powershell
Copy-Item .env.qwen .env -Force
```

Затем запустить LM Studio server:

```powershell
lms server start --port 1234
```

Загрузить модель. Если в LM Studio модель называется иначе, identifier можно выбрать свой:

```powershell
lms load qwen/qwen3.5-9b --identifier qwen3.5-9b --context-length 8192 -y
```

Проверить:

```powershell
lms ps
```

Важно: значение `LM_STUDIO_MODEL` в `.env` должно совпадать с `identifier` из `lms ps`.

## Запуск проекта

```powershell
cd F:\AI_Product_Council_Project
uv sync
uv run streamlit run app.py
```

Открыть:

```text
http://localhost:8501
```

## Как проверить, что работает настоящая модель

1. В интерфейсе не включать `Демо без LLM`.
2. Нажать `1. Получить вопросы агентов`.
3. Если вопросы появились со статусом `llm` или `repaired`, модель отвечает.
4. Ответить на вопросы.
5. Нажать `2. Провести созвон`.
6. Проверить, что в метриках есть ответы `LLM`, а не только `Failed`.

Если много `Failed`, увеличьте таймаут:

```env
LM_STUDIO_TIMEOUT_SECONDS=90
LM_STUDIO_MAX_TOKENS=600
LM_STUDIO_QUESTION_MAX_TOKENS=260
LM_STUDIO_TURN_MAX_TOKENS=600
```

Если модель часто возвращает невалидный JSON, можно включить repair:

```env
LM_STUDIO_ENABLE_REPAIR=true
```

Для Gemma дома repair лучше держать выключенным: она часто начинает объяснять ход рассуждений вместо исправления JSON.
