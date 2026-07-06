# Инструкция: подключение Qwen в аудитории через LM Studio

Рекомендуемая модель для аудитории с RTX 4060:

```text
qwen/qwen3-8b
```

Квантизация:

```text
Q4_K_M
```

## 1. Подготовить LM Studio

1. Открыть LM Studio.
2. Найти и скачать модель `qwen/qwen3-8b`.
3. Выбрать квантизацию `Q4_K_M`.
4. Перейти в раздел Local Server / Developer.
5. Запустить сервер на порту `1234`.

## 2. Вариант через CLI `lms`

Если доступна команда `lms`, можно запустить всё из PowerShell:

```powershell
lms server start --port 1234
lms load qwen/qwen3-8b --identifier qwen3-8b --context-length 8192 -y
```

Проверить, что модель загружена:

```powershell
lms ps
```

В списке должна быть загруженная модель с identifier:

```text
qwen3-8b
```

## 3. Настроить проект

В файле `.env` в корне проекта нужно указать модель:

```env
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_MODEL=qwen3-8b
LM_STUDIO_TIMEOUT_SECONDS=45
LM_STUDIO_MAX_TOKENS=350
```

Если ответы всё ещё часто уходят в fallback из-за таймаута, можно увеличить значения:

```env
LM_STUDIO_TIMEOUT_SECONDS=90
LM_STUDIO_MAX_TOKENS=500
```

## 4. Код менять не нужно

Код проекта менять не требуется, если:

- LM Studio server работает на `http://localhost:1234/v1`;
- в `.env` указан правильный `LM_STUDIO_MODEL`;
- identifier модели совпадает с тем, что показывает `lms ps`.

Если в `lms ps` identifier отличается, нужно поставить именно его:

```env
LM_STUDIO_MODEL=identifier-из-lms-ps
```

## 5. Запустить проект

В PowerShell:

```powershell
cd F:\AI_Product_Council_Project
uv sync
uv run streamlit run app.py
```

Открыть в браузере:

```text
http://localhost:8501
```

## 6. Проверить работу через Qwen

1. В Streamlit открыть sidebar.
2. Выключить галочку `Быстрый демо-режим`.
3. Ввести идею B2B SaaS или оставить пример.
4. Нажать `Запустить совет`.

Если режим быстрый выключен, приложение обращается к LM Studio и использует модель Qwen.

Если отдельные агенты показывают fallback-ответы, это значит, что модель не успела вернуть валидный JSON за заданный таймаут. В этом случае можно увеличить `LM_STUDIO_TIMEOUT_SECONDS`.
