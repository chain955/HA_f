# Health Assistant

Локальный ассистент для анализа здоровья, физических нагрузок и тренировок.
Работает **офлайн** — все LLM-вызовы через локальный Ollama, данные в SQLite,
векторный поиск через embedded ChromaDB.

**Статус:** Phase 2 завершён. Текущая функциональность приближена к архитектуре
v2 (см. `health_assistant_architecture_v2.yaml`): мульти-модельный роутинг,
RAG (демо-набор), semantic memory, 4 маршрута пайплайна с planner-loop и
template-планами, stage events + token streaming в чате.

---

## Возможности

- **Чат с ассистентом** (`/chat`) — WebSocket со stage events и token streaming.
- **4 маршрута пайплайна:**
  - `fast_direct_answer` — приветствия, off-topic, общие вопросы;
  - `tool_simple` — запрос/анализ данных через фиксированные tool-calls;
  - `template_plan` — шаблонные планы (недельный план, проверка перетренированности,
    отчёт о восстановлении, прогресс);
  - `planner` — LLM-цикл с JSON tool-calls (до 5 итераций) для сложных запросов.
- **Безопасность** — pattern-based фильтр (экстренные симптомы → редирект к врачу).
- **Multi-model LLM** — роли `intent_llm`, `safety_llm`, `response`, `planner`;
  модель на каждую роль настраивается в админке.
- **RAG (demo KB)** — 20–40 чанков по 5 категориям (physiology_norms,
  training_principles, recovery_science, sport_specific, nutrition_basics).
- **Semantic memory v1** — сохранение эмбеддингов Q/A и retrieval top-k в
  Context Builder.
- **Data Processing** — recovery score, strain score, heart-rate zones,
  overtraining detection, training load (monotony/strain/ACR), trend analyzer.
- **Админ-панель** — логи пайплайна, конфиг моделей по ролям, RAG browser,
  semantic memory browser, диагностика, генератор seed-данных.
- **Наблюдаемость** — каждый LLM-вызов логируется в `llm_calls`, pipeline-лог
  содержит `stage_trace`, `llm_role_usage`, `rag_chunks_used`.

---

## Tech Stack

| Слой        | Технология                                                    |
|-------------|---------------------------------------------------------------|
| Backend     | FastAPI (Python 3.11+), async                                 |
| Database    | SQLite + SQLAlchemy 2.0 + Alembic                             |
| LLM         | Ollama (multi-model через LLM Registry)                       |
| Vector DB   | ChromaDB (embedded, persistent volume)                        |
| Embeddings  | `nomic-embed-text` через Ollama `/api/embeddings`             |
| Chat UI     | HTML + vanilla JS + WebSocket (stage events + token stream)   |
| Admin UI    | Jinja2 + HTMX + Pico CSS                                      |
| Container   | Docker Compose                                                |
| Сеть        | Внешняя `ollama-net` (Ollama поднимается отдельно)            |

---

## Быстрый старт

### Требования

- Docker + Docker Compose.
- Отдельно поднятый Ollama в сети `ollama-net` с установленными моделями
  (минимум одна LLM + `nomic-embed-text` для эмбеддингов).

```bash
# Пример поднятия Ollama (если ещё не поднят)
docker network create ollama-net
docker run -d --name ollama --network ollama-net \
    -v ollama_data:/root/.ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull qwen2.5:7b
docker exec ollama ollama pull nomic-embed-text
```

### Запуск приложения

```bash
# 1. Настроить .env
cp .env.example .env
# отредактировать при необходимости (модели, креды админки)

# 2. Собрать и запустить
docker compose up --build -d

# 3. Применить миграции
docker compose exec app alembic upgrade head

# 4. Засеять тестовые данные (по умолчанию — 30 дней, 1 профиль, без аномалий)
docker compose exec app python scripts/seed_data.py

# 5. Засеять демо-Knowledge Base (RAG)
docker compose exec app python scripts/seed_knowledge.py
```

### Endpoints

| URL                               | Назначение                                         |
|-----------------------------------|----------------------------------------------------|
| `http://localhost:8000/`          | редирект на `/chat`                                |
| `http://localhost:8000/chat`      | тестовый чат                                       |
| `http://localhost:8000/admin`     | админ-панель (Basic Auth)                          |
| `http://localhost:8000/admin/llm` | конфиг моделей по ролям                            |
| `http://localhost:8000/admin/seed`| генератор тестовых данных                          |
| `http://localhost:8000/admin/knowledge` | RAG browser                                 |
| `http://localhost:8000/admin/memory`    | semantic memory browser                     |
| `http://localhost:8000/admin/diagnostics` | диагностика сервисов                      |
| `http://localhost:8000/health`    | health check                                       |

Дефолтные креды админки — `admin/admin` (см. `.env.example`).

---

## Структура проекта

```
health_assistant/
├── app/
│   ├── main.py                       # FastAPI entry
│   ├── config.py                     # pydantic-settings
│   ├── db.py                         # AsyncSession factory
│   ├── models/                       # SQLAlchemy модели
│   ├── schemas/                      # Pydantic-схемы
│   ├── services/
│   │   ├── llm_service.py            # Ollama HTTP-клиент
│   │   ├── llm_registry.py           # per-role routing + runtime overrides
│   │   ├── llm_call_logger.py        # per-request лог LLM-вызовов
│   │   ├── embedding_service.py      # эмбеддинги через Ollama
│   │   ├── vector_store.py           # обёртка ChromaDB
│   │   ├── semantic_memory.py        # Q/A эмбеддинги + retrieval
│   │   ├── logging_service.py        # JSON logging
│   │   └── data_processing/          # recovery/strain/HR-zones/overtraining/...
│   ├── pipeline/
│   │   ├── context_builder.py
│   │   ├── intent_detection.py       # rule-based + LLM stage 2
│   │   ├── safety_check.py           # pattern-based (v2 отложено)
│   │   ├── router.py                 # 4 маршрута
│   │   ├── tool_executor.py
│   │   ├── template_plan_executor.py # weekly/overtraining/recovery/progress
│   │   ├── planner.py                # LLM tool-calls loop
│   │   ├── response_generator.py     # RAG + streaming
│   │   ├── memory_update.py          # async short/long/semantic
│   │   ├── stage_tracker.py          # stage_trace + llm_role_usage
│   │   ├── stage_events.py           # pub/sub для WebSocket
│   │   └── orchestrator.py           # entry point пайплайна
│   ├── tools/                        # db_tools + rag_retrieve
│   ├── api/                          # chat (WS) + admin (REST)
│   ├── admin/                        # Jinja2 + HTMX
│   └── static/                       # CSS/JS
├── alembic/versions/                 # миграции
├── scripts/
│   ├── seed_data.py                  # Seed Generator v2
│   └── seed_knowledge.py             # RAG демо-чанки
├── tests/                            # unit + integration
├── data/                             # SQLite + chroma persistent
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── CLAUDE.md                         # briefing для Claude
├── PHASE2_PLAN.md                    # план Phase 2 (issues #21–#37)
└── health_assistant_architecture_v2.yaml   # референс-архитектура
```

---

## Pipeline flow

```
User Query
  → Context Builder      # session history + profile + RAG + semantic memory
  → Intent Detection     # rule-based → LLM fallback (low confidence)
  → Safety Check         # pattern-based
  → Router               # fast_direct_answer | tool_simple | template_plan | planner
  ↓
  [blocked]       → redirect
  [fast_direct]   → Response Generator → reply
  [tool_simple]   → Tool Executor → Response Generator → reply
  [template_plan] → Template Executor → Response Generator → reply
  [planner]       → Planner loop (LLM ↔ Tool Executor) → Response Generator → reply
  ↓
  Memory Update (async): short-term + long-term + semantic
  Response Delivery: stage events + token streaming по WebSocket
```

---

## Расширение

### Knowledge Base (RAG)

Сейчас в `scripts/seed_knowledge.py` — 20–40 демо-чанков по 5 категориям YAML.
Чтобы добавить свои источники:

1. Добавить чанки в `seed_knowledge.py` (id, text, category, source, confidence,
   опционально sport_type / experience_level).
2. Пересобрать индекс: `docker compose exec app python scripts/seed_knowledge.py`.
3. Или через админку `/admin/knowledge` — add/delete/reindex по одному.

### Модели LLM

- Базовая модель в `.env` (`OLLAMA_MODEL`).
- Per-role модели — через админку `/admin/llm` (сохраняется в таблице
  `llm_role_config`), либо через переменные `LLM_*_MODEL` в `.env`.
- Если указанная модель недоступна — fallback на `OLLAMA_MODEL` с WARN в логах.

---

## Разработка

### Запуск тестов

```bash
# Unit + integration внутри контейнера
docker compose exec app pytest

# Или локально (требует установленные зависимости)
pytest tests/
```

Покрытие:

- **Unit:** intent detection, safety, routing, data processing, planner
  (mock Ollama), template executor, RAG retrieval (mock Chroma).
- **Integration:** `tests/integration/test_orchestrator_flows.py` — 5 сценариев:
  fast_path, tool_simple, template_plan, planner_loop, safety_block.

### Миграции

```bash
# Создать новую ревизию
docker compose exec app alembic revision -m "описание"

# Применить
docker compose exec app alembic upgrade head

# Откатить на одну ревизию
docker compose exec app alembic downgrade -1
```

---

## Отложено на v3

С пометками `TODO v3` в коде — см. соответствующие заглушки:

- **Data ingestion из реального API** (+ Anomaly detection, Deduplication).
  Источник данных — только Seed Generator v2.
- **Safety Check v2** (контекстный LLM-анализ).
- **Output Validation v2** (hallucination / medical advice check).
- **Periodization** (макро/мезо/микроциклы).
- **Proactive Alerts** (HRV drop, RHR spike, weekly summary).
- **Testing & Evaluation** (eval-датасеты, бенчмарки, RAG quality).

---

## Лицензия

Внутренний проект — лицензия не определена. Использование с разрешения автора.
