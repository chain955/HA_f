"""Seed Knowledge Base — наполнение ChromaDB знаниями для RAG (Issue #24).

Демо-набор (~30 чанков) по 5 категориям архитектуры v2:
  * physiology_norms      — нормы HRV, RHR, VO2max, sleep stages
  * training_principles   — прогрессивная перегрузка, периодизация, суперкомпенсация
  * recovery_science      — sleep / HRV / RHR как маркеры восстановления
  * sport_specific        — бег, велосипед, силовые (базовые принципы)
  * nutrition_basics      — макронутриенты, гидратация, timing

Каждый чанк эмбеддится через EmbeddingService (nomic-embed-text) и пишется:
  1. в ChromaDB-коллекцию knowledge_base (документ + эмбеддинг + metadata)
  2. в SQLite-таблицу rag_chunks (метаданные для отображения в админке)

Идемпотентность:
  * id чанка — детерминированный (uuid5 от текста + категория),
    так что при повторном запуске скрипта старые записи пропускаются.
  * Флаг --reset пересоздаёт коллекцию и таблицу с нуля.

Запуск (внутри контейнера app):
    docker compose exec app python scripts/seed_knowledge.py
    docker compose exec app python scripts/seed_knowledge.py --reset

Как добавить новые источники:
    1. Допишите dict в CHUNKS (text + metadata).
    2. Придерживайтесь категорий из списка CATEGORIES — иначе фильтры
       rag_retrieve не найдут чанк.
    3. Запустите скрипт — он проэмбеддит новые чанки и добавит их в Chroma.
    4. Для переиндексации всего — флаг --reset.

TODO v3:
    * Загрузка из внешних источников (papers, articles).
    * Автоматический chunking-pipeline (длинные документы → чанки).
    * Обновление уже проэмбедженных чанков при изменении текста
      (сейчас при изменении текста меняется id — старая версия остаётся
      до --reset).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Добавляем корень проекта в PYTHONPATH чтобы скрипт работал standalone
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import delete, select

from app.db import AsyncSessionLocal
from app.models.rag_chunk import RAGChunk
from app.services.embedding_service import embedding_service
from app.services.vector_store import COLLECTION_KNOWLEDGE_BASE, vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("seed_knowledge")

CATEGORIES = {
    "physiology_norms",
    "training_principles",
    "recovery_science",
    "sport_specific",
    "nutrition_basics",
}

# UUID namespace для детерминированной генерации id чанков
_NS = uuid.UUID("8c0f1ebd-4b35-4d4f-9b9e-8f1c7e5e3c01")


# ---------------------------------------------------------------------------
# Демо-набор чанков (30 шт.)
# ---------------------------------------------------------------------------

CHUNKS: list[dict[str, Any]] = [
    # ==================== physiology_norms (6) ====================
    {
        "text": (
            "HRV (heart rate variability, вариабельность сердечного ритма) — "
            "разброс интервалов между ударами сердца в покое, обычно измеряется "
            "метрикой RMSSD в миллисекундах. Значения 20–50 мс считаются "
            "нормой для большинства взрослых, 50–100 мс — показатель хорошего "
            "восстановления и тренированности. Рост HRV на фоне стабильной "
            "нагрузки — признак адаптации, резкое падение на 20%+ от личного "
            "базового уровня — маркер стресса или недовосстановления."
        ),
        "metadata": {
            "category": "physiology_norms",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Resting heart rate (RHR) — пульс в состоянии покоя утром сразу "
            "после пробуждения. У взрослых норма 60–80 уд/мин, у тренированных "
            "людей типично 40–60 уд/мин. Скачок RHR на 5–10 ударов выше личного "
            "базового уровня два дня подряд — маркер перетренированности, "
            "инфекции или сильного стресса."
        ),
        "metadata": {
            "category": "physiology_norms",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "VO2max — максимальное потребление кислорода, мл/кг/мин. Основной "
            "показатель аэробной мощности. У нетренированных взрослых 30–40, у "
            "любителей 40–50, у профи выносливости 60–85. Тренируется "
            "интервалами на 90–95% ЧССmax (4×4 минуты) и длительными "
            "аэробными сессиями в зонах 1–2."
        ),
        "metadata": {
            "category": "physiology_norms",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Структура сна: цикл ~90 минут, за ночь 4–6 циклов. Фазы: N1 "
            "(засыпание, 5%), N2 (лёгкий сон, 45–55%), N3 (глубокий сон, "
            "13–23%) и REM (20–25%). Глубокий сон важен для физического "
            "восстановления и гормонального фона, REM — для консолидации "
            "памяти и ментального восстановления."
        ),
        "metadata": {
            "category": "physiology_norms",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Максимальная ЧСС (HRmax) приблизительно рассчитывается формулой "
            "Танаки: 208 − 0.7 × возраст (точнее классической 220 − возраст). "
            "Разброс между людьми ±10–15 уд/мин, поэтому для точности "
            "используется тест на стадионе или беговой дорожке с постепенным "
            "повышением нагрузки до отказа."
        ),
        "metadata": {
            "category": "physiology_norms",
            "source": "public domain summary",
            "confidence": "medium",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Sleep need у взрослых — 7–9 часов. Регулярный дефицит сна (<6 "
            "часов) снижает производительность, HRV, толерантность к "
            "глюкозе и замедляет восстановление. У спортсменов в активной "
            "фазе подготовки потребность может достигать 9–10 часов."
        ),
        "metadata": {
            "category": "physiology_norms",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    # ==================== training_principles (6) ====================
    {
        "text": (
            "Принцип прогрессивной перегрузки: для продолжения адаптации "
            "тренировочный стимул (объём, интенсивность, сложность) должен "
            "постепенно расти. Стандартный ориентир — +5–10% объёма в неделю, "
            "и не более одной переменной одновременно (либо объём, либо "
            "интенсивность)."
        ),
        "metadata": {
            "category": "training_principles",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Суперкомпенсация: после нагрузки наступает фаза утомления, "
            "затем восстановления до исходного уровня, а после — период "
            "повышенной работоспособности (1–3 дня у любителей). Следующая "
            "тренировка идеально попадает в окно суперкомпенсации. Слишком "
            "ранняя повторная нагрузка ведёт к накоплению усталости, слишком "
            "поздняя — к потере эффекта."
        ),
        "metadata": {
            "category": "training_principles",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Базовая периодизация: чередование циклов нагрузки 3:1 (три "
            "недели роста + одна разгрузочная на 50–60% объёма). Разгрузка "
            "нужна для суперкомпенсации и снижения риска травм. Без "
            "разгрузочных недель через 4–6 недель накопленное утомление "
            "останавливает прогресс."
        ),
        "metadata": {
            "category": "training_principles",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "intermediate",
        },
    },
    {
        "text": (
            "Принцип специфичности (SAID): организм адаптируется именно к "
            "тем нагрузкам, которые получает. Для развития выносливости "
            "нужна длительная аэробная работа, для силы — подъём "
            "субмаксимальных весов, для скорости — короткие интервалы на "
            "максимуме. Общая физическая подготовка не заменяет "
            "специфических тренировок."
        ),
        "metadata": {
            "category": "training_principles",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Правило 80/20 в циклических видах спорта: примерно 80% объёма "
            "выполняется в лёгких аэробных зонах (Z1–Z2, разговорный темп), "
            "и только 20% — в пороговых и анаэробных зонах (Z3–Z5). Такая "
            "структура даёт максимальный прогресс при минимальном риске "
            "перетренированности."
        ),
        "metadata": {
            "category": "training_principles",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Acute-chronic workload ratio (ACWR) — отношение острой "
            "недельной нагрузки (последние 7 дней) к хронической "
            "среднемесячной (последние 28 дней). Безопасный диапазон "
            "0.8–1.3, зона 1.3–1.5 — повышенный риск травмы, >1.5 — высокий "
            "риск. Ниже 0.8 — детренировка."
        ),
        "metadata": {
            "category": "training_principles",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    # ==================== recovery_science (6) ====================
    {
        "text": (
            "Снижение HRV на 10–20% от личного базового уровня два дня "
            "подряд — сильный сигнал о недовосстановлении. Рекомендуется "
            "снизить интенсивность тренировки, заменить тяжёлую сессию на "
            "лёгкую Z1–Z2 или день отдыха, пересмотреть сон и питание."
        ),
        "metadata": {
            "category": "recovery_science",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Комбинированный recovery score объединяет сон, HRV и RHR: "
            "низкие баллы (<50) — восстановление недостаточное, планировать "
            "лёгкий день или отдых; 50–70 — средний уровень, умеренная "
            "нагрузка допустима; >70 — организм готов к интенсивной "
            "тренировке."
        ),
        "metadata": {
            "category": "recovery_science",
            "source": "public domain summary",
            "confidence": "medium",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Активное восстановление (active recovery): лёгкая аэробная "
            "нагрузка 20–40 минут в Z1 (пульс до 60–65% от HRmax) после "
            "тяжёлой сессии. Ускоряет кровоток и выведение метаболитов, "
            "снижает DOMS (отсроченную мышечную боль) и ускоряет "
            "восстановление по сравнению с полным отдыхом."
        ),
        "metadata": {
            "category": "recovery_science",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Сон — самый мощный инструмент восстановления. Во время "
            "глубокого сна (N3) вырабатывается гормон роста, происходит "
            "синтез белка и восстановление тканей. Недосып <6 часов две "
            "ночи подряд эквивалентен потере ~15% силы и аэробной "
            "производительности."
        ),
        "metadata": {
            "category": "recovery_science",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Признаки перетренированности (overtraining): рост RHR, падение "
            "HRV, ухудшение сна, снижение мотивации, стабильное падение "
            "производительности >2 недель, частые ОРВИ. Подход — разгрузка "
            "2–4 недели (снижение объёма на 50–70%), акцент на сон, "
            "питание, стресс-менеджмент."
        ),
        "metadata": {
            "category": "recovery_science",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Стратегии восстановления после высокоинтенсивной сессии: "
            "углеводы + белок в течение 30–60 минут (например, 1 г/кг "
            "углеводов и 0.3 г/кг белка), 500–1000 мл жидкости с "
            "электролитами, сон 8+ часов, лёгкая растяжка или мягкий "
            "foam rolling. Ледяные ванны эффект неоднозначен: снижают "
            "болезненность, но могут подавлять гипертрофию."
        ),
        "metadata": {
            "category": "recovery_science",
            "source": "public domain summary",
            "confidence": "medium",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    # ==================== sport_specific (6: 2 running, 2 cycling, 2 strength) ====================
    {
        "text": (
            "Бег — начинающему разумно стартовать с 3 сессий в неделю по "
            "20–30 минут в разговорном темпе, чередуя с ходьбой (например, "
            "3 мин бег / 1 мин шаг). Увеличение недельного объёма — "
            "не более +10% в неделю (правило 10%) для снижения риска "
            "травм колен, ахиллов и голеней."
        ),
        "metadata": {
            "category": "sport_specific",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "running",
            "experience_level": "beginner",
        },
    },
    {
        "text": (
            "Бег — типичная структура недели у любителя на 5–10 км: 1 "
            "длинная пробежка (Z2, 60–90 минут), 1 темповая или "
            "интервальная (Z3–Z4), 1–2 восстановительных (Z1–Z2). "
            "Стретчинг и силовые 1–2 раза в неделю снижают риск травм."
        ),
        "metadata": {
            "category": "sport_specific",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "running",
            "experience_level": "intermediate",
        },
    },
    {
        "text": (
            "Велосипед — ключевые зоны мощности по FTP (Functional "
            "Threshold Power): Z1 <55%, Z2 56–75% (базовая выносливость), "
            "Z3 76–90% (темп), Z4 91–105% (пороговые), Z5 106–120% "
            "(VO2max), Z6 >120% (анаэробные). Большая часть тренировок у "
            "любителей выносливости — Z2."
        ),
        "metadata": {
            "category": "sport_specific",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "cycling",
            "experience_level": "intermediate",
        },
    },
    {
        "text": (
            "Велосипед — новичку достаточно 3 сессий в неделю по 45–90 "
            "минут в Z2 (разговорный темп). Через 4–6 недель добавить 1 "
            "интервальную сессию (например, 4×4 минуты в Z4 с отдыхом 3 "
            "минуты). Педальная техника (каденс 80–95 об/мин) важнее "
            "большой передачи для начинающих."
        ),
        "metadata": {
            "category": "sport_specific",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "cycling",
            "experience_level": "beginner",
        },
    },
    {
        "text": (
            "Силовые тренировки — для гипертрофии мышц оптимальный диапазон "
            "6–12 повторений с 60–80% от 1ПМ, 3–5 подходов на упражнение, "
            "10–20 подходов на мышечную группу в неделю. Для силы — 1–5 "
            "повторений с 85–95% от 1ПМ, 3–5 подходов, отдых 3–5 минут "
            "между подходами."
        ),
        "metadata": {
            "category": "sport_specific",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "strength",
            "experience_level": "intermediate",
        },
    },
    {
        "text": (
            "Силовые для новичка: 2–3 тренировки в неделю на всё тело, "
            "фокус на базовых движениях (присед, жим, тяга, подтягивание, "
            "жим над головой), 2–3 подхода по 8–12 повторений с отдыхом "
            "1.5–3 минуты. Линейный прогресс веса +2.5–5 кг в неделю "
            "работает первые 2–3 месяца, далее требуется периодизация."
        ),
        "metadata": {
            "category": "sport_specific",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "strength",
            "experience_level": "beginner",
        },
    },
    # ==================== nutrition_basics (6) ====================
    {
        "text": (
            "Макронутриенты и их калорийность: углеводы и белки — по 4 "
            "ккал/г, жиры — 9 ккал/г, алкоголь — 7 ккал/г. Для активного "
            "человека типичное распределение: углеводы 45–60% калорий, "
            "белки 15–25%, жиры 25–35%. Точные пропорции зависят от целей "
            "и вида спорта."
        ),
        "metadata": {
            "category": "nutrition_basics",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Белок у тренирующихся: 1.4–2.0 г/кг массы тела в день для "
            "поддержания и роста мышц, до 2.2–2.4 г/кг при дефиците калорий "
            "(защита мышц от катаболизма). Распределение по 20–40 г в "
            "4–5 приёмах пищи эффективнее одной большой порции."
        ),
        "metadata": {
            "category": "nutrition_basics",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Углеводы для выносливости: при тренировках >60 минут — 5–7 "
            "г/кг/день, при >2 часов или в период высоких объёмов — 7–10 "
            "г/кг/день. Перед длительной сессией уместна загрузка за 1–3 "
            "часа (1–3 г/кг углеводов, низкое содержание клетчатки и "
            "жиров)."
        ),
        "metadata": {
            "category": "nutrition_basics",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "intermediate",
        },
    },
    {
        "text": (
            "Гидратация: базовая норма ~30–35 мл/кг массы тела в день в "
            "покое. При тренировке добавить 500–750 мл на каждый час "
            "нагрузки, при жаре — больше. Ориентир достаточной гидратации "
            "— светло-жёлтый цвет мочи. При сессиях >90 минут нужны "
            "электролиты (натрий 300–700 мг/л)."
        ),
        "metadata": {
            "category": "nutrition_basics",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Nutrient timing вокруг тренировки: за 2–3 часа — сбалансированный "
            "приём пищи с углеводами и умеренным белком. За 30–60 минут "
            "до — лёгкая углеводная закуска если нужно. После — в течение "
            "30–120 минут углеводы + белок (например, 1 г/кг углеводов и "
            "0.3 г/кг белка) для гликогена и синтеза мышечного белка."
        ),
        "metadata": {
            "category": "nutrition_basics",
            "source": "public domain summary",
            "confidence": "medium",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
    {
        "text": (
            "Дефицит и профицит калорий: для похудения умеренный дефицит "
            "300–500 ккал/день даёт 0.3–0.7 кг/неделю потери веса с "
            "минимальной потерей мышц (при достаточном белке и силовых). "
            "Для набора массы — профицит 200–400 ккал/день даёт 0.25–0.5 "
            "кг/неделю, меньше жира чем при агрессивном профиците."
        ),
        "metadata": {
            "category": "nutrition_basics",
            "source": "public domain summary",
            "confidence": "high",
            "sport_type": "*",
            "experience_level": "*",
        },
    },
]


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------


def _chunk_id(text: str, category: str) -> str:
    """Детерминированный id чанка (uuid5 от текста+категории)."""
    return str(uuid.uuid5(_NS, f"{category}|{text}"))


def _validate_chunks() -> None:
    """Sanity-check: все чанки заполнены и категории корректны."""
    categories_seen: set[str] = set()
    for idx, chunk in enumerate(CHUNKS):
        text = chunk.get("text", "").strip()
        meta = chunk.get("metadata", {})
        cat = meta.get("category", "")
        if not text:
            raise ValueError(f"CHUNKS[{idx}]: пустой text")
        if cat not in CATEGORIES:
            raise ValueError(
                f"CHUNKS[{idx}]: неизвестная категория {cat!r}, "
                f"ожидается одна из {sorted(CATEGORIES)}"
            )
        categories_seen.add(cat)

    missing = CATEGORIES - categories_seen
    if missing:
        raise ValueError(f"Не представлены категории: {sorted(missing)}")


async def _reset_storage() -> None:
    """Очистить ChromaDB-коллекцию и таблицу rag_chunks."""
    if vector_store.available:
        try:
            vector_store.delete(collection=COLLECTION_KNOWLEDGE_BASE)
            logger.info("ChromaDB knowledge_base очищен")
        except Exception as exc:
            logger.warning("Не удалось очистить ChromaDB: %s", exc)

    async with AsyncSessionLocal() as db:
        await db.execute(delete(RAGChunk))
        await db.commit()
    logger.info("Таблица rag_chunks очищена")


async def _existing_ids() -> set[str]:
    """Получить множество уже существующих id чанков из rag_chunks."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(RAGChunk.id))
        return {row[0] for row in result.all()}


async def seed(reset: bool = False) -> dict[str, int]:
    """Прогнать seed: эмбеддинг → ChromaDB + rag_chunks.

    Returns:
        Статистика: {"total": N, "added": M, "skipped": K}.
    """
    _validate_chunks()

    if not vector_store.available:
        vector_store.initialize()
    if not vector_store.available:
        raise RuntimeError(
            "ChromaDB недоступен — проверьте chroma_path и монтирование volume"
        )

    if reset:
        await _reset_storage()
        existing: set[str] = set()
    else:
        existing = await _existing_ids()

    to_embed: list[tuple[str, str, dict[str, Any]]] = []
    for chunk in CHUNKS:
        text = chunk["text"].strip()
        meta = dict(chunk["metadata"])
        cid = _chunk_id(text, meta["category"])
        if cid in existing:
            continue
        to_embed.append((cid, text, meta))

    if not to_embed:
        logger.info("Новых чанков нет — все %d уже проиндексированы", len(CHUNKS))
        return {"total": len(CHUNKS), "added": 0, "skipped": len(CHUNKS)}

    logger.info("Эмбеддим %d чанков через модель %s...", len(to_embed), embedding_service._model)
    texts = [t for _, t, _ in to_embed]
    embeddings = await embedding_service.embed(texts)

    ids = [cid for cid, _, _ in to_embed]
    metas = [m for _, _, m in to_embed]

    vector_store.add(
        collection=COLLECTION_KNOWLEDGE_BASE,
        ids=ids,
        embeddings=embeddings,
        metadatas=metas,
        documents=texts,
    )
    logger.info("Добавлено в ChromaDB: %d чанков", len(ids))

    async with AsyncSessionLocal() as db:
        for cid, text, meta in to_embed:
            db.add(
                RAGChunk(
                    id=cid,
                    text=text,
                    category=meta["category"],
                    source=meta.get("source", "public domain summary"),
                    confidence=meta.get("confidence", "medium"),
                    sport_type=meta.get("sport_type"),
                    experience_level=meta.get("experience_level"),
                    embedding_id=cid,
                    created_at=datetime.utcnow(),
                )
            )
        await db.commit()
    logger.info("Добавлено в rag_chunks: %d записей", len(ids))

    skipped = len(CHUNKS) - len(to_embed)
    return {"total": len(CHUNKS), "added": len(to_embed), "skipped": skipped}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Knowledge Base (ChromaDB + rag_chunks)")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Очистить существующую коллекцию и таблицу перед seed",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    stats = await seed(reset=args.reset)
    logger.info(
        "Готово. total=%d added=%d skipped=%d",
        stats["total"], stats["added"], stats["skipped"],
    )


if __name__ == "__main__":
    asyncio.run(main())
