"""Tool rag_retrieve — semantic search по Knowledge Base (Phase 2, Issue #24).

Эмбеддит query через EmbeddingService и делает ChromaDB query по коллекции
`knowledge_base` с опциональной фильтрацией по metadata (category, sport_type).

Возвращает список RagChunk (text, category, source, confidence, score).
При отсутствии ChromaDB / ошибках — возвращает пустой список с error.

Регистрируется в ToolExecutor (см. app/pipeline/tool_executor.py).
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

from app.services.embedding_service import embedding_service
from app.services.vector_store import COLLECTION_KNOWLEDGE_BASE, vector_store
from app.tools.db_tools import ToolResult

logger = logging.getLogger(__name__)


@dataclass
class RagChunk:
    """Один найденный чанк знаний."""

    text: str
    category: str
    source: str
    confidence: str
    score: float
    # Опциональные поля из metadata
    sport_type: str | None = None
    experience_level: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_where_filter(
    category: str | None,
    sport_type: str | None,
) -> dict[str, Any] | None:
    """Построить where-фильтр ChromaDB по metadata.

    Если задано несколько полей — оборачиваем в $and (ChromaDB v0.4+ требует
    явной комбинации фильтров).
    """
    conditions: list[dict[str, Any]] = []
    if category:
        conditions.append({"category": category})
    if sport_type:
        # sport_type='*' означает "подходит для любого спорта" — не фильтруем
        if sport_type != "*":
            conditions.append({"sport_type": {"$in": [sport_type, "*"]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _distance_to_score(distance: float) -> float:
    """Конвертировать ChromaDB distance (cosine, 0..2) в score (0..1).

    Чем меньше distance — тем выше score. Для cosine distance 0 — полное совпадение.
    """
    return max(0.0, 1.0 - float(distance) / 2.0)


async def rag_retrieve(
    query: str,
    category: str | None = None,
    sport_type: str | None = None,
    top_k: int = 5,
) -> ToolResult:
    """Найти релевантные чанки Knowledge Base через semantic search.

    Args:
        query: Текст запроса.
        category: Опциональный фильтр по категории
                  (physiology_norms / training_principles / recovery_science /
                   sport_specific / nutrition_basics).
        sport_type: Опциональный фильтр по виду спорта (running, cycling, ...).
                    Чанки с sport_type='*' попадают в выдачу при любом фильтре.
        top_k: Максимальное количество результатов.

    Returns:
        ToolResult с data=list[dict] (RagChunk.to_dict()).
        При отсутствии ChromaDB — success=True, data=[].
    """
    if not query or not query.strip():
        return ToolResult(
            tool_name="rag_retrieve",
            success=False,
            data=None,
            error="Пустой query",
        )

    if not vector_store.available:
        logger.warning("rag_retrieve: ChromaDB недоступен, возвращаем пустой результат")
        return ToolResult(
            tool_name="rag_retrieve",
            success=True,
            data=[],
            error=None,
        )

    try:
        embeddings = await embedding_service.embed(query)
        if not embeddings or not embeddings[0]:
            raise ValueError("EmbeddingService вернул пустой вектор")
        query_embedding = embeddings[0]
    except Exception as exc:
        logger.error("rag_retrieve: ошибка embedding: %s", exc)
        return ToolResult(
            tool_name="rag_retrieve",
            success=False,
            data=None,
            error=f"embedding error: {exc}",
        )

    where = _build_where_filter(category=category, sport_type=sport_type)

    try:
        raw = vector_store.query(
            collection=COLLECTION_KNOWLEDGE_BASE,
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )
    except Exception as exc:
        logger.error("rag_retrieve: ошибка ChromaDB query: %s", exc)
        return ToolResult(
            tool_name="rag_retrieve",
            success=False,
            data=None,
            error=f"vector store error: {exc}",
        )

    chunks = _parse_query_result(raw)

    logger.info(
        "rag_retrieve: query_len=%d category=%s sport=%s → %d чанков",
        len(query), category, sport_type, len(chunks),
    )
    return ToolResult(
        tool_name="rag_retrieve",
        success=True,
        data=[c.to_dict() for c in chunks],
        error=None,
    )


def _parse_query_result(raw: dict[str, Any]) -> list[RagChunk]:
    """Разобрать результат vector_store.query в список RagChunk."""
    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    chunks: list[RagChunk] = []
    for idx, _chunk_id in enumerate(ids):
        text = docs[idx] if idx < len(docs) else ""
        meta = metas[idx] if idx < len(metas) else {}
        distance = distances[idx] if idx < len(distances) else 2.0
        chunks.append(
            RagChunk(
                text=text,
                category=str(meta.get("category", "")),
                source=str(meta.get("source", "")),
                confidence=str(meta.get("confidence", "medium")),
                score=round(_distance_to_score(distance), 4),
                sport_type=meta.get("sport_type"),
                experience_level=meta.get("experience_level"),
            )
        )
    return chunks
