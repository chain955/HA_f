"""Semantic Memory v1 — сохранение и retrieval эмбеддингов Q/A (Issue #25).

Минимальная реализация:
  * remember(user_id, request_id, query, response)
      эмбеддит строку "Q: {query}\nA: {response}" и пишет в коллекцию
      `semantic_memory` с metadata {user_id, request_id, timestamp}.
  * recall(user_id, query, top_k=3, min_score=0.6)
      эмбеддит query, делает vector_store.query с фильтром по user_id,
      возвращает список {text, score, timestamp} с score >= min_score.
  * list_records / clear — для админского API (#35).

TODO v2:
  * TTL: сколько хранить записи (по умолчанию — вечно).
  * Категоризация (intent, sport_type) в metadata для фильтрации.
  * Importance-score: приоритизировать важные Q/A.
  * Дедупликация похожих вопросов (косинусная близость > 0.95 → обновление).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.services.embedding_service import embedding_service
from app.services.vector_store import COLLECTION_SEMANTIC_MEMORY, vector_store

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    """Одна запись semantic memory."""

    id: str
    text: str
    score: float
    timestamp: str
    user_id: str
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "request_id": self.request_id,
        }


def _distance_to_score(distance: float) -> float:
    """Cosine distance → score (0..1). Совпадает с логикой rag_retrieve."""
    return max(0.0, 1.0 - float(distance) / 2.0)


class SemanticMemory:
    """Сервис для работы с коллекцией semantic_memory.

    Graceful: при недоступности ChromaDB методы не бросают, а логируют и
    возвращают пустые результаты / skip, чтобы не ломать основной pipeline.
    """

    DEFAULT_TOP_K = 3
    DEFAULT_MIN_SCORE = 0.6

    async def remember(
        self,
        user_id: str,
        request_id: str | None,
        query: str,
        response: str,
    ) -> str | None:
        """Сохранить Q/A пару в semantic memory.

        Args:
            user_id: Идентификатор пользователя.
            request_id: Идентификатор запроса (PipelineLog.id), опционально.
            query: Текст пользовательского запроса.
            response: Текст ответа ассистента.

        Returns:
            id созданной записи или None если запись не удалось создать.
        """
        if not user_id or not query or not response:
            logger.debug("SemanticMemory.remember: пропуск — пустые поля")
            return None

        if not vector_store.available:
            logger.debug("SemanticMemory.remember: ChromaDB недоступен, пропуск")
            return None

        combined = f"Q: {query}\nA: {response}"
        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            embeddings = await embedding_service.embed(combined)
            if not embeddings or not embeddings[0]:
                raise ValueError("EmbeddingService вернул пустой вектор")
            embedding = embeddings[0]
        except Exception as exc:
            logger.warning("SemanticMemory.remember: ошибка embedding: %s", exc)
            return None

        metadata: dict[str, Any] = {
            "user_id": user_id,
            "timestamp": timestamp,
        }
        if request_id:
            metadata["request_id"] = request_id

        try:
            vector_store.add(
                collection=COLLECTION_SEMANTIC_MEMORY,
                ids=[record_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[combined],
            )
        except Exception as exc:
            logger.warning("SemanticMemory.remember: ошибка ChromaDB add: %s", exc)
            return None

        logger.info(
            "SemanticMemory.remember: user=%s request=%s id=%s",
            user_id, request_id, record_id,
        )
        return record_id

    async def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[MemoryRecord]:
        """Найти релевантные прошлые Q/A для пользователя.

        Args:
            user_id: Идентификатор пользователя.
            query: Текущий запрос (для embedding-поиска).
            top_k: Максимум результатов.
            min_score: Минимальный score (0..1), ниже — отбрасываем.

        Returns:
            Список MemoryRecord с score >= min_score, отсортированный по score desc.
            Пустой список при недоступности ChromaDB / ошибке / отсутствии совпадений.
        """
        if not user_id or not query or not query.strip():
            return []
        if not vector_store.available:
            return []

        try:
            embeddings = await embedding_service.embed(query)
            if not embeddings or not embeddings[0]:
                return []
            query_embedding = embeddings[0]
        except Exception as exc:
            logger.warning("SemanticMemory.recall: ошибка embedding: %s", exc)
            return []

        try:
            raw = vector_store.query(
                collection=COLLECTION_SEMANTIC_MEMORY,
                query_embedding=query_embedding,
                n_results=top_k,
                where={"user_id": user_id},
            )
        except Exception as exc:
            logger.warning("SemanticMemory.recall: ошибка ChromaDB query: %s", exc)
            return []

        records = self._parse_records(raw, min_score=min_score)
        logger.info(
            "SemanticMemory.recall: user=%s top_k=%d → %d записей",
            user_id, top_k, len(records),
        )
        return records

    def _parse_records(
        self,
        raw: dict[str, Any],
        min_score: float,
    ) -> list[MemoryRecord]:
        """Собрать MemoryRecord из query-результата и отфильтровать по score."""
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        records: list[MemoryRecord] = []
        for idx, rid in enumerate(ids):
            text = docs[idx] if idx < len(docs) else ""
            meta = metas[idx] if idx < len(metas) else {}
            distance = distances[idx] if idx < len(distances) else 2.0
            score = round(_distance_to_score(distance), 4)
            if score < min_score:
                continue
            records.append(
                MemoryRecord(
                    id=str(rid),
                    text=text,
                    score=score,
                    timestamp=str(meta.get("timestamp", "")),
                    user_id=str(meta.get("user_id", "")),
                    request_id=meta.get("request_id"),
                )
            )
        records.sort(key=lambda r: r.score, reverse=True)
        return records

    # ------------------------------------------------------------------
    # Админские операции (GET / DELETE)
    # ------------------------------------------------------------------

    def list_records(
        self,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        """Вернуть записи semantic_memory (без embedding-поиска).

        Args:
            user_id: Если задан — фильтр по пользователю, иначе все.
            limit: Максимум записей.

        Returns:
            Список MemoryRecord (score=1.0 как плейсхолдер, поиск не выполняется).
        """
        if not vector_store.available:
            return []

        coll = vector_store._get_collection(COLLECTION_SEMANTIC_MEMORY)
        kwargs: dict[str, Any] = {"limit": limit}
        if user_id:
            kwargs["where"] = {"user_id": user_id}
        try:
            raw = coll.get(**kwargs)
        except Exception as exc:
            logger.warning("SemanticMemory.list_records: ошибка get: %s", exc)
            return []

        ids = raw.get("ids", [])
        docs = raw.get("documents", [])
        metas = raw.get("metadatas", [])

        records: list[MemoryRecord] = []
        for idx, rid in enumerate(ids):
            text = docs[idx] if idx < len(docs) else ""
            meta = metas[idx] if idx < len(metas) else {}
            records.append(
                MemoryRecord(
                    id=str(rid),
                    text=text,
                    score=1.0,
                    timestamp=str(meta.get("timestamp", "")),
                    user_id=str(meta.get("user_id", "")),
                    request_id=meta.get("request_id"),
                )
            )
        return records

    def clear(self, user_id: str | None = None) -> int:
        """Удалить записи semantic_memory.

        Args:
            user_id: Если задан — только записи этого пользователя, иначе все.

        Returns:
            Количество удалённых записей.
        """
        if not vector_store.available:
            return 0

        coll = vector_store._get_collection(COLLECTION_SEMANTIC_MEMORY)
        try:
            if user_id:
                existing = coll.get(where={"user_id": user_id})
                ids = existing.get("ids", [])
                if not ids:
                    return 0
                coll.delete(ids=ids)
                logger.info("SemanticMemory.clear: user=%s удалено=%d", user_id, len(ids))
                return len(ids)
            # Очистить всё
            existing_all = coll.get()
            all_ids = existing_all.get("ids", [])
            if not all_ids:
                return 0
            coll.delete(ids=all_ids)
            logger.info("SemanticMemory.clear: удалено всего=%d", len(all_ids))
            return len(all_ids)
        except Exception as exc:
            logger.warning("SemanticMemory.clear: ошибка: %s", exc)
            return 0


# Глобальный синглтон
semantic_memory = SemanticMemory()
