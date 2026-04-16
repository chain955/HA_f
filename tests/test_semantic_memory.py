"""Unit-тесты для SemanticMemory v1 (Issue #25)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.semantic_memory import (
    MemoryRecord,
    SemanticMemory,
    _distance_to_score,
    semantic_memory,
)


class TestDistanceToScore:
    def test_zero_distance_is_one(self):
        assert _distance_to_score(0.0) == 1.0

    def test_max_distance_is_zero(self):
        assert _distance_to_score(2.0) == 0.0


class TestRemember:
    @pytest.mark.asyncio
    async def test_skip_when_vector_store_unavailable(self):
        sm = SemanticMemory()
        with patch("app.services.semantic_memory.vector_store") as vs:
            vs.available = False
            result = await sm.remember("u1", "r1", "q", "a")
        assert result is None

    @pytest.mark.asyncio
    async def test_skip_empty_fields(self):
        sm = SemanticMemory()
        assert await sm.remember("", "r1", "q", "a") is None
        assert await sm.remember("u1", "r1", "", "a") is None
        assert await sm.remember("u1", "r1", "q", "") is None

    @pytest.mark.asyncio
    async def test_happy_path_returns_id(self):
        sm = SemanticMemory()
        with patch("app.services.semantic_memory.vector_store") as vs, \
             patch("app.services.semantic_memory.embedding_service") as emb:
            vs.available = True
            vs.add = MagicMock()
            emb.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

            result = await sm.remember("u1", "req-1", "Как улучшить HRV?", "Ответ...")

        assert result is not None
        assert len(result) > 0
        vs.add.assert_called_once()
        call_kwargs = vs.add.call_args.kwargs
        assert "semantic_memory" in call_kwargs["collection"]
        assert call_kwargs["metadatas"][0]["user_id"] == "u1"
        assert call_kwargs["metadatas"][0]["request_id"] == "req-1"
        # Документ — объединённая строка Q: ... \n A: ...
        doc = call_kwargs["documents"][0]
        assert "Q: Как улучшить HRV?" in doc
        assert "A: Ответ..." in doc

    @pytest.mark.asyncio
    async def test_embedding_error_returns_none(self):
        sm = SemanticMemory()
        with patch("app.services.semantic_memory.vector_store") as vs, \
             patch("app.services.semantic_memory.embedding_service") as emb:
            vs.available = True
            emb.embed = AsyncMock(side_effect=RuntimeError("boom"))

            assert await sm.remember("u1", "r1", "q", "a") is None

    @pytest.mark.asyncio
    async def test_add_error_returns_none(self):
        sm = SemanticMemory()
        with patch("app.services.semantic_memory.vector_store") as vs, \
             patch("app.services.semantic_memory.embedding_service") as emb:
            vs.available = True
            vs.add = MagicMock(side_effect=RuntimeError("chroma error"))
            emb.embed = AsyncMock(return_value=[[0.1, 0.2]])

            assert await sm.remember("u1", "r1", "q", "a") is None


class TestRecall:
    FAKE_RAW = {
        "ids": [["m1", "m2", "m3"]],
        "documents": [["Q: ...\nA: ответ1", "Q: ...\nA: ответ2", "Q: ...\nA: ответ3"]],
        "metadatas": [[
            {"user_id": "u1", "timestamp": "2026-04-10T12:00:00"},
            {"user_id": "u1", "timestamp": "2026-04-11T12:00:00"},
            {"user_id": "u1", "timestamp": "2026-04-12T12:00:00"},
        ]],
        # Distances: 0.0 → score=1.0, 0.5 → 0.75, 1.4 → 0.3
        "distances": [[0.0, 0.5, 1.4]],
    }

    @pytest.mark.asyncio
    async def test_returns_empty_when_unavailable(self):
        sm = SemanticMemory()
        with patch("app.services.semantic_memory.vector_store") as vs:
            vs.available = False
            result = await sm.recall("u1", "query")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_empty_input(self):
        sm = SemanticMemory()
        assert await sm.recall("", "q") == []
        assert await sm.recall("u1", "") == []

    @pytest.mark.asyncio
    async def test_happy_path_filters_by_min_score(self):
        sm = SemanticMemory()
        with patch("app.services.semantic_memory.vector_store") as vs, \
             patch("app.services.semantic_memory.embedding_service") as emb:
            vs.available = True
            vs.query = MagicMock(return_value=self.FAKE_RAW)
            emb.embed = AsyncMock(return_value=[[0.1, 0.2]])

            result = await sm.recall("u1", "q", top_k=3, min_score=0.6)

        # 3-я запись со score 0.3 отфильтрована
        assert len(result) == 2
        assert result[0].score >= result[1].score
        assert all(isinstance(r, MemoryRecord) for r in result)

    @pytest.mark.asyncio
    async def test_recall_filters_by_user(self):
        sm = SemanticMemory()
        with patch("app.services.semantic_memory.vector_store") as vs, \
             patch("app.services.semantic_memory.embedding_service") as emb:
            vs.available = True
            vs.query = MagicMock(return_value=self.FAKE_RAW)
            emb.embed = AsyncMock(return_value=[[0.1, 0.2]])

            await sm.recall("u1", "q", top_k=3)

        call_kwargs = vs.query.call_args.kwargs
        assert call_kwargs["where"] == {"user_id": "u1"}
        assert call_kwargs["n_results"] == 3

    @pytest.mark.asyncio
    async def test_embedding_error_returns_empty(self):
        sm = SemanticMemory()
        with patch("app.services.semantic_memory.vector_store") as vs, \
             patch("app.services.semantic_memory.embedding_service") as emb:
            vs.available = True
            emb.embed = AsyncMock(side_effect=RuntimeError("boom"))

            assert await sm.recall("u1", "q") == []

    @pytest.mark.asyncio
    async def test_query_error_returns_empty(self):
        sm = SemanticMemory()
        with patch("app.services.semantic_memory.vector_store") as vs, \
             patch("app.services.semantic_memory.embedding_service") as emb:
            vs.available = True
            emb.embed = AsyncMock(return_value=[[0.1]])
            vs.query = MagicMock(side_effect=RuntimeError("bad"))

            assert await sm.recall("u1", "q") == []


class TestListAndClear:
    def test_list_records_when_unavailable(self):
        sm = SemanticMemory()
        with patch("app.services.semantic_memory.vector_store") as vs:
            vs.available = False
            assert sm.list_records() == []

    def test_list_records_happy_path(self):
        sm = SemanticMemory()
        fake_coll = MagicMock()
        fake_coll.get = MagicMock(return_value={
            "ids": ["m1", "m2"],
            "documents": ["Q: a\nA: b", "Q: c\nA: d"],
            "metadatas": [
                {"user_id": "u1", "timestamp": "2026-04-10"},
                {"user_id": "u1", "timestamp": "2026-04-11"},
            ],
        })
        with patch("app.services.semantic_memory.vector_store") as vs:
            vs.available = True
            vs._get_collection = MagicMock(return_value=fake_coll)

            records = sm.list_records(user_id="u1", limit=10)

        assert len(records) == 2
        assert records[0].user_id == "u1"
        assert records[0].score == 1.0
        fake_coll.get.assert_called_once_with(limit=10, where={"user_id": "u1"})

    def test_clear_user_records(self):
        sm = SemanticMemory()
        fake_coll = MagicMock()
        fake_coll.get = MagicMock(return_value={"ids": ["m1", "m2"]})
        fake_coll.delete = MagicMock()
        with patch("app.services.semantic_memory.vector_store") as vs:
            vs.available = True
            vs._get_collection = MagicMock(return_value=fake_coll)

            deleted = sm.clear(user_id="u1")

        assert deleted == 2
        fake_coll.delete.assert_called_once_with(ids=["m1", "m2"])

    def test_clear_all_when_empty(self):
        sm = SemanticMemory()
        fake_coll = MagicMock()
        fake_coll.get = MagicMock(return_value={"ids": []})
        fake_coll.delete = MagicMock()
        with patch("app.services.semantic_memory.vector_store") as vs:
            vs.available = True
            vs._get_collection = MagicMock(return_value=fake_coll)

            deleted = sm.clear()

        assert deleted == 0
        fake_coll.delete.assert_not_called()

    def test_clear_all_records(self):
        sm = SemanticMemory()
        fake_coll = MagicMock()
        fake_coll.get = MagicMock(return_value={"ids": ["m1", "m2", "m3"]})
        fake_coll.delete = MagicMock()
        with patch("app.services.semantic_memory.vector_store") as vs:
            vs.available = True
            vs._get_collection = MagicMock(return_value=fake_coll)

            deleted = sm.clear()

        assert deleted == 3
        fake_coll.delete.assert_called_once_with(ids=["m1", "m2", "m3"])


def test_singleton_exists():
    """Глобальный синглтон semantic_memory доступен для импорта."""
    assert isinstance(semantic_memory, SemanticMemory)
