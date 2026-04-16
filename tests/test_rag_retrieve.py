"""Unit-тесты для tool rag_retrieve (Issue #24).

Используют mock EmbeddingService и mock VectorStore.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.rag_retrieve import (
    RagChunk,
    _build_where_filter,
    _distance_to_score,
    _parse_query_result,
    rag_retrieve,
)


# --------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------- #


class TestDistanceToScore:
    def test_zero_distance_is_one(self):
        assert _distance_to_score(0.0) == 1.0

    def test_max_distance_is_zero(self):
        assert _distance_to_score(2.0) == 0.0

    def test_half_distance(self):
        assert _distance_to_score(1.0) == 0.5

    def test_clipped_at_zero(self):
        """Distance > 2 → score 0 (не отрицательный)."""
        assert _distance_to_score(10.0) == 0.0


class TestBuildWhereFilter:
    def test_no_filters(self):
        assert _build_where_filter(None, None) is None

    def test_category_only(self):
        assert _build_where_filter("physiology_norms", None) == {
            "category": "physiology_norms"
        }

    def test_sport_star_skipped(self):
        """sport_type='*' не добавляется в фильтр."""
        assert _build_where_filter(None, "*") is None

    def test_sport_type_uses_in(self):
        """sport_type фильтр включает '*' для универсальных чанков."""
        f = _build_where_filter(None, "running")
        assert f == {"sport_type": {"$in": ["running", "*"]}}

    def test_both_combined_with_and(self):
        f = _build_where_filter("sport_specific", "running")
        assert f == {
            "$and": [
                {"category": "sport_specific"},
                {"sport_type": {"$in": ["running", "*"]}},
            ]
        }


class TestParseQueryResult:
    def test_parses_full_result(self):
        raw = {
            "ids": [["id1", "id2"]],
            "documents": [["текст 1", "текст 2"]],
            "metadatas": [[
                {
                    "category": "physiology_norms",
                    "source": "test",
                    "confidence": "high",
                    "sport_type": "*",
                },
                {
                    "category": "training_principles",
                    "source": "test",
                    "confidence": "medium",
                },
            ]],
            "distances": [[0.0, 1.0]],
        }
        chunks = _parse_query_result(raw)
        assert len(chunks) == 2
        assert chunks[0].text == "текст 1"
        assert chunks[0].score == 1.0
        assert chunks[1].score == 0.5
        assert chunks[0].category == "physiology_norms"
        assert chunks[1].confidence == "medium"

    def test_empty_result(self):
        raw = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        assert _parse_query_result(raw) == []

    def test_missing_fields_fallback(self):
        """Если какие-то поля не пришли — score/category defaults работают."""
        raw = {
            "ids": [["id1"]],
            "documents": [["t"]],
            "metadatas": [[{}]],
            "distances": [[0.0]],
        }
        chunks = _parse_query_result(raw)
        assert chunks[0].category == ""
        assert chunks[0].confidence == "medium"


# --------------------------------------------------------------------- #
# rag_retrieve — с mock-ами
# --------------------------------------------------------------------- #


@pytest.fixture
def fake_raw_result():
    return {
        "ids": [["id1"]],
        "documents": [["HRV — вариабельность сердечного ритма..."]],
        "metadatas": [[{
            "category": "physiology_norms",
            "source": "test",
            "confidence": "high",
            "sport_type": "*",
        }]],
        "distances": [[0.2]],
    }


class TestRagRetrieve:
    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self):
        result = await rag_retrieve(query="")
        assert result.success is False
        assert result.data is None
        assert "Пустой query" in (result.error or "")

    @pytest.mark.asyncio
    async def test_unavailable_store_returns_empty(self):
        """Если ChromaDB недоступен — success=True, data=[]."""
        with patch("app.tools.rag_retrieve.vector_store") as vs:
            vs.available = False
            result = await rag_retrieve(query="что такое HRV")
        assert result.success is True
        assert result.data == []

    @pytest.mark.asyncio
    async def test_happy_path(self, fake_raw_result):
        with patch("app.tools.rag_retrieve.vector_store") as vs, \
             patch("app.tools.rag_retrieve.embedding_service") as emb:
            vs.available = True
            vs.query = MagicMock(return_value=fake_raw_result)
            emb.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

            result = await rag_retrieve(query="что такое HRV", top_k=3)

        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 1
        assert result.data[0]["category"] == "physiology_norms"
        assert result.data[0]["score"] > 0.8

    @pytest.mark.asyncio
    async def test_embedding_error_returns_error(self):
        with patch("app.tools.rag_retrieve.vector_store") as vs, \
             patch("app.tools.rag_retrieve.embedding_service") as emb:
            vs.available = True
            emb.embed = AsyncMock(side_effect=RuntimeError("boom"))

            result = await rag_retrieve(query="test")

        assert result.success is False
        assert "embedding" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_vector_store_error_returns_error(self):
        with patch("app.tools.rag_retrieve.vector_store") as vs, \
             patch("app.tools.rag_retrieve.embedding_service") as emb:
            vs.available = True
            emb.embed = AsyncMock(return_value=[[0.1, 0.2]])
            vs.query = MagicMock(side_effect=RuntimeError("chroma error"))

            result = await rag_retrieve(query="test")

        assert result.success is False
        assert "vector store" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_category_filter_passed_to_query(self, fake_raw_result):
        with patch("app.tools.rag_retrieve.vector_store") as vs, \
             patch("app.tools.rag_retrieve.embedding_service") as emb:
            vs.available = True
            vs.query = MagicMock(return_value=fake_raw_result)
            emb.embed = AsyncMock(return_value=[[0.1, 0.2]])

            await rag_retrieve(
                query="test",
                category="physiology_norms",
                sport_type="running",
                top_k=2,
            )

        call_kwargs = vs.query.call_args.kwargs
        assert call_kwargs["n_results"] == 2
        where = call_kwargs["where"]
        assert "$and" in where
        assert {"category": "physiology_norms"} in where["$and"]

    @pytest.mark.asyncio
    async def test_top_k_forwarded(self, fake_raw_result):
        with patch("app.tools.rag_retrieve.vector_store") as vs, \
             patch("app.tools.rag_retrieve.embedding_service") as emb:
            vs.available = True
            vs.query = MagicMock(return_value=fake_raw_result)
            emb.embed = AsyncMock(return_value=[[0.1, 0.2]])

            await rag_retrieve(query="q", top_k=7)

        assert vs.query.call_args.kwargs["n_results"] == 7


class TestRagChunk:
    def test_to_dict(self):
        c = RagChunk(
            text="x",
            category="physiology_norms",
            source="s",
            confidence="high",
            score=0.9,
        )
        d = c.to_dict()
        assert d["text"] == "x"
        assert d["score"] == 0.9
        assert d["sport_type"] is None
