"""Тесты для SlotState и slot_state_from_entities."""

from datetime import date, timedelta

from app.pipeline.slot_state import SlotState, slot_state_from_entities
from app.tools.schemas import (
    AnalysisType,
    BodyPartEnum,
    IntensityEnum,
    MetricEnum,
    SportTypeEnum,
)


class TestSlotStateFromEntities:
    def test_empty_entities_returns_empty_slots(self) -> None:
        slots = slot_state_from_entities({})
        assert slots.time_range is None
        assert slots.sport_types == []
        assert slots.metrics == []
        assert slots.analysis_type == AnalysisType.NONE

    def test_time_range_label_resolved(self) -> None:
        slots = slot_state_from_entities({"time_range": "за неделю"})
        today = date.today()
        assert slots.time_range is not None
        assert slots.time_range.date_to == today
        assert slots.time_range.date_from == today - timedelta(days=6)
        assert slots.time_range.label == "за неделю"

    def test_sport_type_coerced(self) -> None:
        slots = slot_state_from_entities({"sport_type": "running"})
        assert slots.sport_types == [SportTypeEnum.RUNNING]
        assert slots.sport_type == SportTypeEnum.RUNNING

    def test_metric_coerced(self) -> None:
        slots = slot_state_from_entities({"metric": "heart_rate"})
        assert slots.metrics == [MetricEnum.HEART_RATE]

    def test_list_of_metrics(self) -> None:
        slots = slot_state_from_entities({"metrics": ["heart_rate", "steps"]})
        assert MetricEnum.HEART_RATE in slots.metrics
        assert MetricEnum.STEPS in slots.metrics

    def test_body_part_coerced(self) -> None:
        slots = slot_state_from_entities({"body_part": "колено"})
        assert slots.body_parts == [BodyPartEnum.KNEE]

    def test_intensity_coerced(self) -> None:
        slots = slot_state_from_entities({"intensity": "тяжело"})
        assert slots.intensity == IntensityEnum.HARD

    def test_unknown_sport_type_silently_dropped(self) -> None:
        """Неизвестное значение enum не должно ронять pipeline."""
        slots = slot_state_from_entities({"sport_type": "квиддич"})
        assert slots.sport_types == []

    def test_unknown_metric_silently_dropped(self) -> None:
        slots = slot_state_from_entities({"metric": "unknown_metric_xyz"})
        assert slots.metrics == []

    def test_mix_of_valid_and_invalid(self) -> None:
        slots = slot_state_from_entities(
            {"metrics": ["heart_rate", "unknown", "steps"]}
        )
        assert MetricEnum.HEART_RATE in slots.metrics
        assert MetricEnum.STEPS in slots.metrics
        assert len(slots.metrics) == 2

    def test_raw_query_preserved(self) -> None:
        slots = slot_state_from_entities({}, raw_query="Сколько шагов за неделю")
        assert slots.raw_query == "Сколько шагов за неделю"


class TestSlotStateAccessors:
    def test_sport_type_none_when_empty(self) -> None:
        slots = SlotState()
        assert slots.sport_type is None

    def test_sport_type_returns_first(self) -> None:
        slots = SlotState(sport_types=[SportTypeEnum.RUNNING, SportTypeEnum.SWIMMING])
        assert slots.sport_type == SportTypeEnum.RUNNING

    def test_missing_reports_empty_slots(self) -> None:
        slots = SlotState()
        missing = slots.missing(["time_range", "sport_types", "metrics"])
        assert set(missing) == {"time_range", "sport_types", "metrics"}

    def test_missing_skips_filled_slots(self) -> None:
        slots = SlotState(sport_types=[SportTypeEnum.RUNNING])
        missing = slots.missing(["sport_types", "metrics"])
        assert missing == ["metrics"]

    def test_to_entities_dict_roundtrip(self) -> None:
        original = {
            "time_range": "за неделю",
            "sport_type": "running",
            "metric": "steps",
            "intensity": "тяжело",
        }
        slots = slot_state_from_entities(original)
        rt = slots.to_entities_dict()
        assert rt["time_range"] == "за неделю"
        assert rt["sport_type"] == "running"
        assert rt["metric"] == "steps"
        assert rt["intensity"] == "тяжело"
