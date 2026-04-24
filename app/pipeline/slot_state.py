"""SlotState — типизированное нормализованное состояние слотов запроса.

IntentResult параллельно отдаёт:
— entities: dict — «сырые» строковые значения (обратная совместимость со старым
  роутером и кодом, который уже читает dict);
— slots: SlotState — типизированные, валидированные, с заполненным TimeRange.

SlotState — источник правды для tool_executor при сборке Pydantic-args.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.tools.schemas import (
    AnalysisType,
    BodyPartEnum,
    IntensityEnum,
    MetricEnum,
    SportTypeEnum,
    TimeRange,
)
from app.tools.time_utils import build_time_range

logger = logging.getLogger(__name__)


class SlotState(BaseModel):
    """Нормализованное состояние слотов пользовательского запроса.

    Все поля опциональны: intent detection заполняет только то, что смогло
    извлечь. Пустые поля идут в missing_slots для механизма clarification
    """

    time_range: TimeRange | None = None
    sport_types: list[SportTypeEnum] = Field(default_factory=list)
    metrics: list[MetricEnum] = Field(default_factory=list)
    body_parts: list[BodyPartEnum] = Field(default_factory=list)
    intensity: IntensityEnum | None = None
    analysis_type: AnalysisType = AnalysisType.NONE
    raw_query: str = ""

    model_config = {"extra": "forbid"}

    # --- удобные аксессоры для tool_executor (чаще нужен один скаляр) ---

    @property
    def sport_type(self) -> SportTypeEnum | None:
        """Первый sport_type, если есть (legacy single-value путь)."""
        return self.sport_types[0] if self.sport_types else None

    @property
    def metric(self) -> MetricEnum | None:
        """Первая metric, если есть (legacy single-value путь)."""
        return self.metrics[0] if self.metrics else None

    @property
    def body_part(self) -> BodyPartEnum | None:
        return self.body_parts[0] if self.body_parts else None

    def missing(self, required: list[str]) -> list[str]:
        """Список требуемых слотов, которые не заполнены.

        Используется для clarification loop.
        """
        missing: list[str] = []
        for name in required:
            value = getattr(self, name, None)
            if value is None or value == [] or value == "":
                missing.append(name)
        return missing

    def to_entities_dict(self) -> dict[str, Any]:
        """Снапшот в формате старого entities dict — для обратной совместимости.

        Router и template_plan_executor пока читают entities dict; не ломаем
        их, пока не перейдут на SlotState целиком.
        """
        out: dict[str, Any] = {}
        if self.time_range is not None and self.time_range.label:
            out["time_range"] = self.time_range.label
        if self.sport_type is not None:
            out["sport_type"] = self.sport_type.value
        if self.metric is not None:
            out["metric"] = self.metric.value
        if self.body_part is not None:
            out["body_part"] = self.body_part.value
        if self.intensity is not None:
            out["intensity"] = self.intensity.value
        return out


def slot_state_from_entities(entities: dict[str, Any], raw_query: str = "") -> SlotState:
    """Построить SlotState из legacy entities dict.

    Неизвестные значения enum'ов игнорируются с WARN-логом — вместо того чтобы
    ронять пайплайн. Это защищает от ситуации, когда LLM stage 2 вернул
    произвольное строковое значение (напр. новый спорт), а tool_executor
    ждёт строго enum.
    """
    data: dict[str, Any] = {"raw_query": raw_query}

    # time_range: entities["time_range"] — строка-label; резолвим в TimeRange
    label = entities.get("time_range")
    if label:
        tr = build_time_range(str(label))
        if tr is not None:
            data["time_range"] = tr

    # list-поля — принимаем и одиночное значение (legacy) и список
    data["sport_types"] = _coerce_enum_list(
        entities.get("sport_types") or entities.get("sport_type"),
        SportTypeEnum,
        field_name="sport_type",
    )
    data["metrics"] = _coerce_enum_list(
        entities.get("metrics") or entities.get("metric"),
        MetricEnum,
        field_name="metric",
    )
    data["body_parts"] = _coerce_enum_list(
        entities.get("body_parts") or entities.get("body_part"),
        BodyPartEnum,
        field_name="body_part",
    )

    # скалярные enum'ы
    intensity_raw = entities.get("intensity")
    if intensity_raw:
        try:
            data["intensity"] = IntensityEnum(intensity_raw)
        except ValueError:
            logger.warning("SlotState: неизвестное значение intensity=%r, игнор", intensity_raw)

    analysis_raw = entities.get("analysis_type")
    if analysis_raw:
        try:
            data["analysis_type"] = AnalysisType(analysis_raw)
        except ValueError:
            logger.warning(
                "SlotState: неизвестное значение analysis_type=%r, игнор", analysis_raw
            )

    try:
        return SlotState.model_validate(data)
    except ValidationError as exc:
        # Падение здесь недопустимо — SlotState строится в hot path. Возвращаем
        # пустой, если что-то совсем разъехалось, и логируем для отладки.
        logger.warning("SlotState.model_validate упал: %s. Возвращаем пустой.", exc)
        return SlotState(raw_query=raw_query)


def _coerce_enum_list(
    value: Any, enum_cls: type, field_name: str
) -> list:
    """Привести значение к списку enum-членов.

    - None / "" → []
    - строка → [enum(value)] если валидно, иначе []
    - list[str] → [enum(v) for v in list if valid]
    """
    if value is None or value == "":
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        logger.warning(
            "SlotState: неожиданный тип для %s: %r (ожидался str|list)", field_name, type(value)
        )
        return []

    out = []
    for item in items:
        if not item:
            continue
        try:
            out.append(enum_cls(item))
        except ValueError:
            logger.warning(
                "SlotState: неизвестное значение %s=%r, игнор", field_name, item
            )
    return out
