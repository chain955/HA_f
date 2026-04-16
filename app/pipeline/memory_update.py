"""Memory Update — асинхронное обновление памяти после обработки запроса.

Scope v1 (Issue #25, stub в этой итерации):
  * semantic memory: сохраняется эмбеддинг пары (query, response) через
    semantic_memory.remember.

TODO (Issue #32, полная версия Memory Update):
  * short-term: обновление history сессии (сейчас делается
    orchestrator._save_messages — перенести сюда).
  * long-term: rule-based extraction фактов из диалога в профиль
    (новые injuries, цели, preferred_sports).
  * async выполнение, не блокирующее response delivery.
"""

from __future__ import annotations

import logging

from app.services.semantic_memory import semantic_memory

logger = logging.getLogger(__name__)


class MemoryUpdater:
    """Стаб v1 — обновляет только semantic memory.

    Вызывается из orchestrator после генерации ответа. Исключения
    поглощаются (memory_update не должен ломать pipeline).
    """

    async def update(
        self,
        user_id: str,
        request_id: str | None,
        query: str,
        response: str,
    ) -> None:
        """Сохранить Q/A в semantic memory.

        Args:
            user_id: Идентификатор пользователя.
            request_id: Идентификатор запроса (PipelineLog.id).
            query: Исходный запрос пользователя.
            response: Ответ ассистента.
        """
        try:
            await semantic_memory.remember(
                user_id=user_id,
                request_id=request_id,
                query=query,
                response=response,
            )
        except Exception as exc:
            logger.warning("MemoryUpdater.update: ошибка: %s", exc)


memory_updater = MemoryUpdater()
