from __future__ import annotations
import asyncio
from typing import Awaitable, Callable, Dict, List, Any, Tuple

Callback = Callable[[str, Dict[str, Any]], Awaitable[None]]


class Bus:
    def __init__(self) -> None:
        self._subs: Dict[str, List[Callback]] = {}
        self._queue: "asyncio.Queue[Tuple[str, Dict[str, Any]]]" = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def subscribe(self, topic: str, cb: Callback) -> None:
        self._subs.setdefault(topic, []).append(cb)

    async def publish(self, topic: str, event: Dict[str, Any]) -> None:
        await self._queue.put((topic, event))

    async def _run(self) -> None:
        while True:
            topic, event = await self._queue.get()
            for cb in self._subs.get(topic, []):
                try:
                    await cb(topic, event)
                except Exception:
                    # swallow but could log
                    pass

    def start(self) -> None:
        if not self._task:
            self._task = asyncio.create_task(self._run())

