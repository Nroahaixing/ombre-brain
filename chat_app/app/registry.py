"""Registry for the single warm Claude conversation actor.
当 claude_agent_sdk 不可用时静默降级（DeepSeek/OpenAI 模式无需 SDK）。
"""

import asyncio
import logging
from collections.abc import Callable
from time import monotonic

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import ClaudeAgentOptions
    from app.actor import ActorBusyError, ConvActor
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.info("claude_agent_sdk not installed — Claude agent mode disabled")


IDLE_TTL_SECONDS = 900
REAPER_INTERVAL_SECONDS = 15


class ConvRegistry:
    def __init__(self, project_dir: str) -> None:
        self.project_dir = project_dir
        self._actor: ConvActor | None = None
        self._lock = asyncio.Lock()
        self._reaper_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(
                self._reaper(),
                name="claude-actor-reaper",
            )

    async def stop(self) -> None:
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None
        await self.invalidate()

    async def submit(
        self,
        conv_id: str,
        prompt: str,
        options: ClaudeAgentOptions,
        fingerprint: str,
        timing_callback: Callable[[str], None] | None,
    ) -> asyncio.Queue:
        async with self._lock:
            actor = self._actor
            if actor is not None and not actor.alive:
                self._actor = None
                actor = None
            if actor is not None and actor.conv_id != conv_id:
                if actor.busy:
                    raise ActorBusyError("另一段会话仍在回复")
                await actor.close()
                self._actor = None
                actor = None
            if actor is None:
                actor = ConvActor(conv_id, self.project_dir)
                self._actor = actor
            stage = (
                "actor_warm_hit"
                if actor.is_warm_for(fingerprint)
                else "actor_cold_start"
            )
            outbox = await actor.submit(
                prompt,
                options,
                fingerprint,
                timing_callback,
            )
            if timing_callback:
                timing_callback(stage)
            return outbox

    async def assert_available(self) -> None:
        async with self._lock:
            actor = self._actor
            if actor is not None and actor.alive and actor.busy:
                raise ActorBusyError("上一条消息仍在回复")

    async def invalidate(self, conv_id: str | None = None) -> None:
        async with self._lock:
            actor = self._actor
            if actor is None or (conv_id is not None and actor.conv_id != conv_id):
                return
            await actor.close()
            if self._actor is actor:
                self._actor = None

    async def _reaper(self) -> None:
        while True:
            await asyncio.sleep(REAPER_INTERVAL_SECONDS)
            async with self._lock:
                actor = self._actor
                if (
                    actor is not None
                    and not actor.busy
                    and monotonic() - actor.last_active >= IDLE_TTL_SECONDS
                ):
                    await actor.close()
                    if self._actor is actor:
                        self._actor = None


class NoopRegistry:
    """空注册表 — SDK 不可用时的降级方案"""
    async def start(self) -> None: pass
    async def stop(self) -> None: pass
    async def invalidate(self) -> None: pass
    async def submit(self, *args, **kwargs): raise RuntimeError("Claude SDK not available — use OpenAI/DeepSeek backend")
    async def assert_available(self) -> None: pass


_registry: ConvRegistry | NoopRegistry | None = None


def configure_registry(project_dir: str):
    global _registry
    if SDK_AVAILABLE:
        _registry = ConvRegistry(project_dir)
    else:
        _registry = NoopRegistry()
    return _registry


def get_registry():
    if _registry is None:
        raise RuntimeError("Registry 尚未启动")
    return _registry

