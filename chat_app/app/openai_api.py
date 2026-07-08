"""
OpenAI direct API backend — simplest chat backend when no Anthropic key.
Uses gpt-4o-mini by default (cheap and fast).
"""

import json
import logging
import os
from collections.abc import AsyncGenerator, Callable
from uuid import uuid4

from app.memory import build_profile_context, read_memory
from app.memory_manager import retrieve as retrieve_memory
from app.store import conversation_messages

logger = logging.getLogger(__name__)


async def stream_openai_chat(
    message: str,
    conv_id: str,
    session_id: str | None = None,
    model: str = "gpt-4o-mini",
    effort: str = "medium",
    extended: bool = True,
    timing_callback: Callable[[str], None] | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream chat via OpenAI API."""

    # Build system prompt with memory
    profile_context = build_profile_context().strip()
    memory_file = "" if profile_context else read_memory().strip()

    system = "你是一个友好的 AI 助手，使用中文回复。你叫小克。"
    if profile_context:
        system += f"\n\n<profile>\n{profile_context}\n</profile>"
    if memory_file:
        system += f"\n\n<记忆>\n{memory_file}\n</记忆>"

    import asyncio
    memory_hits = await asyncio.to_thread(retrieve_memory, message)
    if memory_hits:
        system += f"\n\n<长期记忆>\n{memory_hits}\n</长期记忆>"

    # Build messages from conversation history
    messages = [{"role": "system", "content": system}]
    try:
        rows, _, _ = conversation_messages(conv_id)
        for msg in rows[-20:]:
            role = msg["role"]
            text = msg.get("text") or ""
            if text.strip():
                messages.append({"role": role, "content": text[:4000]})
    except Exception:
        pass

    messages.append({"role": "user", "content": message})

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未配置")

    # Call OpenAI API
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"OpenAI API 错误 {resp.status}: {error[:200]}")

            full_text = ""
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text += content
                            yield {"event": "delta", "text": content}
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    yield {"event": "done", "session_id": f"openai-{uuid4().hex[:12]}"}
    logger.info("openai_chat_done conv=%s len=%d", conv_id[:8], len(full_text))
