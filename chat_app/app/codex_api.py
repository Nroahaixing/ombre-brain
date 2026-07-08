"""Codex CLI adapter for the local chat stream."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from app.claude import SYSTEM_PROMPT
from app.memory import build_profile_context, read_memory
from app.memory_manager import retrieve as retrieve_memory
from app.store import ConversationNotFound, conversation_messages

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent  # chat_app/app/* → repo root
PROJECT_DIR = str(ROOT)
CODEX_BIN = Path("/home/ggcuser/.local/bin/codex")
HISTORY_LIMIT = 24


async def _get_codex_system_prompt(message: str) -> str:
    """Build system prompt for Codex with memory retrieval."""
    profile_context = build_profile_context().strip()
    memory_file = "" if profile_context else read_memory().strip()

    prompt = SYSTEM_PROMPT
    if profile_context:
        prompt += f"\n\n<profile>\n{profile_context}\n</profile>"
    if memory_file:
        prompt += f"\n\n<long_term_memory>\n{memory_file}\n</long_term_memory>"

    memory_hits = await asyncio.to_thread(retrieve_memory, message)
    if memory_hits:
        prompt += f"\n\n<retrieved_memories>\n{memory_hits}\n</retrieved_memories>"

    return prompt


async def stream_codex_chat(
    message: str,
    conv_id: str,
    session_id: str | None = None,
    model: str = "codex",
    effort: str = "medium",
    extended: bool = True,
    timing_callback: Callable[[str], None] | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream chat via Codex CLI (OpenAI backend)."""
    system_prompt = await _get_codex_system_prompt(message)

    history: list[dict] = []
    try:
        rows, _, _ = conversation_messages(conv_id)
        for msg in rows[-HISTORY_LIMIT:]:
            role = msg["role"]
            text = msg.get("text") or ""
            if text.strip():
                history.append({"role": role, "content": text[:4000]})
    except ConversationNotFound:
        pass

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    logger.info("codex chat: %d history messages, %d total", len(history), len(messages))

    proc = await asyncio.create_subprocess_exec(
        str(CODEX_BIN), "chat", "--json", "--no-tools",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    input_json = json.dumps({"messages": messages}, ensure_ascii=False)
    stdout, _ = await proc.communicate(input=input_json.encode())

    if proc.returncode != 0:
        raise RuntimeError(f"Codex 退出码 {proc.returncode}")

    result = json.loads(stdout.decode())
    text = result.get("content", result.get("message", {}).get("content", ""))

    if text:
        yield {"event": "delta", "text": text}
    yield {"event": "done", "session_id": f"codex-{conv_id[:8]}"}
