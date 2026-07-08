"""
========================================
prompt_builder.py — 统一 Prompt 构建器

模块化组装最终 Prompt。每个模块独立维护、独立测试、独立扩展。
不再拼接成长字符串。

Prompt 结构：
┌──────────────────────────────┐
│ System Prompt                │  ← Persona Engine
│ + Environment Context        │  ← Environment Manager
│ + Relationship State        │  ← Relationship Manager
│ + Conversation Summary      │  ← Context Builder
│ + Retrieved Memories        │  ← Ombre-Brain
│ + Session Restore           │  ← Session Forge (if applicable)
├──────────────────────────────┤
│ Messages (Recent)            │
│ + Current User Message       │
└──────────────────────────────┘
========================================
"""

import logging
from typing import Any

from app.persona_engine import get_persona
from app.environment_manager import build_environment_prompt
from app.relationship_manager import build_relationship_context
from app.context_builder import build_context as _build_ctx, count_tokens, get_summary
from app.session_forge import session_restore_prompt, session_stats
from app.memory_manager import retrieve as retrieve_memory

logger = logging.getLogger("prompt.builder")


async def build_prompt(
    system_prompt: str,  # from openai_api SYSTEM_PROMPT (kept for backward compat)
    messages: list[dict[str, Any]],
    current_message: str,
    conv_id: str = "",
    include_environment: bool = True,
    include_relationship: bool = True,
    handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    统一构建完整 Prompt。

    返回: {
        "system": str,
        "messages": list,
        "stats": dict,
        "modules": dict,  # 各模块的输出（调试用）
    }
    """

    # 1. Persona (固定，不衰减)
    persona = get_persona(include_nsfw=True)

    # 2. Environment (实时)
    env = build_environment_prompt() if include_environment else ""

    # 3. Relationship (结构化)
    rel = build_relationship_context() if include_relationship else ""

    # 4. Session Restore (如果有 handoff)
    restore = session_restore_prompt(handoff) if handoff else ""

    # 5. Retrieved Memories
    retrieved = await _async_retrieve(current_message)

    # 6. Build context via existing Context Builder
    ctx = _build_ctx(
        system_prompt=system_prompt,
        relationship_memory=f"{rel}\n{env}",
        messages=messages,
        retrieved_memories=retrieved,
        current_message=current_message,
        conv_id=conv_id,
    )

    # 7. Inject Persona + Session Restore into system
    final_system = persona
    if restore:
        final_system += f"\n\n{restore}"
    final_system += f"\n\n{ctx['system']}"

    # 8. Stats
    stats = {
        "persona_tokens": count_tokens(persona),
        "environment_tokens": count_tokens(env),
        "relationship_tokens": count_tokens(rel),
        "retrieved_tokens": count_tokens(retrieved),
        "restore_tokens": count_tokens(restore),
        "message_tokens": ctx["stats"]["message_tokens"],
        "total_tokens": ctx["stats"]["total_tokens"],
        "has_summary": ctx["stats"]["has_summary"],
        "has_retrieved": ctx["stats"]["has_retrieved"],
        "conv_stats": session_stats(conv_id),
    }

    return {
        "system": final_system,
        "messages": ctx["messages"],
        "stats": stats,
        "modules": {
            "persona": persona[:200] + "...",
            "environment": env,
            "relationship": rel,
            "restore": restore[:200] + "..." if restore else "",
            "retrieved": retrieved[:200] + "..." if retrieved else "",
        },
    }


async def _async_retrieve(message: str) -> str:
    import asyncio
    return await asyncio.to_thread(retrieve_memory, message)
