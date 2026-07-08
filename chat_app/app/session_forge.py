"""
========================================
session_forge.py — Session Forge 适配器

适配 Session Forge 概念到 Web Chat：
- 不是 tmux-based，而是通过 Conversation Summary + Memory Tail 实现
- 新 Session 自动恢复：最近事件、当前话题、最近摘要
- 与现有 Memory 和 Summary 共存

核心功能：
- session_handoff(): 生成 Session 交接文档
- session_restore_context(): 恢复 Session 上下文
- session_stats(): 获取当前 Session 统计
========================================
"""

import time
from datetime import datetime
from typing import Any

from app.context_builder import get_summary, count_tokens
from app.store import conversation_messages


def session_handoff(conv_id: str) -> dict[str, Any]:
    """
    生成 Session 交接文档。
    包含：最近对话摘要、当前话题、Token 使用量。
    """
    try:
        msgs, _, _ = conversation_messages(conv_id)
    except Exception:
        return {
            "conv_id": conv_id, "timestamp": datetime.now().isoformat(),
            "summary": "", "recent_tail": "", "message_count": 0, "total_tokens": 0,
        }

    summary = get_summary(conv_id)
    recent = msgs[-10:] if len(msgs) > 10 else msgs
    tail_lines = []
    for m in recent:
        role = "pai" if m.get("role") == "user" else "An"
        text = (m.get("text") or "")[:150]
        if text.strip():
            tail_lines.append(f"{role}: {text}")

    return {
        "conv_id": conv_id,
        "timestamp": datetime.now().isoformat(),
        "summary": summary or "（暂无摘要）",
        "recent_tail": "\n".join(tail_lines),
        "message_count": len(msgs),
        "total_tokens": sum(count_tokens(m.get("text", "")) for m in msgs),
    }


def session_restore_prompt(handoff: dict[str, Any]) -> str:
    """
    从交接文档生成恢复 Prompt。
    注入到新 Session 的 System Prompt 中。
    """
    if not handoff:
        return ""

    return (
        f"[会话恢复 — 你正在继续之前的对话，不是从零开始]\n"
        f"## 之前的对话摘要\n{handoff.get('summary', '')}\n\n"
        f"## 最近对话\n{handoff.get('recent_tail', '')}\n\n"
        f"[请自然地延续以上对话。不要打招呼或重新介绍自己。]"
    )


def session_stats(conv_id: str) -> dict[str, Any]:
    """获取当前 Session 统计"""
    try:
        msgs, _, _ = conversation_messages(conv_id)
        total_tokens = sum(count_tokens(m.get("text", "")) for m in msgs)
        msg_count = len(msgs)
    except Exception:
        msgs = []
        total_tokens = 0
        msg_count = 0

    summary = get_summary(conv_id)

    return {
        "message_count": msg_count,
        "total_tokens": total_tokens,
        "has_summary": bool(summary),
        "timestamp": time.time(),
    }