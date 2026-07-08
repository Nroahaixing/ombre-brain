"""
========================================
context_manager.py — 上下文管理器

核心职责：
1. Token 自动估算 & 裁剪
2. 长对话自动摘要（旧消息 → Summary）
3. 确保每次发送给 LLM 的 prompt 不超 token 限制

Prompt 组成：
┌─────────────────────────────┐
│ System Prompt    (~8K tok)  │
├─────────────────────────────┤
│ Conversation Summary (~2K)  │  ← 旧的对话被压缩到这儿
├─────────────────────────────┤
│ Long-term Memory (~2K)      │  ← Ombre-Brain 检索结果
├─────────────────────────────┤
│ Recent Messages  (~50K)     │  ← 最近 15 轮完整保留
├─────────────────────────────┤
│ Current Message             │
└─────────────────────────────┘
        ≈ 60-70K tokens（远低于 100K 上限）
========================================
"""

import re
from typing import Any

# Token 估算常量
CN_CHARS_PER_TOKEN = 1.5
EN_CHARS_PER_TOKEN = 4.0

# 各模块 token 预算
MAX_TOTAL_TOKENS = 90_000
RECENT_ROUNDS = 15          # 保留最近 N 轮对话
RECENT_CHARS_MAX = 80_000   # 最近消息最大字符数
OLD_SUMMARY_CHARS = 4_000   # 旧对话摘要最大字符数


def count_tokens(text: str) -> int:
    """粗略 token 估算。中文 ~1.5 chars/token，英文 ~4 chars/token"""
    if not text:
        return 0
    cn = len(re.findall(r'[一-鿿　-〿＀-￯]', text))
    en = len(text) - cn
    return int(cn / CN_CHARS_PER_TOKEN + en / EN_CHARS_PER_TOKEN)


def build_chat_context(
    messages: list[dict[str, Any]],
    existing_summary: str = "",
) -> tuple[str, str, dict[str, Any]]:
    """
    从 SQLite 消息列表构建发送给 LLM 的上下文。

    参数:
        messages: 数据库消息列表 [{role, text, ...}, ...]
        existing_summary: 已有的对话摘要（跨轮次累积）

    返回:
        (context_text, new_summary_text, stats)
        - context_text: 拼好的上下文文本（用于 prompt）
        - new_summary_text: 本次需要摘要的旧消息（供调用方用 LLM 生成新摘要）
        - stats: {total_tokens, recent_rounds, summary_chars, truncated}
    """
    n = len(messages)

    # 少于 15 轮 → 全部保留，无需摘要
    if n <= RECENT_ROUNDS * 2:
        lines = []
        for m in messages:
            role = "用户" if m.get("role") == "user" else "助手"
            text = (m.get("text") or "").strip()
            if text:
                lines.append(f"{role}: {text}")
        context = "\n\n".join(lines)
        return context, "", {
            "total_tokens": count_tokens(context),
            "recent_rounds": n // 2,
            "summary_chars": 0,
            "truncated": False,
        }

    # 拆分：最近 N 轮 + 旧消息
    split = max(0, n - RECENT_ROUNDS * 2)
    recent = messages[split:]
    old = messages[:split]

    # 构建最近消息（从最新往前加，控制总字符数）
    recent_lines = []
    recent_chars = 0
    for m in reversed(recent):
        role = "用户" if m.get("role") == "user" else "助手"
        text = (m.get("text") or "").strip()
        if not text:
            continue
        line = f"{role}: {text}"
        if recent_chars + len(line) > RECENT_CHARS_MAX:
            break
        recent_lines.insert(0, line)
        recent_chars += len(line)

    recent_text = "\n\n".join(recent_lines)

    # 构建旧消息文本（供摘要用）
    old_lines = []
    old_chars = 0
    for m in old:
        role = "用户" if m.get("role") == "user" else "助手"
        text = (m.get("text") or "").strip()[:300]
        if not text:
            continue
        line = f"{role}: {text}"
        if old_chars + len(line) > OLD_SUMMARY_CHARS:
            old_lines.append("...[更早的对话已省略]")
            break
        old_lines.append(line)
        old_chars += len(line)

    new_summary_material = "\n".join(old_lines) if old_lines else ""

    # 组装最终上下文
    parts = []

    if existing_summary:
        parts.append(f"## 历史对话摘要\n{existing_summary}")
    elif new_summary_material:
        # 第一次需要摘要时，把旧消息原文放进去（会有点长，但只有一次）
        parts.append(f"## 更早的对话\n{new_summary_material[:OLD_SUMMARY_CHARS]}")

    if recent_text:
        parts.append(recent_text)

    context = "\n\n---\n\n".join(parts)
    total_tokens = count_tokens(context)

    # 如果仍然超过上限，截断
    truncated = False
    if total_tokens > MAX_TOTAL_TOKENS:
        # 从最旧的消息开始删
        while total_tokens > MAX_TOTAL_TOKENS and len(parts) > 1:
            parts.pop(0)
            context = "\n\n---\n\n".join(parts)
            total_tokens = count_tokens(context)
        truncated = True

    return context, new_summary_material, {
        "total_tokens": total_tokens,
        "recent_rounds": len(recent_lines) // 2,
        "summary_chars": len(existing_summary),
        "truncated": truncated,
    }


# ============================================================
# 摘要生成（给 Haiku 用，便宜快速）
# ============================================================

SUMMARY_INSTRUCTION = """你是一个对话摘要工具。
阅读以下对话记录，提取关键信息，输出 3-5 句中文摘要。

摘要必须包含（如果有的话）：
- 用户身份信息（名字、职业、地点）
- 用户偏好和习惯
- 正在进行的项目或任务
- 重要的决定或结论
- 技术讨论的要点

摘要绝不包含：
- 闲聊、问候、寒暄
- 一次性问答的细节
- 情绪描述（"用户看起来很生气"）

格式：纯中文文本，3-5 句话，不超过 150 字。"""


def build_summary_prompt(messages_text: str) -> str:
    """构建摘要生成的用户 prompt"""
    return (
        f"{SUMMARY_INSTRUCTION}\n\n"
        f"=== 对话记录 ===\n"
        f"{messages_text[:6000]}\n"
        f"=== 结束 ===\n\n"
        f"请输出摘要："
    )
