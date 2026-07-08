"""
========================================
context_builder.py — 统一上下文构建器

替代分散在 main.py / openai_api.py / claude.py 中的 prompt 拼接逻辑。
所有后端共享同一套上下文组装。

最终 Prompt 结构：
┌───────────────────────────────────────┐
│ System Prompt        (~8K tokens)     │  ← 人格 + 规则
│ Relationship Memory  (~2K tokens)     │  ← 角色关系 + 用户信息（不衰减）
│ Conversation Summary (~2K tokens)     │  ← 旧对话自动压缩
│ Retrieved Memories   (~2K tokens)     │  ← Ombre-Brain 语义检索
│ Recent Conversation  (~30K tokens)    │  ← 最近 N 轮
│ Current Message                       │
└───────────────────────────────────────┘

设计原则：
- Relationship Memory 不因对话长度丢失
- Conversation Summary 持续累积
- Retrieved Memories 只注入高相关内容
- Token 预算精确控制
========================================
"""

import re
from typing import Any

# ---- Token 估算 ----
_CN_RE = re.compile(r'[一-鿿　-〿＠-￯]')

def count_tokens(text: str) -> int:
    if not text: return 0
    cn = len(_CN_RE.findall(text))
    en = len(text) - cn
    return int(cn / 1.5 + en / 4.0)

# ---- Token Budgets ----
BUDGET_SYSTEM     = 8_000
BUDGET_RELATIONSHIP = 2_500
BUDGET_SUMMARY    = 2_500
BUDGET_RETRIEVED  = 2_500
BUDGET_RECENT     = 25_000
MAX_RECENT_ROUNDS = 12

# ---- 摘要缓存（进程内，重启丢失但不影响功能） ----
_summary_cache: dict[str, str] = {}


def get_summary(conv_id: str) -> str:
    return _summary_cache.get(conv_id, "")


def set_summary(conv_id: str, summary: str) -> None:
    _summary_cache[conv_id] = summary


def build_context(
    system_prompt: str,
    relationship_memory: str,
    messages: list[dict[str, Any]],
    retrieved_memories: str,
    current_message: str,
    conv_id: str = "",
) -> dict[str, Any]:
    """
    统一构建发送给 LLM 的完整上下文。

    返回: {
        "system": str,       # 完整 system prompt
        "messages": list,    # OpenAI/API 格式的消息数组
        "stats": dict,       # 统计信息
    }
    """

    # 1. System Prompt 基础
    system = system_prompt

    # 2. Relationship Memory（不衰减的长期身份记忆）
    if relationship_memory.strip():
        system += f"\n\n## 关系记忆（永久保留）\n{relationship_memory}"

    # 3. Conversation Summary
    conv_summary = get_summary(conv_id)
    if conv_summary:
        summary_text = _trim_to_budget(conv_summary, BUDGET_SUMMARY)
        system += f"\n\n## 对话历史摘要\n{summary_text}"

    # 4. Retrieved Long-term Memories
    if retrieved_memories.strip():
        retrieved_text = _trim_to_budget(retrieved_memories, BUDGET_RETRIEVED)
        system += (
            f"\n\n## 长期记忆（检索结果）\n"
            f"以下是与你当前对话相关的长期记忆。自然地参考，不要逐字复读：\n"
            f"{retrieved_text}"
        )

    # 5. Build messages array
    chat_messages = _build_recent_messages(messages, current_message)

    # 6. Stats
    system_tokens = count_tokens(system)
    msg_tokens = sum(count_tokens(m.get("content", "")) for m in chat_messages)
    total = system_tokens + msg_tokens

    return {
        "system": system,
        "messages": chat_messages,
        "stats": {
            "system_tokens": system_tokens,
            "message_tokens": msg_tokens,
            "total_tokens": total,
            "recent_rounds": len(chat_messages) // 2,
            "has_summary": bool(conv_summary),
            "has_retrieved": bool(retrieved_memories.strip()),
            "has_relationship": bool(relationship_memory.strip()),
        },
    }


def _build_recent_messages(
    messages: list[dict[str, Any]],
    current_message: str,
) -> list[dict[str, str]]:
    """构建最近消息数组，控制 token 预算"""
    result = []
    total_chars = 0

    # 从最新到最旧取消息，直到达到预算
    for msg in reversed(messages):
        text = (msg.get("text") or "").strip()
        if not text:
            continue

        line_chars = len(text)
        if total_chars + line_chars > BUDGET_RECENT * 2:  # ~2 chars/token for mixed text
            break

        role = "assistant" if msg.get("role") == "assistant" else "user"
        result.insert(0, {"role": role, "content": text})
        total_chars += line_chars

    # 确保至少有一部分历史
    if len(result) > MAX_RECENT_ROUNDS * 2:
        result = result[-(MAX_RECENT_ROUNDS * 2):]

    # 添加当前消息
    result.append({"role": "user", "content": current_message})

    return result


def _trim_to_budget(text: str, token_budget: int) -> str:
    """按 token 预算裁剪文本"""
    chars_budget = token_budget * 2  # ~2 chars/token
    if len(text) <= chars_budget:
        return text
    # 从后面截断（保留最新内容）
    return text[:chars_budget] + "\n...[已截断]"


def extract_relationship_memory(memories_json: list[dict]) -> str:
    """
    从 Ombre-Brain 记忆列表提取关系记忆。
    关系记忆 = identity + relationship + profile + preference 类型。
    """
    rel_domains = {"identity", "relationship", "profile", "preference", "personal_info"}
    lines = []
    for m in memories_json:
        domain = m.get("domain", "")
        if isinstance(domain, str):
            domain_parts = set(domain.split(","))
        elif isinstance(domain, list):
            domain_parts = set(domain)
        else:
            continue

        if domain_parts & rel_domains:
            content = m.get("content_preview", m.get("content", ""))[:200]
            if content.strip():
                lines.append(f"- {content}")

        if len(lines) >= 15:
            break

    return "\n".join(lines)


def build_summary_prompt(messages_text: str) -> str:
    """构建摘要生成的 prompt"""
    return (
        "你是一个对话摘要工具。阅读以下对话记录，提取关键信息，输出 3-5 句中文摘要。\n\n"
        "摘要必须包含：用户身份信息、偏好和习惯、重要决定或结论、正在进行的项目。\n"
        "摘要绝不包含：闲聊寒暄、一次性问答细节、情绪描述。\n"
        "格式：纯中文，不超过 150 字。\n\n"
        f"=== 对话记录 ===\n{messages_text[:6000]}\n=== 结束 ===\n\n摘要："
    )
