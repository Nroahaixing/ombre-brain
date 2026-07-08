"""
========================================
memory_manager.py — 统一记忆管理器

替换原有的 memory_search.py (ChromaDB 3900 端口)。
直接调用 Ombre-Brain 的 HTTP API 做检索和存储。

核心功能：
1. retrieve(query)       — 聊天前检索相关记忆（→ Ombre-Brain /api/search）
2. save_fact(content)     — 提取重要事实后保存（→ Ombre-Brain hold）
3. should_save(text)      — 智能判断是否值得保存（不是所有内容都存）
4. memory_debug_context() — 获取调试信息

架构：
  Chatnest 聊天 → memory_manager.retrieve() → Ombre-Brain HTTP API → BucketManager → .md 文件
  Chatnest 聊天 → memory_manager.save_fact() → Ombre-Brain HTTP API → hold

设计原则：
- 长期保存：用户偏好、身份信息、长期目标、技术栈、重要事实
- 不保存：普通闲聊、一次性问题、无意义寒暄
========================================
"""

import json
import logging
import os
import re
from typing import Any
from urllib import request, error

logger = logging.getLogger("memory.manager")

OMBRE_URL = os.environ.get("OMBRE_BRAIN_URL", "http://127.0.0.1:8000")
TIMEOUT = 3.0  # 比旧的 1.5s 长，因为 Ombre-Brain 首次检索需要 embedding

# ============================================================
# Debug Mode
# ============================================================
DEBUG = os.environ.get("MEMORY_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

_memory_debug: dict[str, Any] = {}

def debug_log(key: str, value: Any) -> None:
    if DEBUG:
        _memory_debug[key] = value

def debug_dump() -> str:
    if not DEBUG:
        return ""
    lines = ["\n===== Memory Debug ====="]
    for k, v in _memory_debug.items():
        val_str = str(v)
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        lines.append(f"{k}: {val_str}")
    lines.append("========================\n")
    _memory_debug.clear()
    return "\n".join(lines)


# ============================================================
# 智能记忆过滤器 — 判断是否值得长期保存
# ============================================================

# 值得保存的模式（按优先级排序）
SAVE_PATTERNS = [
    # 用户身份 & 偏好
    (r"(?:我叫|我是|我的名字是|你可以叫我|称呼我)[^\n。]{3,30}", "identity"),
    (r"(?:我喜欢|我习惯|我偏好|我讨厌|我不喜欢|我通常)[^\n。]{5,80}", "preference"),
    # 长期目标 & 项目
    (r"(?:我的目标是|我正在做|我在开发|我在写|我在研究|我在学|我计划|我想做)[^\n。]{8,100}", "goal"),
    (r"(?:我的项目|我的工作|我们公司在|我们团队)[^\n。]{8,100}", "project"),
    # 技术栈
    (r"(?:我用|我的技术栈|我擅长|我的工具)[^\n。]{5,60}", "tech_stack"),
    (r"(?:Python|React|TypeScript|Rust|Go|Java|Node\.js|Next\.js|Docker|Kubernetes|AWS|GCP)[^\n。]{5,60}", "tech_stack"),
    # 重要事实
    (r"(?:记住|别忘了|重要|关键的)[^\n。]{8,120}", "important"),
    (r"(?:我住在|我在|我的地址|我的邮箱|我的电话)[^\n。]{5,60}", "personal"),
]

# 不保存的模式
SKIP_PATTERNS = [
    r"^(?:你好|嗨|hello|hi|hey|bye|再见|晚安|早安|谢谢|多谢|ok|好的|嗯|哦|哈哈|笑死)[\s!！。.,，]*$",
    r"^(?:今天天气|几点了|现在几点|今天是周几|今天星期几)[^\n。]*$",
    r"^(?:什么是|怎么|如何|帮我|能不能|可以帮我|请问)[^\n。]*$",  # 一般问答
    r"^[^a-zA-Z一-鿿]*$",  # 纯符号/数字
]


def should_save_fact(text: str) -> tuple[bool, str, str]:
    """
    判断一段文本是否值得存入长期记忆。

    返回: (是否保存, 原因, 分类标签)
    """
    text = text.strip()
    if len(text) < 8:
        return False, "too_short", ""
    if len(text) > 500:
        text = text[:500]

    # 检查跳过模式
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return False, "skip_pattern", ""

    # 检查保存模式
    for pattern, category in SAVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True, category, category

    # 默认：如果文本够长且包含实质性内容（有动词+名词结构），则保存
    if len(text) > 30 and re.search(r"[一-鿿]{4,}", text):
        # 中文文本，有一定长度
        words = len(re.findall(r"[一-鿿]", text))
        if words > 10:
            return True, "general_fact", "general"

    return False, "no_pattern_match", ""


# ============================================================
# Ombre-Brain API 调用
# ============================================================

def _http_post(path: str, payload: dict, timeout: float = TIMEOUT) -> dict | None:
    """调用 Ombre-Brain HTTP API。静默失败。"""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{OMBRE_URL}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (error.URLError, OSError, json.JSONDecodeError, TimeoutError) as e:
        logger.warning(f"Ombre-Brain API 调用失败 ({path}): {e}")
        return None


def _http_get(path: str, timeout: float = TIMEOUT) -> dict | None:
    try:
        req = request.Request(
            f"{OMBRE_URL}{path}",
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (error.URLError, OSError, json.JSONDecodeError, TimeoutError) as e:
        logger.warning(f"Ombre-Brain API 调用失败 ({path}): {e}")
        return None


# ============================================================
# 核心记忆函数
# ============================================================

def retrieve(query: str, budget_chars: int = 2000, top_k: int = 8) -> str:
    """
    聊天前检索相关记忆。
    调用 Ombre-Brain GET /api/search?q= 做语义+关键词检索。

    返回: 格式化的记忆文本，可直接加入 prompt
    """
    q = (query or "").strip()
    if len(q) < 3:
        debug_log("Memory Retrieve", "query too short, skipped")
        return ""

    # Ombre-Brain /api/search 是 GET 端点（不是 POST）
    import urllib.parse
    encoded_q = urllib.parse.quote(q)
    data = _http_get(f"/api/search?q={encoded_q}")
    if not data:
        debug_log("Memory Retrieve", "API 调用失败或返回空")
        return ""

    # Ombre-Brain 返回的是数组 [{id, name, score, domain, content_preview, ...}]
    results = data if isinstance(data, list) else data.get("results", [])
    if not results:
        debug_log("Memory Retrieve", f"检索成功但无结果 (query={q[:50]})")
        return ""

    lines = []
    for r in results[:top_k]:
        content = r.get("content_preview", r.get("content", ""))[:300]
        domain = r.get("domain", "")
        bucket_id = r.get("id", "")
        if content.strip():
            domain_tag = f"[{domain}]" if domain else ""
            lines.append(f"- {domain_tag} {content}")

    joined = "\n".join(lines)
    debug_log("Memory Retrieve", f"{len(results)} 条结果, {len(joined)} chars")
    debug_log("Retrieved Memories", joined)

    return joined


def save_fact(content: str, tags: str = "chat", importance: int = 6) -> bool:
    """
    保存一条事实到长期记忆。

    Args:
        content: 事实文本
        tags: 标签（用于分类）
        importance: 重要性 (1-10)
    """
    should, reason, category = should_save_fact(content)
    if not should:
        debug_log("Memory Filter", f"跳过: {reason} — {content[:80]}")
        return False

    if category and category not in tags:
        tags = f"{tags},{category}"

    payload = {
        "content": content,
        "tags": tags,
        "importance": min(importance, 10),
        "source": "chat_auto",
    }

    data = _http_post("/api/buckets", payload)
    if data and not data.get("error"):
        debug_log("Memory Saved", f"[{tags}] {content[:120]}")
        logger.info(f"memory saved: [{category}] {content[:80]}...")
        return True

    debug_log("Memory Save Failed", str(data))
    return False


def extract_and_save_facts(user_message: str, assistant_reply: str) -> list[str]:
    """
    从一轮对话中提取值得保存的事实。

    用简单的规则匹配提取用户侧的关键信息。
    不依赖 LLM（避免额外 API 调用），但准确率高。

    返回: 已保存的事实列表
    """
    saved = []

    # 从用户消息中提取
    for line in user_message.split("\n"):
        line = line.strip()
        if not line:
            continue
        should, reason, category = should_save_fact(line)
        if should:
            success = save_fact(line, f"chat,{category}")
            if success:
                saved.append(line[:100])

    # 从 assistant 回复中不提取（回复是分析性的，不是事实）
    # 但检查用户是否说了 "记住 XXX"
    remember_pattern = re.findall(r"(?:记住|别忘了|请记住)[：:\s]*(.+?)(?:[。！!，,\n]|$)", user_message)
    for fact in remember_pattern:
        fact = fact.strip()
        if len(fact) > 5:
            success = save_fact(fact, "chat,important", importance=8)
            if success:
                saved.append(fact[:100])

    return saved


def health_check() -> dict:
    """检查 Ombre-Brain 服务是否正常运行"""
    data = _http_get("/health", timeout=2.0)
    if data and data.get("status") == "ok":
        return {"status": "ok", "version": data.get("version", "?")}
    return {"status": "unreachable"}
