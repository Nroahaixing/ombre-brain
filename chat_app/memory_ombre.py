"""
Chatnest ↔ Ombre-Brain 记忆桥接层

替换 Chatnest 自带的简单文件记忆，改用 Ombre-Brain 的 MCP 记忆工具。
支持两种模式：
1. HTTP 模式：Ombre-Brain 作为独立 HTTP 服务运行，Chatnest 通过 API 调用
2. 内置模式：直接导入 Ombre-Brain 模块（同进程）
"""

import json
import httpx
import os
from typing import Optional

OMBRE_BRAIN_URL = os.environ.get("OMBRE_BRAIN_URL", "http://127.0.0.1:8000")


async def recall_memory(query: str, limit: int = 5) -> list[dict]:
    """
    聊天前检索相关记忆。
    调用 Ombre-Brain 的 breath 工具做语义检索。
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{OMBRE_BRAIN_URL}/api/search",
                json={"query": query, "limit": limit},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", [])
    except Exception:
        pass
    return []


async def save_memory(content: str, tags: list[str] | None = None, importance: float = 0.5):
    """
    聊天后保存重要记忆。
    调用 Ombre-Brain 的 hold 工具。
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{OMBRE_BRAIN_URL}/api/buckets",
                json={
                    "content": content,
                    "tags": tags or [],
                    "importance": importance,
                    "source": "chat",
                },
            )
    except Exception:
        pass


async def search_memory(query: str, domain: str | None = None) -> list[dict]:
    """更丰富的记忆检索（带 domain 过滤）"""
    try:
        params = {"query": query, "limit": 10}
        if domain:
            params["domain"] = domain

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{OMBRE_BRAIN_URL}/api/search",
                params=params,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", [])
    except Exception:
        pass
    return []


async def get_memory_context(query: str) -> str:
    """
    获取格式化的记忆上下文，用于注入聊天 prompt。
    返回可直接拼接到系统提示的文本。
    """
    memories = await recall_memory(query, limit=8)
    if not memories:
        return ""

    lines = ["\n## 相关记忆\n"]
    for m in memories:
        content = m.get("content", "")[:300]
        domain = m.get("domain", "general")
        importance = m.get("importance", 0.5)
        star = "⭐" if importance > 0.7 else "·"
        lines.append(f"- {star} [{domain}] {content}")
    return "\n".join(lines)
