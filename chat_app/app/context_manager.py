"""
========================================
context_manager.py — 向后兼容包装器

所有功能已迁移到 context_builder.py。
此文件保留以确保现有代码无需修改。
========================================
"""

from app.context_builder import (
    count_tokens,
    build_context,
    build_summary_prompt,
    get_summary,
    set_summary,
    extract_relationship_memory,
)

# Legacy alias
build_chat_context = build_context
