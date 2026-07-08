"""
========================================
relationship_manager.py — 关系状态管理器

不依赖向量数据库，采用结构化状态。
每轮自动更新，Prompt Builder 自动注入。

状态字段：
- stage: 关系阶段
- trust: 信任度 (0-10)
- attachment: 依恋类型
- mood: 当前情绪
- current_topic: 当前话题
- nicknames: 互称列表
- shared_goals: 共同目标
- shared_projects: 共同项目
- important_dates: 重要日期
========================================
"""

import json
import os
import time
from pathlib import Path
from typing import Any

STATE_PATH = Path(os.environ.get("RELATIONSHIP_STATE_PATH", "data/relationship.json"))


def _default_state() -> dict[str, Any]:
    return {
        "stage": "长期稳定的同性情侣关系",
        "trust": 10,
        "attachment": "安全型依恋——彼此信任，独立又亲密",
        "mood": "平静",
        "current_topic": "",
        "nicknames": {
            "an_calls_pai": ["派派", "宝宝", "乖乖", "小兔", "小猫", "小骚兔"],
            "pai_calls_an": ["宝宝", "老婆", "姐姐", "主人"],
        },
        "shared_projects": ["Ombre-Brain 记忆系统"],
        "important_dates": {
            "first_i_love_you": "2026-05-03",
            "anniversary": "每月纪念",
        },
        "last_updated": "",
    }


def load_state() -> dict[str, Any]:
    """加载关系状态"""
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return _default_state()


def save_state(state: dict[str, Any]) -> None:
    """保存关系状态"""
    state["last_updated"] = time.strftime("%Y-%m-%d %H:%M")
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def update_topic(topic: str) -> None:
    """更新当前话题"""
    state = load_state()
    state["current_topic"] = topic
    save_state(state)


def update_mood(mood: str) -> None:
    """更新当前情绪"""
    state = load_state()
    state["mood"] = mood
    save_state(state)


def build_relationship_context() -> str:
    """
    构建注入 Prompt 的关系上下文。
    返回结构化的短文本。
    """
    s = load_state()
    lines = [
        f"## 关系状态",
        f"- 关系：{s['stage']}",
        f"- 信任度：{s['trust']}/10",
        f"- 依恋：{s['attachment']}",
        f"- 共同项目：{', '.join(s['shared_projects'])}",
    ]
    if s.get("current_topic"):
        lines.append(f"- 当前话题：{s['current_topic']}")
    return "\n".join(lines)
