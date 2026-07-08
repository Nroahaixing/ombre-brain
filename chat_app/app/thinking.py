"""
========================================
thinking.py — Thinking 模式模块

实现 Claude 风格的 Thinking 功能。
完全独立模块，不影响人格、聊天历史、长期记忆。

两种模式：
- Normal (off)：直接输出回答
- Thinking (on)：先生成内部推理，再输出回答

工作流程（Thinking ON）：
1. 第一轮 API 调用：用 thinking_profile 生成完整内部推理
2. 推理结果作为 thinking event 发送给前端
3. 第二/同一轮：基于推理结果生成最终回答
4. 最终回答作为 delta event 正常流式输出

Thinking 不写入：
- 聊天历史（SQLite）
- 长期记忆（Ombre-Brain）
- Conversation Summary

接口统一，未来可切换到底层模型原生 Thinking（Claude extended thinking）。
========================================
"""

from collections.abc import AsyncGenerator
from typing import Callable

# Thinking 输出格式标记
THINKING_START = "<thinking>"
THINKING_END = "</thinking>"

THINKING_SYSTEM = f"""在回复用户之前，你必须先用 {THINKING_START} 和 {THINKING_END} 标签输出你的内部推理。这是强制要求。

格式：
{THINKING_START}
写下你的内心推理：用户这句话的真正意图、你该如何回应、有没有需要参考的记忆。写完整的内心独白，连贯的长段落。这是写给你自己的，诚实、不修饰。
{THINKING_END}

然后输出你给用户的最终回复。不要在标签外再写推理内容。

务必使用 {THINKING_START} 开始推理，用 {THINKING_END} 结束推理。标签必须成对出现。"""

THINKING_SYSTEM_OLD = """你正在进行内心独白。这是一个私密的思考过程，不会展示给用户。

请从以下几个角度展开你的思考：
1. 用户这句话的真正意图和情绪是什么？
2. 什么样的回应方式最适合当下的语境？
3. 有哪些长期记忆或关系信息与当前话题相关？
4. 如果要表达亲密或情感，什么样的语言和节奏最合适？

思考必须是连贯的长段落，用中文，第一人称"我"。
这是你写给自己的，不是写给用户的。诚实、流动、不修饰。
不要输出最终回答——只输出你的思考过程。"""


class ThinkingMode:
    """Thinking 模式管理器"""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def set(self, enabled: bool) -> None:
        self.enabled = enabled


# 全局实例
thinking_mode = ThinkingMode(enabled=False)


async def generate_thinking(
    message: str,
    model: str,
    stream_fn: Callable[..., AsyncGenerator[dict, None]],
    conv_id: str = "",
    **kwargs,
) -> str:
    """
    生成 Thinking 推理过程。

    调用同一个模型，但使用 thinking system prompt。
    返回完整的推理文本。
    """
    # 使用一个临时的 thinking stream
    thinking_text = ""

    async for chunk in stream_fn(
        message=message,
        conv_id=f"{conv_id}_thinking" if conv_id else "thinking",
        model=model,
        **{k: v for k, v in kwargs.items() if k not in ("message", "conv_id", "model")},
    ):
        if chunk.get("event") == "delta":
            thinking_text += chunk.get("text", "")

    return thinking_text


def build_thinking_prompt(user_message: str, system_prompt: str) -> tuple[str, str]:
    """
    为 Thinking 模式构建双重 prompt。

    返回: (thinking_prompt, final_prompt)
    - thinking_prompt: 用于生成推理的 prompt（包含原始 system + thinking 指令）
    - final_prompt: 用于生成最终回答的 prompt（包含推理结果）
    """
    thinking_system = system_prompt + "\n\n" + THINKING_SYSTEM
    return thinking_system, system_prompt
