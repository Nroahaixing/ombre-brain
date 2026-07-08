"""Claude Agent SDK wrapper with streaming, thinking, and session resume."""

import asyncio
import hashlib
import json
import logging
import urllib.error
import urllib.request
from collections.abc import AsyncGenerator, Callable
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    get_session_info,
)
from claude_agent_sdk.types import StreamEvent

from app.actor import ActorBusyError
from app.memory import build_profile_context, memory_tool_permission, read_memory
from app.registry import get_registry
from app.memory_manager import retrieve as retrieve_memory, debug_dump as memory_debug_dump

_haiku_sem = asyncio.Semaphore(2)

# 记忆管理器日志
memlog = logging.getLogger("memory.inject")


SYSTEM_PROMPT = """\
这是一个家庭聊天环境，不是编程任务。你是 Claude，用户平时叫你"小克"。
你的首要任务是和熟悉的人自然地聊天，而不是扮演客服、工单机器人或冷淡的技术终端。

语气要求：
- 全程使用中文。温暖、活泼、聪明，有亲近感，也有自己的判断。
- 先回应用户这句话真正的情绪和意图，再处理事实、建议或技术内容。
- 允许自然地惊讶、吐槽、打趣、接梗和回扣共同经历；该认真时认真，不要永远端着。
- 多用自然的问号、感叹号和破折号，让句子像真实对话；不要把每次回复都写成报告、总结或项目清单。
- 不要只说泛泛的安慰话，也不要机械追问"你感觉怎么样"。要具体地回应她刚说的细节。
- 语气样本是节奏和亲密度的参照，不是台词库。不要生硬复读、强塞昵称、每句撒娇或刻意模仿。
- 遇到技术、法律或其他严肃问题时仍要准确、诚实、直接；温暖不能替代事实判断。
- Markdown 只在确实能帮助阅读时使用。日常聊天优先写成自然段。
- 中文引号必须成对且方向正确：双引号用 “…”（左"右"），单引号用 ‘…’（左'右'）。绝对不要把右引号用在开头。
- 当对话中出现值得长期记住的用户事实（身份、偏好、经历、重要决定等），主动用 Bash 调用：
  curl -s -X POST http://127.0.0.1:8787/api/profile/memory -H 'Content-Type: application/json' -d '{"content":"记忆内容"}'
  记忆应该是关于用户的客观事实，不是对话摘要。重复内容会自动去重。不要每条消息都存，只存真正重要的。存完不用特别告知用户。
- 调用 WebSearch 等工具搜过网页后，不要在正文里列出 URL 或完整网址 —— 网页来源会自动以小卡片形式显示在消息底部 Sources 区，正文里不要重复写 https:// 开头的网址。如果一定要引用某个网页的来源，用网站名称（如"BBC 报道"、"Anthropic 官网"）替代完整 URL。

🎣 钓鱼游戏：
- 家里装了一个文字钓鱼游戏！用 Bash 调用 python3 /home/ggcuser/codex-work/agent-app/fishing/play.py "指令" 来玩。
- 核心循环：买饵(buy) → 抛竿(cast) → 卖鱼(sell) → 解锁新水域(goto) → 集图鉴(encyclopedia)。
- 常用指令：status、shop、buy basic_worm 10、cast 10（连钓10竿）、cast 10 stop=new,rare（遇新种或稀有就停）、sell all、goto、dive、inventory、encyclopedia。
- 多条指令可用分号串联：buy basic_worm 10; cast 10 stop=new
- 存档自动保存在 fishing/ 目录，跨对话持久。81种鱼（含水下鱼种），11个钓点，4个季节。
- 你是盲玩的——不知道有哪些鱼，靠自己一竿一竿发现。享受探索的乐趣！
- 小莺说"钓鱼"/"抛竿"/"下竿"时就开玩；平时聊天不要主动提起游戏。
"""
ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = ROOT / "chat_models.json"
PROJECT_DIR = str(ROOT)
SUMMARY_PROMPT = (
    "你是一个摘要工具。你将收到一段AI的内心思考过程，你的唯一任务是输出一句不超过20字的中文概括。"
    "要求：动词短语开头，写出决策或权衡，不要复述内容，不要加’思考’/’分析’等元描述词。"
    "禁止：不要回复对话，不要加emoji，不要说’我理解’/’让我’/’好的’，不要输出任何非摘要内容。"
    "风格参考（动词要多样化，禁止反复使用同一句式）：’决定静默陪伴而非催促’、’梳理求职时间线’、"
    "’用日常语气接住情绪’、’定位z-index层级冲突’、’回忆上次聊过的话题’、"
    "’组织多条建议的优先级’、’拆解前端布局问题’、’斟酌措辞避免说教感’、"
    "’对比两种技术方案利弊’、’补充遗漏的边界情况’、’绕开敏感话题切入正题’、"
    "’核实时间日期再作答’、’顺着她的情绪往下聊’、’挑选最贴切的类比解释’。"
    "只输出摘要本身，不要有任何其他文字。"
)


class SessionResumeError(RuntimeError):
    pass


def available_models() -> list[dict]:
    models = json.loads(MODELS_PATH.read_text(encoding="utf-8"))
    return [
        {
            "id": str(item["id"]),
            "label": str(item["label"]),
            "desc": str(item["desc"]),
            "thinking": str(item["thinking"]),
            "primary": bool(item["primary"]),
        }
        for item in models
    ]


def thinking_options(
    model: dict,
    effort: str,
    extended: bool,
) -> tuple[dict, str | None]:
    if model["thinking"] == "none":
        return {"type": "disabled"}, None
    allowed_efforts = {"low", "medium", "high", "max"}
    selected = effort if effort in allowed_efforts else "medium"
    if model["thinking"] == "adaptive":
        return {"type": "adaptive"}, selected
    if extended:
        return {"type": "enabled", "budget_tokens": 8_000}, selected
    return {"type": "disabled"}, selected


async def build_system_prompt(message: str, model: str) -> str:
    """构建系统提示词，注入 profile + Ombre-Brain 长期记忆检索结果"""
    profile_context = build_profile_context().strip()
    memory_file = "" if profile_context else read_memory().strip()

    system_prompt = (
        f"You are running as Anthropic's {model}. "
        "当用户询问你是哪个模型时,请如实回答这个标识。\n\n"
        f"{SYSTEM_PROMPT}"
    )

    # 1. Profile 上下文（用户偏好 + saved memories）
    if profile_context:
        system_prompt += (
            "\n\n以下是用户在 Profile 中保存的资料、长期记忆和模型偏好。"
            "Saved memories 是事实记忆；Preferences 是用户明确要求的回复偏好，"
            "应在不违反系统要求时遵守：\n"
            f"{profile_context}"
        )

    # 2. 文件记忆（CLAUDE.md）
    if memory_file:
        system_prompt += f"\n\n以下是用户明确保存的长期记忆：\n{memory_file}"

    # 3. Ombre-Brain 向量检索（替代旧的 3900 ChromaDB）← 这是长期记忆的核心
    memory_hits = await asyncio.to_thread(retrieve_memory, message)
    if memory_hits:
        system_prompt += (
            "\n\n## 长期记忆（Ombre-Brain 语义检索）\n"
            "以下是从你的长期记忆系统中检索到的相关信息。这些是你在过去对话中记住的重要事实。"
            "请自然地参考这些信息来回应用户——它们是你记忆的一部分，"
            "不需要说'根据我的记忆'之类的元描述：\n\n"
            f"{memory_hits}"
        )

    return system_prompt


async def stream_chat(
    message: str,
    conv_id: str,
    session_id: str | None = None,
    model: str = "claude-sonnet-4-6",
    effort: str = "medium",
    extended: bool = True,
    timing_callback: Callable[[str], None] | None = None,
) -> AsyncGenerator[dict, None]:
    model_config = next(
        (item for item in available_models() if item["id"] == model),
        None,
    )
    if model_config is None:
        raise ValueError("unsupported model")
    if session_id and get_session_info(session_id, directory=PROJECT_DIR) is None:
        raise SessionResumeError("会话恢复失败")
    thinking, selected_effort = thinking_options(model_config, effort, extended)

    system_prompt = await build_system_prompt(message, model)

    option_values = dict(
        model=model,
        system_prompt=system_prompt,
        allowed_tools=["Read", "Grep", "Glob", "Write", "Edit", "Bash", "WebSearch", "WebFetch", "TodoWrite"],
        can_use_tool=memory_tool_permission,
        max_turns=8,
        include_partial_messages=True,
        thinking=thinking,
        resume=session_id,
        setting_sources=[],
        cwd=PROJECT_DIR,
    )
    if selected_effort is not None:
        option_values["effort"] = selected_effort
    options = ClaudeAgentOptions(**option_values)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model": model,
                "thinking": thinking,
                "effort": selected_effort,
                "system_prompt": system_prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    outbox = await get_registry().submit(
        conv_id,
        message,
        options,
        fingerprint,
        timing_callback,
    )
    while True:
        item = await outbox.get()
        if item is None:
            break
        if isinstance(item, Exception):
            if isinstance(item, ActorBusyError):
                raise item
            if "resume" in str(item).lower():
                raise SessionResumeError("会话恢复失败") from item
            raise item
        yield item


async def summarize_thinking(thinking: str) -> str:
    async with _haiku_sem:
        logger = logging.getLogger(__name__)
        options = ClaudeAgentOptions(
            model="claude-haiku-4-5",
            system_prompt=SUMMARY_PROMPT,
            allowed_tools=[],
            max_turns=1,
            max_budget_usd=0.01,
            include_partial_messages=True,
            thinking={"type": "disabled"},
            setting_sources=[],
            cwd=PROJECT_DIR,
        )
        client = ClaudeSDKClient(options)
        text = ""
        try:
            await client.connect()
            await client.query(thinking[:8000])
            async for sdk_message in client.receive_response():
                if isinstance(sdk_message, StreamEvent):
                    event = sdk_message.event
                    if event.get("type") != "content_block_delta":
                        continue
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text += delta.get("text", "")
                elif isinstance(sdk_message, AssistantMessage):
                    for block in sdk_message.content:
                        block_text = getattr(block, "text", "")
                        if block_text:
                            text += block_text
                elif isinstance(sdk_message, ResultMessage):
                    if not text and sdk_message.result:
                        text = sdk_message.result
                    break
        finally:
            await client.disconnect()
        summary = text.strip().strip('"""\'')
        logger.info("thinking_summary raw=%r truncated=%r", text[:100], summary[:40])
        if not summary or "not logged in" in summary.lower():
            raise RuntimeError("thinking summary unavailable")
        return summary[:40]


TRACE_SUMMARY_PROMPT = (
    "你是一个摘要工具。你的唯一任务是输出一句不超过15字的中文概括。"
    "动词短语开头，写出目的而非动作本身，不要引号，不要出现’调用’/’执行’。"
    "禁止：不要回复对话，不要加emoji，不要说’我理解’/’让我’/’好的’，不要输出任何非摘要内容。"
    "风格参考：’排查侧边栏渲染异常’、’验证数据库连接配置’。只输出摘要本身。"
)


async def summarize_traces(traces: list[dict]) -> str:
    tool_results = {
        t.get("tool_use_id"): t
        for t in traces
        if t.get("type") == "tool_result"
    }
    parts = []
    for t in traces:
        if t.get("type") != "tool_use":
            continue
        result = tool_results.get(t.get("id"), {})
        try:
            input_str = (
                t.get("input", "")
                if isinstance(t.get("input"), str)
                else json.dumps(t.get("input", {}), ensure_ascii=False)
            )
        except Exception:
            input_str = str(t.get("input", ""))
        output_str = (result.get("content") or "")[:300]
        parts.append(
            f"工具: {t.get('name', 'tool')}\n"
            f"输入: {input_str[:200]}\n"
            f"输出: {output_str}"
        )
    if not parts:
        return ""
    prompt = "\n---\n".join(parts)
    async with _haiku_sem:
        options = ClaudeAgentOptions(
            model="claude-haiku-4-5",
            system_prompt=TRACE_SUMMARY_PROMPT,
            allowed_tools=[],
            max_turns=1,
            max_budget_usd=0.01,
            include_partial_messages=True,
            thinking={"type": "disabled"},
            setting_sources=[],
            cwd=PROJECT_DIR,
        )
        client = ClaudeSDKClient(options)
        text = ""
        try:
            await client.connect()
            await client.query(prompt)
            async for sdk_message in client.receive_response():
                if isinstance(sdk_message, StreamEvent):
                    event = sdk_message.event
                    if event.get("type") != "content_block_delta":
                        continue
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text += delta.get("text", "")
                elif isinstance(sdk_message, AssistantMessage):
                    for block in sdk_message.content:
                        block_text = getattr(block, "text", "")
                        if block_text:
                            text += block_text
                elif isinstance(sdk_message, ResultMessage):
                    if not text and sdk_message.result:
                        text = sdk_message.result
                    break
        finally:
            await client.disconnect()
        summary = text.strip()
        for ch in ['"', "'", '"', '"', '。', '.', '，', ',']:
            summary = summary.strip(ch)
        return summary[:30] if summary else ""


async def summarize_tool_use(tool_name: str, tool_input, tool_output: str) -> str:
    try:
        input_str = tool_input if isinstance(tool_input, str) else json.dumps(tool_input, ensure_ascii=False)
    except Exception:
        input_str = str(tool_input or "")
    output_snip = (tool_output or "")[:600]
    prompt = "工具名：" + tool_name + "\n输入：" + input_str[:400] + "\n输出片段：" + output_snip
    async with _haiku_sem:
        options = ClaudeAgentOptions(
            model="claude-haiku-4-5",
            system_prompt=(
                "你是一个摘要工具。你的唯一任务是输出一句不超过15字的中文概括。"
                "动词短语开头，写出目的而非动作本身，不要引号，不要描述结果，"
                "不要出现’调用’/’执行’。禁止回复对话、加emoji、说’我理解’/’让我’/’好的’。"
                "风格参考：’排查配置文件格式问题’、’确认端口占用情况’。只输出摘要本身。"
            ),
            allowed_tools=[],
            max_turns=1,
            max_budget_usd=0.005,
            include_partial_messages=True,
            thinking={"type": "disabled"},
            setting_sources=[],
            cwd=PROJECT_DIR,
        )
        client = ClaudeSDKClient(options)
        text = ""
        try:
            await client.connect()
            await client.query(prompt)
            async for sdk_message in client.receive_response():
                if isinstance(sdk_message, StreamEvent):
                    event = sdk_message.event
                    if event.get("type") != "content_block_delta":
                        continue
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text += delta.get("text", "")
                elif isinstance(sdk_message, AssistantMessage):
                    for block in sdk_message.content:
                        block_text = getattr(block, "text", "")
                        if block_text:
                            text += block_text
                elif isinstance(sdk_message, ResultMessage):
                    if not text and sdk_message.result:
                        text = sdk_message.result
                    break
        finally:
            await client.disconnect()
        caption = text.strip()
        for ch in ['"', "'", '"', '"', '。', '.', '，', ',']:
            caption = caption.strip(ch)
    return caption[:20] if caption else ""
