"""
========================================
environment_manager.py — 环境感知管理器

赋予 AI 实时环境感知能力。
每轮聊天实时获取，不写入 Memory/Summary/Relationship。

数据来源：
- 时间/日期：Python datetime（零延迟，零 API 调用）
- 天气：open-meteo 免费 API（无需注册，按需获取）
- 位置：IP 反查或手动配置

注入格式：结构化短文本，控制 Token 消耗。
========================================
"""

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any
from urllib import request, error

logger = logging.getLogger("env.mgr")

# ---- 缓存 ----
_weather_cache: dict[str, Any] = {}
_weather_cache_time: float = 0
WEATHER_CACHE_TTL = 1800  # 30 分钟


def get_time_context(tz_offset: int | None = None) -> dict[str, Any]:
    """
    获取当前时间上下文。
    tz_offset: UTC 偏移小时数（如 +8 表示北京时间），默认使用系统时区。
    """
    if tz_offset is not None:
        tz = timezone(timedelta(hours=tz_offset))
        now = datetime.now(tz)
    else:
        now = datetime.now()

    hour = now.hour

    # 时段判断
    if 5 <= hour < 8:
        period = "清晨"
    elif 8 <= hour < 12:
        period = "上午"
    elif 12 <= hour < 14:
        period = "中午"
    elif 14 <= hour < 18:
        period = "下午"
    elif 18 <= hour < 22:
        period = "晚上"
    elif 22 <= hour < 24:
        period = "深夜"
    else:
        period = "凌晨"

    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

    return {
        "time": now.strftime("%H:%M"),
        "date": now.strftime("%Y-%m-%d"),
        "weekday": weekday_names[now.weekday()],
        "period": period,
        "hour": hour,
        "month": now.month,
        "year": now.year,
    }


def get_weather_context(city: str = "Beijing", lat: float | None = None,
                        lon: float | None = None) -> dict[str, Any]:
    """
    获取天气上下文（open-meteo 免费 API，无需注册）。

    返回 None 或天气数据字典。
    缓存 30 分钟。
    """
    global _weather_cache, _weather_cache_time

    cache_key = f"{lat},{lon}" if lat else city
    now = time.time()

    # 读缓存
    if cache_key in _weather_cache and (now - _weather_cache_time) < WEATHER_CACHE_TTL:
        return _weather_cache[cache_key]

    # 默认坐标：北京
    if lat is None:
        lat, lon = 39.9, 116.4

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
            f"&timezone=auto"
        )
        req = request.Request(url, headers={"User-Agent": "OmbreBrain/1.0"})
        with request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        current = data.get("current", {})
        weather_code = current.get("weather_code", 0)
        weather_desc = _code_to_weather(weather_code)

        result = {
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "weather": weather_desc,
            "wind_speed": current.get("wind_speed_10m"),
            "city": city,
        }

        _weather_cache[cache_key] = result
        _weather_cache_time = now
        return result

    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
        # 返回过期缓存（如果有）
        if cache_key in _weather_cache:
            return _weather_cache[cache_key]
        return {"error": "unavailable"}


def _code_to_weather(code: int) -> str:
    """WMO weather code → 中文描述"""
    if code == 0: return "晴"
    if code in (1, 2, 3): return "多云"
    if code in (45, 48): return "雾"
    if code in (51, 53, 55): return "毛毛雨"
    if code in (61, 63, 65): return "雨"
    if code in (71, 73, 75): return "雪"
    if code in (80, 81, 82): return "阵雨"
    if code in (95, 96, 99): return "雷暴"
    return "阴"


def build_environment_prompt(tz_offset: int = 8, city: str = "Beijing",
                              lat: float | None = None, lon: float | None = None) -> str:
    """
    构建注入 Prompt 的环境文本。
    控制在 300 字符以内。
    """
    time_ctx = get_time_context(tz_offset)
    weather_ctx = get_weather_context(city, lat, lon)

    parts = []

    # 时间
    parts.append(
        f"现在时间是 {time_ctx['date']} {time_ctx['weekday']} "
        f"{time_ctx['time']}，{time_ctx['period']}。"
    )

    # 天气
    if weather_ctx and "error" not in weather_ctx:
        parts.append(
            f"天气：{weather_ctx.get('weather', '?')}，"
            f"{weather_ctx.get('temperature', '?')}°C，"
            f"湿度 {weather_ctx.get('humidity', '?')}%。"
        )

    return " ".join(parts)
