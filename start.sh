#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CHAT_PORT="${CHAT_PORT:-8787}"
OMBRE_PORT="${OMBRE_PORT:-8000}"

echo "=============================================="
echo "  Ombre-Brain + Chatnest 统一启动"
echo "  聊天界面 : http://127.0.0.1:${CHAT_PORT}"
echo "  记忆管理 : http://127.0.0.1:${OMBRE_PORT}"
echo "=============================================="

# 1. 启动 Ombre-Brain MCP 记忆服务（后台）
echo "[1/2] 启动 Ombre-Brain 记忆引擎..."
cd "$ROOT"
python3 -m src.server --port "$OMBRE_PORT" &
OMBRE_PID=$!
sleep 2

# 检查 Ombre-Brain 是否成功启动
if ! kill -0 $OMBRE_PID 2>/dev/null; then
    echo "❌ Ombre-Brain 启动失败"
    exit 1
fi
echo "  ✅ Ombre-Brain 运行中 (pid=$OMBRE_PID)"

# 2. 启动 Chatnest 聊天服务（前台）
echo "[2/2] 启动 Chatnest 聊天界面..."
export OMBRE_BRAIN_URL="http://127.0.0.1:${OMBRE_PORT}"

cd "$ROOT/chat_app"
python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$CHAT_PORT" \
    --log-level info

# 清理
kill $OMBRE_PID 2>/dev/null
echo "已停止所有服务"
