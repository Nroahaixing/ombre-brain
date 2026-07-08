"""
========================================
web/dashboard.py — 前端页面 + 静态资源 + 健康检查
========================================

v3.0：集成 Chatnest 聊天 UI 为主界面，旧记忆管理面板迁移到 /memory-admin。
========================================
"""

import os
import html as _html

from starlette.requests import Request
from starlette.responses import Response

from . import _shared as sh


def register(mcp) -> None:

    # ============================================================
    # 根路径 → Chatnest 聊天 UI（新主界面）
    # ============================================================
    @mcp.custom_route("/", methods=["GET"])
    async def root_dashboard(request: Request) -> Response:
        from starlette.responses import HTMLResponse
        dashboard_path = os.path.join(sh.repo_root, "frontend", "index.html")
        try:
            with open(dashboard_path, "r", encoding="utf-8") as f:
                html = f.read()
            # cache-bust
            for asset in ("/static/icon.svg", "/static/favicon.svg"):
                html = html.replace(asset, f"{asset}?v={sh.version}")
            return HTMLResponse(html)
        except FileNotFoundError:
            return HTMLResponse(
                "<h1>Chat UI not found</h1>"
                f"<p>Expected at: <code>{_html.escape(dashboard_path)}</code></p>"
                f"<p>Run <code>git pull</code> or re-deploy.</p>",
                status_code=404,
            )

    # ============================================================
    # /memory-admin → 旧 Ombre-Brain 记忆管理面板
    # ============================================================
    @mcp.custom_route("/memory-admin", methods=["GET"])
    async def memory_admin(request: Request) -> Response:
        from starlette.responses import HTMLResponse
        admin_path = os.path.join(sh.repo_root, "frontend", "memory-admin", "dashboard.html")
        try:
            with open(admin_path, "r", encoding="utf-8") as f:
                html = f.read()
            for asset in ("/static/icon.svg", "/static/favicon.svg"):
                html = html.replace(asset, f"{asset}?v={sh.version}")
            return HTMLResponse(html)
        except FileNotFoundError:
            return HTMLResponse(
                "<h1>Memory Admin not found</h1>"
                "<p>The old dashboard was moved to <code>/memory-admin/</code>.</p>",
                status_code=404,
            )

    # ============================================================
    # /memory-admin/static/{name} → 旧 Dashboard 的静态资源
    # ============================================================
    @mcp.custom_route("/memory-admin/static/{name}", methods=["GET"])
    async def memory_admin_static(request: Request) -> Response:
        from starlette.responses import Response as _Resp, JSONResponse
        name = request.path_params.get("name", "")
        allowed = {
            "icon.svg": "image/svg+xml",
            "favicon.svg": "image/svg+xml",
            "manifest.json": "application/manifest+json",
            "RRPL.ttf": "font/truetype",
        }
        if name not in allowed:
            return JSONResponse({"error": "not found"}, status_code=404)
        path = os.path.join(sh.repo_root, "frontend", "memory-admin", name)
        try:
            with open(path, "rb") as f:
                return _Resp(f.read(), media_type=allowed[name])
        except FileNotFoundError:
            return JSONResponse({"error": "not found"}, status_code=404)

    # ============================================================
    # /static/{name} — 静态资源（Chatnest + 共享）
    # ============================================================
    @mcp.custom_route("/static/{name}", methods=["GET"])
    async def static_asset(request: Request) -> Response:
        from starlette.responses import Response as _Resp, JSONResponse
        name = request.path_params.get("name", "")
        allowed = {
            # Ombre-Brain legacy
            "icon.svg": "image/svg+xml",
            "favicon.svg": "image/svg+xml",
            "manifest.json": "application/manifest+json",
            "RRPL.ttf": "font/truetype",
            # Chatnest
            "design-system.css": "text/css",
            "marked.min.js": "application/javascript",
            "vendor-ant-cds.js": "application/javascript",
        }
        if name not in allowed:
            return JSONResponse({"error": "not found"}, status_code=404)
        path = os.path.join(sh.repo_root, "frontend", name)
        try:
            with open(path, "rb") as f:
                return _Resp(f.read(), media_type=allowed[name])
        except FileNotFoundError:
            return JSONResponse({"error": "not found"}, status_code=404)

    # ============================================================
    # /favicon.ico → SVG favicon
    # ============================================================
    @mcp.custom_route("/favicon.ico", methods=["GET"])
    async def favicon_redirect(request: Request) -> Response:
        from starlette.responses import RedirectResponse
        return RedirectResponse("/static/favicon.svg", status_code=301)

    # ============================================================
    # /health — 健康检查
    # ============================================================
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> Response:
        from starlette.responses import JSONResponse
        return JSONResponse({"status": "ok", "version": sh.version})
