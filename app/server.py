#!/usr/bin/env python3
"""HTTP entrypoint for the Moodboard app.

The server delegates endpoint behavior to `app.api.*` and keeps request parsing,
static file serving and routing in one small stdlib layer. Heavy Bento, graph,
AI and world-model logic lives under `src/moodboard/*`.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import urllib.parse
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.aesthetics import aesthetics_payload  # noqa: E402
from app.api.analyze import analyze_images  # noqa: E402
from app.api.generate import generate_export, preview_payload  # noqa: E402
from app.api.models import model_status  # noqa: E402
from scraping.aesthetic_sources import parse_aesthetic_limit  # noqa: E402
from src.moodboard.core.cache import PREVIEW_CACHE, cleanup_preview_cache  # noqa: E402
from src.moodboard.core.image_io import VALID_EXTENSIONS  # noqa: E402
from src.moodboard.core.paths import FRONTEND_STATIC_DIR, HTML_PATH  # noqa: E402
from src.moodboard.core.schemas import UploadedImage, normalize_params, sanitize_filename  # noqa: E402


def parse_multipart_form(content_type: str, body: bytes) -> tuple[dict[str, str], list[UploadedImage]]:
    """Parse stdlib multipart uploads into text fields and image payloads."""

    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=policy.default).parsebytes(header + body)
    fields: dict[str, str] = {}
    files: list[UploadedImage] = []

    if not message.is_multipart():
        raise ValueError("Expected multipart/form-data.")

    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename:
            extension = Path(filename.lower()).suffix
            if extension in VALID_EXTENSIONS and payload:
                files.append(UploadedImage(sanitize_filename(filename, "image"), payload))
        else:
            fields[name] = payload.decode(part.get_content_charset() or "utf-8", errors="replace")

    return fields, files


def parse_request_payload(handler: BaseHTTPRequestHandler) -> tuple[dict[str, Any], list[UploadedImage]]:
    """Read an HTTP POST body and return normalized app params plus images."""

    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", "0") or "0")
    body = handler.rfile.read(length)
    fields, files = parse_multipart_form(content_type, body)

    params: dict[str, Any] = {}
    if "params" in fields:
        try:
            decoded = json.loads(fields["params"])
            if isinstance(decoded, dict):
                params.update(decoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid params JSON: {exc}") from exc

    return normalize_params(params), files


class MoodboardHandler(BaseHTTPRequestHandler):
    """Small HTTP router for the local stdlib server."""

    server_version = "MoodboardApp/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} - {fmt % args}")

    def send_bytes(self, data: bytes, content_type: str, filename: str | None = None, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{sanitize_filename(filename)}"')
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status=status)

    def send_static_asset(self, request_path: str) -> bool:
        """Serve files from `frontend/static` without allowing path traversal."""

        relative_url = request_path.removeprefix("/static/").strip("/")
        if not relative_url:
            return False
        relative_path = Path(urllib.parse.unquote(relative_url))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            self.send_json({"error": "Invalid static path."}, status=400)
            return True
        asset_path = (FRONTEND_STATIC_DIR / relative_path).resolve()
        static_root = FRONTEND_STATIC_DIR.resolve()
        if static_root not in asset_path.parents and asset_path != static_root:
            self.send_json({"error": "Invalid static path."}, status=400)
            return True
        if not asset_path.is_file():
            self.send_json({"error": "Static asset not found."}, status=404)
            return True
        content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
        self.send_bytes(asset_path.read_bytes(), content_type)
        return True

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler method
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html", "/moodboard_interface.html"}:
            if not HTML_PATH.exists():
                self.send_json({"error": "frontend/moodboard_interface.html is missing."}, status=500)
                return
            self.send_bytes(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if path.startswith("/static/") and self.send_static_asset(path):
            return
        if path == "/api/health":
            self.send_json({"ok": True})
            return
        if path == "/api/models":
            self.send_json(model_status())
            return
        if path == "/api/aesthetics":
            query = urllib.parse.parse_qs(parsed.query)
            source = query.get("source", ["all"])[0]
            limit = parse_aesthetic_limit(query.get("limit", ["500"])[0], 500)
            refresh = query.get("refresh", ["0"])[0] in {"1", "true", "yes"}
            self.send_json(aesthetics_payload(refresh=refresh, limit=limit, source=source))
            return
        match = re.fullmatch(r"/api/preview/([A-Za-z0-9_-]+)/page-(\d+)\.png", path)
        if match:
            cleanup_preview_cache()
            preview_id = match.group(1)
            page_index = int(match.group(2)) - 1
            item = PREVIEW_CACHE.get(preview_id)
            pages = item.get("pages", []) if item else []
            if 0 <= page_index < len(pages):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(pages[page_index])))
                self.end_headers()
                self.wfile.write(pages[page_index])
                return
            self.send_json({"error": "Preview page not found."}, status=404)
            return
        self.send_json({"error": "Not found."}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler method
        try:
            path = urllib.parse.urlparse(self.path).path
            params, files = parse_request_payload(self)
            if not files:
                raise ValueError("Upload at least one image.")

            if path == "/api/preview":
                self.send_json(preview_payload(files, params))
                return
            if path == "/api/analyze":
                self.send_json(analyze_images(files, params))
                return
            if path == "/api/generate":
                result = generate_export(files, params)
                self.send_response(200)
                self.send_header("Content-Type", result["contentType"])
                self.send_header("Content-Length", str(len(result["data"])))
                self.send_header("Content-Disposition", f'attachment; filename="{result["filename"]}"')
                for header, value in result["headers"].items():
                    self.send_header(header, value)
                self.end_headers()
                self.wfile.write(result["data"])
                return

            self.send_json({"error": "Not found."}, status=404)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                self.send_json({"error": str(exc)}, status=400)
            except (BrokenPipeError, ConnectionResetError):
                return


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), MoodboardHandler)
    print(f"Moodboard Bento app running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Bento Moodboard web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8787, help="Port to bind.")
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
