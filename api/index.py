from __future__ import annotations

import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moodboard_app import (  # noqa: E402
    build_export,
    normalize_params,
    parse_multipart_form,
    preview_params,
    render_pages,
    RESAMPLE,
)


class handler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_headers_only(self, content_type: str, content_length: int = 0, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.end_headers()

    def read_payload(self) -> tuple[dict[str, Any], list[Any]]:
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0") or "0")
        fields, files = parse_multipart_form(content_type, self.rfile.read(length))
        params: dict[str, Any] = {}
        if "params" in fields:
            decoded = json.loads(fields["params"])
            if isinstance(decoded, dict):
                params = decoded
        return normalize_params(params), files

    def inline_preview_url(self, page: Any) -> str:
        preview = page.copy()
        preview.thumbnail((1000, 1000), RESAMPLE)
        buffer = io.BytesIO()
        preview.save(buffer, "JPEG", quality=76, optimize=True)
        preview.close()
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def do_GET(self) -> None:  # noqa: N802 - Vercel handler API
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_bytes((ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/docs/screenshot.png":
            self.send_bytes((ROOT / "docs" / "screenshot.png").read_bytes(), "image/png")
            return
        self.send_json({"error": "Not found."}, status=404)

    def do_HEAD(self) -> None:  # noqa: N802 - Vercel handler API
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_headers_only("text/html; charset=utf-8", (ROOT / "index.html").stat().st_size)
            return
        if path == "/docs/screenshot.png":
            self.send_headers_only("image/png", (ROOT / "docs" / "screenshot.png").stat().st_size)
            return
        self.send_headers_only("application/json; charset=utf-8", status=404)

    def do_POST(self) -> None:  # noqa: N802 - Vercel handler API
        try:
            path = urlparse(self.path).path
            params, files = self.read_payload()
            if not files:
                raise ValueError("Upload at least one image.")

            if path == "/api/preview":
                pages, infos, images_per_page = render_pages(files, preview_params(params))
                encoded_pages = []
                for page in pages:
                    encoded_pages.append(self.inline_preview_url(page))
                    page.close()
                self.send_json(
                    {
                        "pages": encoded_pages,
                        "imageCount": len(infos),
                        "pageCount": len(encoded_pages),
                        "imagesPerPage": images_per_page,
                    }
                )
                return

            if path == "/api/generate":
                pages, infos, images_per_page = render_pages(files, params)
                data, content_type, filename = build_export(pages, params["formats"])
                for page in pages:
                    page.close()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("X-Image-Count", str(len(infos)))
                self.send_header("X-Page-Count", str(len(pages)))
                self.send_header("X-Images-Per-Page", str(images_per_page))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_json({"error": "Not found."}, status=404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)
