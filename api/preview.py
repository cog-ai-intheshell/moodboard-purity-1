from __future__ import annotations

import base64
import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moodboard_app import (  # noqa: E402
    page_to_preview_bytes,
    parse_multipart_form,
    preview_params,
    render_pages,
    normalize_params,
)


class handler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 - Vercel handler API
        try:
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", "0") or "0")
            fields, files = parse_multipart_form(content_type, self.rfile.read(length))
            if not files:
                raise ValueError("Upload at least one image.")

            raw_params: dict[str, Any] = {}
            if "params" in fields:
                decoded = json.loads(fields["params"])
                if isinstance(decoded, dict):
                    raw_params = decoded

            pages, infos, images_per_page = render_pages(files, preview_params(normalize_params(raw_params)))
            encoded_pages = []
            for page in pages:
                png = page_to_preview_bytes(page)
                page.close()
                encoded_pages.append("data:image/png;base64," + base64.b64encode(png).decode("ascii"))

            self.send_json(
                {
                    "pages": encoded_pages,
                    "imageCount": len(infos),
                    "pageCount": len(encoded_pages),
                    "imagesPerPage": images_per_page,
                }
            )
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)
