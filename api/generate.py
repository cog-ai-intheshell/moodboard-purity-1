from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moodboard_app import build_export, normalize_params, parse_multipart_form, render_pages  # noqa: E402


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

            params = normalize_params(raw_params)
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
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)
