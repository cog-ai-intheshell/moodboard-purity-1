#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility entrypoint for the reorganized Moodboard app.

The implementation now lives in the package layout:

- ``app/server.py`` for the local HTTP server;
- ``app/api/*`` for endpoint adapters;
- ``src/moodboard/*`` for Bento, analysis, AI models and shared core code.

Keeping this small shim preserves the old ``python3 moodboard_app.py`` command
without reintroducing the legacy monolith as an import dependency.
"""

from __future__ import annotations

from app.server import main


if __name__ == "__main__":
    main()
