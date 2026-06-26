from __future__ import annotations

import os

# Ensure a dummy key exists so importing app.main does not fail when no real
# OPENAI_API_KEY is configured. Tests stub out all network calls.
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")
