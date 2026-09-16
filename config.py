"""
Viora configuration.
All configurable settings in one place. LLM backend can be swapped easily.
"""

import os

# ── Server ──────────────────────────────────────────────────────────────────
SERVER_HOST = os.environ.get("VIORA_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("VIORA_PORT", "5050"))
DEBUG = os.environ.get("VIORA_DEBUG", "false").lower() == "true"

# ── Storage ─────────────────────────────────────────────────────────────────
# Root data directory (relative to this project)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MESSAGES_FILE = os.path.join(DATA_DIR, "messages.json")
TRAJECTORY_EVENTS_FILE = os.path.join(DATA_DIR, "trajectory_events.json")
TRAJECTORIES_FILE = os.path.join(DATA_DIR, "trajectories.json")
PSYCHO_EVENTS_FILE = os.path.join(DATA_DIR, "psycho_events.json")
PSYCHO_STATES_FILE = os.path.join(DATA_DIR, "psycho_states.json")
PSYCHO_FINGERPRINTS_FILE = os.path.join(DATA_DIR, "psycho_fingerprints.json")
MRT_DATA_FILE = os.path.join(DATA_DIR, "mrt_data.json")

# ── LLM (OpenAI-compatible via DeepSeek) ───────────────────────────────────
LLM_API_BASE = os.environ.get("VIORA_LLM_API_BASE", "https://api.deepseek.com/v1")
LLM_API_KEY = os.environ.get("VIORA_LLM_API_KEY", "")
LLM_MODEL = os.environ.get("VIORA_LLM_MODEL", "deepseek-v4-flash")

# Timeout for LLM calls (seconds)
LLM_TIMEOUT = int(os.environ.get("VIORA_LLM_TIMEOUT", "60"))
