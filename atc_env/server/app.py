"""FastAPI app created via the OpenEnv factory.

One line wires the full OpenEnv HTTP surface:
  POST /reset
  POST /step
  GET  /state
  GET  /health   (added by create_app)

Usage:
  uvicorn atc_env.server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from openenv.core.env_server import create_app

from ..models import ATCAction, ATCObservation
from .atc_environment import ATCEnvironment

app = create_app(
    ATCEnvironment,
    ATCAction,
    ATCObservation,
    env_name="atc_env",
)
