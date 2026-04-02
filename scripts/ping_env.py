"""Ping a Space or local server and verify reset() works."""

from __future__ import annotations

import json
import sys

import requests


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/ping_env.py <base_url>")
        return 1

    base_url = sys.argv[1].rstrip("/")
    health = requests.get(f"{base_url}/health", timeout=10)
    print(f"health_status={health.status_code}")
    health.raise_for_status()

    reset = requests.post(f"{base_url}/reset", json={"task_id": "delhi_monsoon_recovery_easy"}, timeout=20)
    print(f"reset_status={reset.status_code}")
    reset.raise_for_status()
    payload = reset.json()
    print(json.dumps({"task_id": payload["observation"]["task_id"], "done": payload["done"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
