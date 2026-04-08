"""UI route tests for the ATC inference console."""

from __future__ import annotations

from fastapi.testclient import TestClient

import server.app as app_module


def test_root_ui_contains_inference_console() -> None:
    client = TestClient(app_module.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Inference Deck" in response.text
    assert 'id="hf-token"' in response.text
    assert 'id="scene-card"' in response.text


def test_ui_inference_endpoint_returns_mocked_payload(monkeypatch) -> None:
    client = TestClient(app_module.app)

    def fake_runner(payload: app_module.ui_runner.InferenceRunRequest):
        assert payload.model_name == "heuristic-baseline"
        return {
            "model": "heuristic-baseline",
            "api_base_url": "https://router.huggingface.co/v1",
            "total_runtime_seconds": 1.23,
            "average_agent_score": 0.88,
            "average_random_baseline": 0.165,
            "average_improvement": 0.715,
            "task_count": 1,
            "scene": "landing",
            "scene_title": "Smooth landing",
            "scene_caption": "Mocked result",
            "generated_at": "2026-04-08 00:00:00 UTC",
            "tasks": [
                {
                    "task_id": "delhi_monsoon_recovery_easy",
                    "title": "Delhi Monsoon Recovery",
                    "difficulty": "Easy",
                    "summary": "10 flights | 2 runways | weather disruption",
                    "random_baseline": 0.21,
                    "score": 0.88,
                    "improvement": 0.67,
                    "steps_used": 3,
                    "success": True,
                    "outcome": "Clean landing",
                    "logs": ["[START] ...", "[END] ..."],
                    "stderr": [],
                }
            ],
        }

    monkeypatch.setattr(app_module.ui_runner, "run_requested_inference", fake_runner)

    response = client.post(
        "/ui/run-inference",
        json={"hf_token": "", "model_name": "heuristic-baseline"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scene"] == "landing"
    assert payload["tasks"][0]["steps_used"] == 3


def test_ui_inference_endpoint_requires_token_for_hosted_model() -> None:
    client = TestClient(app_module.app)

    response = client.post(
        "/ui/run-inference",
        json={"hf_token": "", "model_name": "Qwen/Qwen2.5-7B-Instruct"},
    )

    assert response.status_code == 400
    assert "HF token is required" in response.json()["detail"]
