"""Gold SFT JSON round-trips through AMAN/DMAN parsers."""

from training.dataset import parse_aman_action, parse_dman_action
from training.sft_data import build_grounded_json_sft_rows


def test_build_grounded_sft_rows_parseable():
    rows = build_grounded_json_sft_rows(10, seed=0)
    assert len(rows) >= 9
    for r in rows:
        gold = r["messages"][-1]["content"]
        role = r["agent_role"]
        if role == "AMAN":
            assert parse_aman_action(gold) is not None
        elif role == "DMAN":
            assert parse_dman_action(gold) is not None
