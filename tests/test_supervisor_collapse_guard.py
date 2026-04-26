from training.train_grpo import _supervisor_collapse_diagnostics


def test_zero_variance_does_not_raise_when_signal_is_healthy() -> None:
    sup_rewards = [0.31] * 48
    sup_parse = [1] * 48
    sup_sigs = [f"score_json_{i}" for i in range(48)]
    sup_plans = ['[{"flight_id":"A1","runway":"09L","assigned_minute":10,"hold_minutes":0}]'] * 48

    diag = _supervisor_collapse_diagnostics(
        sup_rewards_tail=sup_rewards,
        sup_parse_tail=sup_parse,
        sup_action_sig_tail=sup_sigs,
        sup_plan_json_tail=sup_plans,
    )
    assert diag["should_raise"] is False


def test_zero_variance_raises_on_true_collapse_signature() -> None:
    sup_rewards = [0.0] * 48
    sup_parse = [0] * 48
    sup_sigs = ["parse_fail"] * 48
    sup_plans = ["[]"] * 48

    diag = _supervisor_collapse_diagnostics(
        sup_rewards_tail=sup_rewards,
        sup_parse_tail=sup_parse,
        sup_action_sig_tail=sup_sigs,
        sup_plan_json_tail=sup_plans,
    )
    assert diag["should_raise"] is True
