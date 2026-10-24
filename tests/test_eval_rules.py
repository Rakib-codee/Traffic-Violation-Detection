from scripts.eval_rules import evaluate

from tvd.rules.scenarios import SCENARIOS


def test_scenario_suite_has_enough_scenarios():
    assert len(SCENARIOS) >= 5
    # every scenario must declare expected violations explicitly
    assert all(s.expected is not None for s in SCENARIOS)


def test_eval_rules_runs_and_writes_metrics(tmp_path):
    result = evaluate(tmp_path)
    assert (tmp_path / "metrics.json").exists()
    for key in ("precision", "recall", "f1", "false_alerts_per_hour"):
        assert key in result
    assert result["n_scenarios"] == len(SCENARIOS)
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0


def test_eval_rules_current_suite_is_clean(tmp_path):
    """The committed scenario suite must score perfectly; a regression in the
    rule engine makes this test fail before it reaches a demo."""
    result = evaluate(tmp_path)
    assert result["fp"] == 0
    assert result["fn"] == 0
