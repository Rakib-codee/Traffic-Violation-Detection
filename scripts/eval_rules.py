"""Event-level rule evaluation on the labeled synthetic scenario suite.

Writes results/rules/metrics.json with per-type precision/recall/F1 and
false alerts per simulated hour. The hand-labeled event set (real video)
is not run yet — see docs/DATA.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tvd.rules.scenarios import SCENARIOS, run_scenario
from tvd.utils import get_logger

logger = get_logger(__name__)


def evaluate(out_dir: Path = Path("results/rules")) -> dict:
    tp = fp = fn = 0
    rows = []
    total_seconds = 0.0
    fps = 10.0

    for scenario in SCENARIOS:
        predicted, expected = run_scenario(scenario, fps=fps)
        pred_set = set(predicted)
        exp_set = set(expected)
        s_tp = len(pred_set & exp_set)
        s_fp = len(pred_set - exp_set)
        s_fn = len(exp_set - pred_set)
        tp, fp, fn = tp + s_tp, fp + s_fp, fn + s_fn
        total_seconds += scenario.frames / fps
        rows.append(
            {
                "scenario": scenario.name,
                "tp": s_tp,
                "fp": s_fp,
                "fn": s_fn,
                "predicted": [(t, v.value) for t, v in predicted],
                "expected": [(t, v.value) for t, v in expected],
            }
        )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_alerts_per_hour = fp / (total_seconds / 3600.0) if total_seconds else 0.0

    result = {
        "note": "synthetic scenario suite; hand-labeled real-video event set not run",
        "n_scenarios": len(SCENARIOS),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alerts_per_hour": round(false_alerts_per_hour, 2),
        "scenarios": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    for row in rows:
        status = "OK" if row["fp"] == 0 and row["fn"] == 0 else "MISMATCH"
        logger.info(
            "%s: %s (tp=%d fp=%d fn=%d)",
            row["scenario"],
            status,
            row["tp"],
            row["fp"],
            row["fn"],
        )
    logger.info(
        "event-level: precision=%.3f recall=%.3f f1=%.3f false-alerts/h=%.1f",
        precision,
        recall,
        f1,
        false_alerts_per_hour,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/rules")
    args = parser.parse_args()
    evaluate(Path(args.out_dir))


if __name__ == "__main__":
    main()
