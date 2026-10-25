"""Generate paper figures from the dropout study output.

Reads results/dropout/metrics.json (written by scripts/eval_dropout.py) and
writes PDF/PNG figures under paper/figures/. Run after the dropout study.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from tvd.utils import get_logger

logger = get_logger(__name__)

KEY_RE = re.compile(
    r"^(?P<mode>[a-z]+)/(?P<level>[0-9.]+)(?: \(len (?P<burst>\d+)\))?/stitch=(?P<stitch>on|off)$"
)
ABL_RE = re.compile(
    r"^(?P<mode>[a-z]+)/(?P<level>[0-9.]+)(?: \(len (?P<burst>\d+)\))?"
    r"/(?P<gap>\d+)f/(?P<dist>[0-9]+)px$"
)


def mean_std(per_seed, key):
    values = [s[key] for s in per_seed]
    return statistics.mean(values), statistics.pstdev(values)


def key_level(key: str) -> float:
    match = KEY_RE.match(key) or ABL_RE.match(key)
    return float(match.group("level"))


def recall_curve(metrics: dict, stitch: str) -> tuple[list, list, list]:
    """(levels, means, stds) of event recall over bernoulli dropout levels."""
    levels, means, stds = [], [], []
    for key, per_seed in sorted(metrics.items(), key=lambda kv: key_level(kv[0])):
        m = KEY_RE.match(key)
        if not m or m.group("mode") != "bernoulli" or m.group("stitch") != stitch:
            continue
        mean, std = mean_std(per_seed, "recall")
        levels.append(float(m.group("level")))
        means.append(mean)
        stds.append(std)
    return levels, means, stds


def ablation_recall(metrics: dict, gap: int, dist: int):
    levels, means, stds = [], [], []
    for key, per_seed in sorted(metrics.items(), key=lambda kv: key_level(kv[0])):
        m = ABL_RE.match(key)
        if not m or m.group("mode") != "bernoulli":
            continue
        if int(m.group("gap")) != gap or int(m.group("dist")) != dist:
            continue
        mean, std = mean_std(per_seed, "recall")
        levels.append(float(m.group("level")))
        means.append(mean)
        stds.append(std)
    return levels, means, stds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="results/dropout/metrics.json")
    parser.add_argument("--out-dir", default="paper/figures")
    args = parser.parse_args()

    payload = json.loads(Path(args.metrics).read_text())
    main_metrics = payload["main"]
    ablation = payload.get("gate_ablation", {})
    cfg_gap, cfg_dist = payload["config_gates"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    series = [(recall_curve(main_metrics, "off"), "no stitching", "o", "--", "#c0392b")]
    series.append(
        (
            recall_curve(main_metrics, "on"),
            f"stitching {cfg_gap}f/{cfg_dist:.0f}px (config)",
            "s",
            "-",
            "#27ae60",
        )
    )
    for gap, dist in ((3, 30), (10, 80)):
        if ablation:
            series.append(
                (
                    ablation_recall(ablation, gap, dist),
                    f"stitching {gap}f/{dist}px (ablation)",
                    "^",
                    ":",
                    "#2980b9",
                )
            )
    for data, label, marker, ls, color in series:
        levels, means, stds = data
        if not levels:
            continue
        ax.errorbar(
            levels,
            means,
            yerr=stds,
            marker=marker,
            linestyle=ls,
            color=color,
            capsize=3,
            linewidth=1.5,
            label=label,
        )

    ax.set_xlabel("per-frame track dropout probability")
    ax.set_ylabel("event-level recall (mean ± std)")
    ax.set_xticks([0.1, 0.2, 0.3])
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"dropout_recall.{ext}"
        fig.savefig(path, dpi=200)
        logger.info("wrote %s", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
