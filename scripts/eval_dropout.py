"""Event-level robustness study: track dropout x track stitching.

Measures how violation-event precision/recall degrades when per-frame tracks
are randomly dropped, with track stitching off vs on. Dropout is POST-HOC
(filtering tracker output): a surrogate for detector/tracker misses that is
strictly harsher than real detector loss, since ByteTrack would bridge some
misses internally. Everything is seeded; means are over --seeds seeds.

Writes docs/EVENT_ROBUSTNESS.md (committed) and results/dropout/metrics.json
(gitignored, per-seed detail).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from tvd.config import load_event_set_config, load_tracker_config
from tvd.pipeline import run_clip
from tvd.rules import EventWindow, match_events
from tvd.utils import get_logger

logger = get_logger(__name__)

BERNOULLI_LEVELS = (0.1, 0.2, 0.3)
# (burst-start probability per frame, burst length); lengths straddle the
# stitch gate (configs/tracker.yaml stitch_max_gap_frames) to show both the
# stitchable regime and the gate boundary
BURST_LEVELS = ((0.05, 4), (0.05, 8))
SEEDS = (1, 2, 3)
# stitch-gate ablation settings: (max_gap_frames, max_dist_px); the config's
# own gates are measured by the main table, so the ablation skips them
ABLATION_GATES = ((3, 30), (10, 80))


@dataclass
class TrackDropout:
    """Wraps a tracker; drops Track entries with a seeded RNG."""

    base: object
    mode: str  # "bernoulli" | "burst"
    p: float
    seed: int
    burst_len: int = 8

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._bursts: dict[int, int] = {}

    def track_frame(self, frame, frame_idx):
        tracks = self.base.track_frame(frame, frame_idx)
        out = []
        for t in tracks:
            if self.mode == "bernoulli":
                if self._rng.random() < self.p:
                    continue
            else:
                remaining = self._bursts.get(t.track_id, 0)
                if remaining > 0:
                    self._bursts[t.track_id] = remaining - 1
                    continue
                if self._rng.random() < self.p:
                    self._bursts[t.track_id] = self.burst_len - 1
                    continue
            out.append(t)
        return out


def run_cell(
    manifest,
    tracker_config,
    device,
    work_dir,
    mode,
    level,
    stitch,
    seeds,
    burst_len=8,
    config_override=None,
):
    """One (mode, level, stitching) cell; returns per-seed metrics."""
    per_seed = []
    for seed in seeds:
        tp = fp = fn = 0
        for clip_idx, clip in enumerate(manifest.clips):

            def transform(base, clip_idx=clip_idx, seed=seed, burst_len=burst_len):
                return TrackDropout(
                    base, mode, level, seed=seed * 100 + clip_idx, burst_len=burst_len
                )

            events, _, _ = run_clip(
                clip,
                tracker_config,
                device,
                work_dir,
                stitch=stitch,
                tracker_transform=transform,
                config_override=config_override,
            )
            predicted = [
                EventWindow(e.violation_type.value, e.start_frame, e.end_frame) for e in events
            ]
            expected = [
                EventWindow(e.type, e.start_frame, e.end_frame) for e in clip.expected_events
            ]
            c_tp, c_fp, c_fn = match_events(predicted, expected, manifest.match_tolerance_frames)
            tp, fp, fn = tp + c_tp, fp + c_fp, fn + c_fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_seed.append(
            {
                "seed": seed,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
        logger.info(
            "mode=%s level=%s stitch=%s seed=%d: P=%.3f R=%.3f F1=%.3f",
            mode,
            level,
            stitch,
            seed,
            precision,
            recall,
            f1,
        )
    return per_seed


def mean_metric(per_seed, key):
    return round(sum(s[key] for s in per_seed) / len(per_seed), 3)


def std_metric(per_seed, key):
    return round(statistics.pstdev(s[key] for s in per_seed), 3)


def fmt(per_seed, key):
    return f"{mean_metric(per_seed, key):.3f} ± {std_metric(per_seed, key):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/event_set.yaml")
    parser.add_argument("--tracker-config", default="configs/tracker.yaml")
    parser.add_argument("--out-dir", default="results/dropout")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    manifest = load_event_set_config(args.manifest)
    out_dir = Path(args.out_dir)
    cfg = load_tracker_config(args.tracker_config)
    cfg_gates = (cfg.stitch_max_gap_frames, cfg.stitch_max_dist_px)
    # cell = (mode label, dropout param, burst length used when mode is burst)
    cells = [("bernoulli", p, None) for p in BERNOULLI_LEVELS]
    cells += [("burst", p, length) for p, length in BURST_LEVELS]

    results = {}
    for stitch in (False, True):
        for mode, level, burst_len in cells:
            per_seed = run_cell(
                manifest,
                args.tracker_config,
                args.device,
                out_dir / "work",
                mode,
                level,
                stitch,
                SEEDS,
                burst_len=burst_len or 8,
            )
            results[(mode, level, burst_len, stitch)] = per_seed

    # stitch-gate ablation (stitching on); gates equal to the config are
    # already measured in the main table
    ablation = {}
    for gap, dist in ABLATION_GATES:
        if (gap, dist) == cfg_gates:
            continue
        for mode, level, burst_len in cells:
            override = {
                "stitch_enabled": True,
                "stitch_max_gap_frames": gap,
                "stitch_max_dist_px": dist,
            }
            per_seed = run_cell(
                manifest,
                args.tracker_config,
                args.device,
                out_dir / "work",
                mode,
                level,
                True,
                SEEDS,
                burst_len=burst_len or 8,
                config_override=override,
            )
            ablation[(mode, level, burst_len, gap, dist)] = per_seed

    lines = [
        "# Event-level robustness under track dropout",
        "",
        "Regenerated by `scripts/eval_dropout.py` from real runs; do not edit by hand.",
        "",
        f"Event set: {args.manifest} (3 clips). Dropout is a post-hoc surrogate",
        "for detector/tracker misses (harsher than real detector loss). Burst cells",
        f"show the burst length in parentheses; the stitch gate is {cfg_gates[0]} frames"
        f" / {cfg_gates[1]:.0f} px, so len-8 bursts sit beyond it.",
        f"Values are mean ± std over seeds {SEEDS}.",
        "",
        "| mode | dropout | stitching | precision | recall | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for mode, level, burst_len in cells:
        for stitch in (False, True):
            per_seed = results[(mode, level, burst_len, stitch)]
            label = f"{level} (len {burst_len})" if burst_len else str(level)
            lines.append(
                f"| {mode} | {label} | {'on' if stitch else 'off'} "
                f"| {fmt(per_seed, 'precision')} | {fmt(per_seed, 'recall')} "
                f"| {fmt(per_seed, 'f1')} |"
            )

    if ablation:
        lines += [
            "",
            "Stitch-gate ablation (stitching on; mean ± std over seeds). Gates are "
            "(max gap frames / max distance px):",
            "",
            "| mode | dropout | gates | precision | recall | F1 |",
            "|---|---|---|---|---|---|",
        ]
        for (mode, level, burst_len, gap, dist), per_seed in sorted(
            ablation.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])
        ):
            label = f"{level} (len {burst_len})" if burst_len else str(level)
            lines.append(
                f"| {mode} | {label} | {gap}f/{dist:.0f}px "
                f"| {fmt(per_seed, 'precision')} | {fmt(per_seed, 'recall')} "
                f"| {fmt(per_seed, 'f1')} |"
            )
        lines += [
            "",
            f"Current config gates ({cfg_gates[0]}f/{cfg_gates[1]:.0f}px) are in the main "
            "table above.",
        ]

    doc = "\n".join(lines) + "\n"
    Path("docs/EVENT_ROBUSTNESS.md").write_text(doc)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "main": {
            f"{mode}/{level}{f' (len {burst_len})' if burst_len else ''}"
            f"/stitch={'on' if stitch else 'off'}": per_seed
            for (mode, level, burst_len, stitch), per_seed in results.items()
        },
        "gate_ablation": {
            (
                f"{mode}/{level}{f' (len {burst_len})' if burst_len else ''}/{gap}f/{dist:.0f}px"
            ): per_seed
            for (mode, level, burst_len, gap, dist), per_seed in ablation.items()
        },
        "config_gates": list(cfg_gates),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    logger.info("wrote docs/EVENT_ROBUSTNESS.md and %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
