"""TrackEval-based tracking evaluation (HOTA, IDF1, ID switches, MOTA)."""

from __future__ import annotations

import numbers
import re
from pathlib import Path

from tvd.utils import get_logger

logger = get_logger(__name__)

_THRESHOLD_FIELD = re.compile(r"^(?P<name>.+)\(\d+\)$")

DEFAULT_CLASSES = ("pedestrian",)  # trackeval 1.3.0 only allows the 'pedestrian'
# slot for custom MOT benchmarks; vehicle GT is mapped to class id 1 (see
# scripts/convert_detrac_to_mot.py) and CLASSES_TO_EVAL stays pedestrian.


def run_trackeval(
    gt_folder: Path,
    trackers_folder: Path,
    trackers_to_eval: list[str],
    benchmark: str = "MOTCUSTOM",
    classes: tuple[str, ...] = DEFAULT_CLASSES,
) -> dict:
    """Run TrackEval HOTA/Identity/CLEAR over the given tracker outputs.

    gt_folder must follow the MOTChallenge layout (<gt>/<seq>/gt/gt.txt +
    seqinfo.ini); each tracker needs <trackers>/<name>/data/<seq>.txt in MOT
    format. Returns {tracker: {class: {metric: {field: value}}}}.
    """
    from trackeval import Evaluator, datasets

    gt_folder = Path(gt_folder)
    trackers_folder = Path(trackers_folder)

    # trackeval requires a seqmap listing sequences; derive it from the GT dir
    split_name = f"{benchmark}-all"
    seqmap = gt_folder / "seqmaps" / f"{split_name}.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p.name for p in gt_folder.iterdir() if (p / "gt" / "gt.txt").exists())
    if not seqs:
        raise ValueError(f"no sequences with gt/gt.txt found under {gt_folder}")
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n")

    eval_config = {
        "USE_PARALLEL": False,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_EMPTY_CLASSES": False,
    }
    dataset_config = {
        "GT_FOLDER": str(gt_folder),
        "TRACKERS_FOLDER": str(trackers_folder),
        "TRACKERS_TO_EVAL": trackers_to_eval,
        "TRACKER_SUB_FOLDER": "data",
        "BENCHMARK": benchmark,
        "SPLIT_TO_EVAL": "all",
        "CLASSES_TO_EVAL": list(classes),
        # custom benchmarks have no <benchmark>-<split> subfolder in the GT dir
        "SKIP_SPLIT_FOL": True,
    }
    dataset = datasets.MotChallenge2DBox(dataset_config)
    evaluator = Evaluator(eval_config)
    from trackeval.metrics import CLEAR, HOTA, Identity

    metrics_list = [HOTA(), Identity(), CLEAR()]
    raw, messages = evaluator.evaluate([dataset], metrics_list)

    # raw: {dataset: {tracker: {seq: {class: {metric: {field: value}}}}}}
    # average fields across sequences so callers get one flat summary
    results: dict = {}
    per_dataset = next(iter(raw.values()))
    for tracker, seqs in per_dataset.items():
        merged: dict = {}
        for seq_results in seqs.values():
            for cls, class_results in seq_results.items():
                dst = merged.setdefault(cls, {})
                for metric, fields in class_results.items():
                    mfields = dst.setdefault(metric, {})
                    for field, value in fields.items():
                        if isinstance(value, numbers.Real):
                            mfields.setdefault(field, []).append(float(value))

        def collapse(fields: dict[str, list[float]]) -> dict[str, float]:
            # average per-alpha-threshold fields into their combined metric
            grouped: dict[str, list[float]] = {}
            for field, values in fields.items():
                match = _THRESHOLD_FIELD.match(field)
                base = match.group("name") if match else field
                grouped.setdefault(base, []).append(sum(values) / len(values))
            return {k: sum(v) / len(v) for k, v in grouped.items()}

        results[tracker] = {
            cls: {metric: collapse(fields) for metric, fields in metrics.items()}
            for cls, metrics in merged.items()
        }

    for tracker in trackers_to_eval:
        summary = summarize(results.get(tracker, {}))
        logger.info("%s: %s", tracker, summary)
    return results


def summarize(tracker_results: dict) -> dict[str, float]:
    """Flatten the TrackEval result dict to {METRIC_field: value} for 'all'."""
    flat: dict[str, float] = {}
    for _cls, class_results in tracker_results.items():
        for metric, fields in class_results.items():
            for field, value in fields.items():
                if isinstance(value, (int, float)):
                    flat[f"{metric}_{field}"] = round(float(value), 4)
    return flat
