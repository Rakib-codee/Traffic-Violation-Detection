# Traffic Violation Detection

End-to-end pipeline that detects and tracks vehicles in traffic video, understands
the scene (lanes, stop line, signal state), applies rules to flag candidate
violations, and produces reviewable evidence through a dashboard.

**Scope note:** output is always a *candidate event for human review* — never an
automated penalty. Plates and faces are blurred in stored outputs by default.

MVP violations: red-light running, wrong-way driving.
Extensions: helmet non-use, illegal turn, stopped-vehicle detection, optional ANPR.

## Setup

```bash
make setup        # create .venv and install pinned dependencies
make test         # run the test suite
make lint         # ruff check + format check
```

## Quick start

```bash
make run          # generate a synthetic sample video into data/samples/
```

## Detection (Phase 1)

```bash
# put a YOLO-format dataset in data/ (see docs/DATA.md), then:
python scripts/train_detector.py --train-config configs/train.yaml
python scripts/eval_detector.py --weights weights/detector.pt
# pipeline smoke test (tiny coco8 sample, 1 epoch):
python scripts/train_detector.py --train-config configs/train.smoke.yaml
```

The eval script regenerates `docs/RESULTS.md` (committed) and
`results/detection/<run>/metrics.json` (gitignored).

## Repository layout

```
configs/            model, tracker, system, and per-camera scene configs (YAML)
src/tvd/            library code: detect/ track/ scene/ signal/ rules/ evidence/ api/ ui/ utils/
scripts/            train, eval, run_video, sample-data generators
tests/              pytest suite (includes rule tests on synthetic trajectories)
data/               gitignored; see docs/DATA.md for sources and licenses
results/            gitignored run outputs
docs/               data licenses, privacy/ethics notes
```

## Reproducibility

All runs are seeded (see `configs/system.yaml`) and log their config.
Results files are regenerated from real runs only.

## Privacy and ethics

See `docs/DATA.md` for dataset licenses and `docs/` for the privacy note.
Raw footage of people is never committed to this repository.
