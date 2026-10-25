# Final report

Short summary of the system, measured results, and known limits. All numbers
come from real, seeded runs; artifacts not yet run are marked **not run**.
Detailed tables: `docs/RESULTS.md`, `docs/ROBUSTNESS.md`; data sources and
licenses: `docs/DATA.md`; privacy/ethics: `docs/PRIVACY.md`.

## System

Video -> YOLO detection -> ByteTrack tracking -> scene config (stop line,
lanes, signal ROI) -> signal-state heuristic with temporal smoothing -> rule
engine (red-light, wrong-way) -> evidence builder (blurred clip + keyframe +
JSON) -> SQLite -> FastAPI + Streamlit review dashboard. Phase 6 adds an
optional helmet rule (secondary detector on motorcycle crops), an optional
ANPR module, and a robustness evaluation.

Every output is a candidate event for human review, never an automated
penalty. Plates/faces are blurred in stored evidence by default.

## Measured results

Detection (val splits, seed 42; per-class tables in `docs/RESULTS.md`):

| model | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| vehicle detector (`weights/detector.pt`, UA-DETRAC val, 500 imgs) | 0.9702 | 0.8407 |
| helmet detector (`weights/helmet.pt`, 372 imgs) | 0.9204 | 0.5864 |
| plate detector (`weights/plate.pt`, 2048 imgs) | 0.9738 | 0.6873 |

Robustness (vehicle detector under deterministic degradations, UA-DETRAC val;
mAP@0.5): night 0.9466, rain 0.9698, glare 0.9694 vs clean 0.9702.

Rule engine (7 synthetic-trajectory scenarios): precision 1.000, recall 1.000,
F1 1.000, 0.0 false alerts/hour. The scenario suite is synthetic by
construction and acts as unit tests.

Real-video event set (3 clips, 20.1 s of UA-DETRAC footage; manifest
`configs/event_set.yaml`): precision 0.800, recall 1.000, F1 0.889.
Breakdown: negative clip (normal traffic) 0 false alerts; constructed
wrong-way clip (reversed playback) 7/7 verified events; constructed red-light
clip (synthetic signal overlay) 1/1 genuine event plus **2 false positives**
— both tracker artifacts, not rule-logic errors: the occluded van's bounding
box re-crosses the stop line when the occlusion clears (~frame 37), and the
violating car is flagged twice across a tracker ID switch (~frame 40). The
headline false-alerts/hour (358) is extrapolated from a 6.7 s clip and is not
a meaningful rate; per-clip FP counts are the honest figure. Red-light recall
rests on one constructed event; wrong-way recall is a lower bound (tracker
fragmentation). This is a smoke test, not field performance.

Tracking: ByteTrack is integrated and produces stable IDs on the sample clips
used during development; the UA-DETRAC benchmark eval (HOTA/IDF1/ID switches)
is **not run** — dataset access was requested by the repo owner and is still
pending (see `docs/DATA.md`).

System: the pipeline processes the 50-frame synthetic demo clip at ~23 fps on
an Apple-silicon laptop (MPS). No Jetson/ONNX/TensorRT edge test was done
(stretch goal, **not run**).

## Limitations

- Red-light rule depends on per-camera scene calibration and a visible signal
  ROI; views without one have the rule disabled by config.
- Signal state is a color/ROI heuristic, not a learned classifier.
- Event-level metrics on real video rest on 3 short clips (20.1 s) with one
  genuine violation per constructed case — a smoke test, not field
  performance.
- Occlusion and tracker ID switches produce duplicate/spurious red-light
  events near the stop line (2 FP in the red-light clip); a cooldown or
  track-stitching step would be the natural follow-up.
- Plate blurring uses a heuristic plate band; the ANPR module does not feed
  the blur step.
- Helmet/plate detectors are trained on third-party datasets (licenses in
  `docs/DATA.md`); check regional fit before reuse.

## Reproduction

Fresh clone: `make setup && make demo` (COCO weights only, no datasets), or
`docker build -t tvd . && docker run --rm tvd`. Full pipeline with trained
weights: see README (weights and datasets are gitignored; training notebooks
in `notebooks/`).
