# Presentation deck — Traffic Violation Detection

Slide-by-slide content for the project defense / demo. ~12 slides, 8–10 min.
Visual assets: `docs/assets/demo-tracking.gif` (hero), tables from
`docs/RESULTS.md`, `docs/EVENT_ROBUSTNESS.md`.

---

## Slide 1 — Title

**Automatic Traffic Violation Detection from Video**
Detect → track → understand the scene → flag candidate violations for human review.
Rakib Islam — [course/project name], 2026
Visual: `docs/assets/demo-tracking.gif`

## Slide 2 — Problem & scope

- Traffic violations are rare events in hours of footage — manual review
  does not scale.
- Goal: flag **candidate** events (red-light, wrong-way; helmet non-use as
  extension) with reviewable evidence.
- Explicit non-goal: no automated penalties — every output goes to a human
  reviewer; plates/faces blurred by default.

## Slide 3 — System pipeline

```
video -> YOLO detection -> ByteTrack (+ track stitching)
      -> scene config (stop line, lanes, signal ROI) -> signal heuristic
      -> rule engine -> evidence (blurred clip + JSON) -> SQLite
      -> FastAPI + Streamlit review dashboard
```
Per-camera YAML calibration (stop line, lane polygons, signal ROI); cameras
with no visible signal have the red-light rule disabled by config.

## Slide 4 — Detection results

| model | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| vehicle (UA-DETRAC val, 500 img) | 0.970 | 0.841 |
| helmet (372 img) | 0.920 | 0.586 |
| plate (2048 img) | 0.974 | 0.687 |

Robustness under night/rain/glare: mAP@0.5 drop ≤ 0.03.

## Slide 5 — Rule engine

- Red-light: stop-line crossing while signal is red (tolerance window,
  occlusion-recross suppression).
- Wrong-way: sustained opposing motion vs lane direction.
- Validated on 9 synthetic-trajectory scenarios (unit-test style): P/R/F1 = 1.0.

## Slide 6 — Evaluation on real video: the hard part

Violations are rare — no labeled violation dataset exists for our cameras.
Our approach: **constructed test cases with ground truth by construction**:
- negative clip: normal traffic (false-alarm check)
- reversed playback → every moving vehicle is wrong-way
- synthetic signal overlay → controlled red/green windows

Every label verified by frame-by-frame video review.

## Slide 7 — Real-video results

3 clips (20.1 s UA-DETRAC): **P = R = F1 = 1.000**, 0 false alerts.
The set caught real bugs before any field use:
- occlusion bbox re-cross → spurious red-light (fixed: recross window;
  set scored P=0.80 before the fix)
- a "tracker duplicate" that turned out to be a genuine second violator —
  we corrected the label, not the metric.

## Slide 8 — Novelty: track stitching

Problem we measured: tracker gaps/ID switches fragment vehicles → rule
engine loses violations (recall collapses under misses).
Solution: lightweight online tracklet stitching — constant-velocity
geometric gating, no re-id model — plus a matching gap tolerance in the
rule engine.

## Slide 9 — Stitching: measured impact

Event recall under seeded track dropout (3 seeds, precision stays 1.000):

| dropout | off | on |
|---|---|---|
| 10% random | 0.593 | **0.963** |
| 20% random | 0.370 | **0.704** |
| 30% random | 0.333 | **0.593** |
| occlusion bursts (4 f) | 0.778 | **0.889** |

Honest limits: bursts longer than the 5-frame gate are not bridged; dense
queues needed tightened gates (mis-merge risk without appearance features).

## Slide 10 — Privacy & ethics

- Candidates for human review only — never automated fines.
- Plates/faces blurred in stored evidence by default.
- Short evidence clips, not continuous footage; dataset licenses recorded
  (`docs/DATA.md`); deployment policy recommendations in `docs/PRIVACY.md`.

## Slide 11 — Engineering quality

- 102 pytest tests, ruff-clean, seeded runs, pinned dependencies.
- Fresh-clone demo: `make demo`; Docker CPU image: `docker run --rm tvd`.
- All results regenerated from real runs; "not run" items stated explicitly
  (UA-DETRAC tracking benchmark — access pending).

## Slide 12 — Limits & future work

- Small event set (smoke test, not field performance).
- Geometry-only stitching; appearance/re-id is the next step.
- Signal state is a heuristic; a small CNN is a drop-in upgrade.
- UA-DETRAC HOTA/IDF1 benchmark once dataset access arrives.

---

### Demo flow (live or recorded)

1. `make demo` — fresh-clone end-to-end (synthetic video, rule eval).
2. `python scripts/run_video.py --video data/samples/detrac_mvi20012.mp4
   --weights weights/detector.pt --run-name detrac-demo` — tracking overlay.
3. `python scripts/eval_events.py` — real-video event metrics.
4. Optional: dashboard on `results/events.db`.
