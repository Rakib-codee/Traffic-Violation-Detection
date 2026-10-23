# Project 2: Automatic Traffic Violation Detection from Video

## 1. Goal
Build an end-to-end pipeline that detects and tracks vehicles in traffic video, understands the scene (lanes, stop line, signal state), applies rules to flag violations, and produces reviewable evidence (short clip + metadata) through a simple dashboard.

## 2. Scope
**MVP violations:** red-light running, wrong-way driving.
**Extension:** helmet non-use (motorcycle riders), illegal turn, stopped-vehicle detection.
**Optional module:** license plate recognition (plate detector + OCR) with privacy controls.
**Out of scope:** automated fines or legal enforcement. Output is "candidate event for human review".

## 3. System architecture
```
Video/RTSP -> Frame reader -> Detector (YOLO / RT-DETR)
   -> Tracker (ByteTrack / BoT-SORT) -> Trajectories
   -> Scene config (stop line, lane polygons, direction, signal ROI)
   -> Signal-state classifier
   -> Rule engine (per violation type)
   -> Evidence builder (clip + annotated frames + JSON)
   -> Storage (SQLite/Postgres) -> Dashboard (FastAPI + Streamlit/React)
```

## 4. Datasets (check licenses)
- Vehicles/tracking: UA-DETRAC, AI City Challenge tracks, BDD100K.
- Helmet: public helmet-detection datasets (e.g. Roboflow Universe / Kaggle sets; verify license).
- Plates (optional): CCPD or similar; verify license and region.
- Own footage: only from public/permitted sources; document permission.
- Violations are rare, so create a small hand-labeled **event-level** test set (video clip, violation type, timestamp) for evaluation.

## 5. Components and approach
1. **Detection:** fine-tune YOLO/RT-DETR on vehicle classes (car, bus, truck, motorcycle, person, helmet/no-helmet).
2. **Tracking:** ByteTrack baseline; compare BoT-SORT; tune for occlusion.
3. **Scene calibration:** annotate stop line, lane polygons, expected direction per lane in a YAML file; tool to draw them on a frame.
4. **Signal state:** classify traffic-light ROI (red/yellow/green) with a small CNN or color+shape heuristic; temporal smoothing.
5. **Rule engine:** red-light = vehicle crosses stop line while signal is red (with direction and tolerance window); wrong-way = trajectory direction opposes lane direction for N frames.
6. **Robustness:** augmentation for night/rain/glare, optional image enhancement, confidence thresholds, uncertainty flags.
7. **Evidence:** 5-10 s clip around the event, annotated frames, JSON metadata.
8. **Privacy:** blur plates and faces in stored data by default; access-controlled raw evidence; retention policy documented.
9. **Speed (optional):** homography calibration for speed estimation.

## 6. Metrics
- Detection: mAP@0.5 and mAP@0.5:0.95 per class.
- Tracking: HOTA, IDF1, ID switches.
- Violation events: precision, recall, F1 at event level; false alerts per hour.
- System: FPS, latency, GPU memory; edge test (Jetson / ONNX / TensorRT) as stretch.

## 7. Phases, milestones, exit criteria
Each phase ends with passing checks, then a commit and `git push`.

| Phase | Weeks | Deliverables | Exit criteria |
|---|---|---|---|
| 0. Setup | 1 | Repo, env, CI, sample videos, config schema | `make test` passes |
| 1. Detection | 1-2 | Fine-tuned detector, eval script | mAP table committed |
| 2. Tracking | 2-3 | Tracker integration, trajectory export, HOTA/IDF1 eval | Stable IDs on sample clips |
| 3. Scene tools | 3-4 | Calibration/annotation tool, signal classifier | Config validated on 2+ camera views |
| 4. Rule engine | 4-5 | Red-light and wrong-way rules, unit tests with synthetic trajectories | Tests pass; event-level eval runs |
| 5. Evidence + dashboard | 5-6 | Clip generator, DB, FastAPI + dashboard, plate blurring | End-to-end demo on a sample video |
| 6. Extensions | 6-7 | Helmet rule, optional ANPR, robustness tests | Added metrics for each |
| 7. Packaging | 7-8 | Docker, final report, demo video, README | Fresh-clone run reproduces demo |

## 8. Repo structure
```
traffic-violation-detection/
  README.md
  pyproject.toml / Dockerfile
  configs/            # model, tracker, camera scenes (yaml)
  src/tvd/
    detect/ track/ scene/ signal/ rules/ evidence/ api/ ui/ utils/
  scripts/            # train, eval, run_video, build_eval_set
  tests/              # rule tests with synthetic trajectories
  data/               # gitignored, with download notes
  results/ docs/
```

## 9. Risks and mitigations
- **Rare events, tiny eval set:** build event-level test set early; report confidence intervals; be honest about size.
- **Camera-specific configs:** keep scene YAML per camera; test on at least two views.
- **ID switches break rules:** use track smoothing and minimum-duration thresholds.
- **Privacy/legal:** blur by default, human-in-the-loop review, document data source and local law.
- **Domain shift (night/rain):** include in training and evaluation splits.

## 10. Final deliverables
Code, Dockerized demo, scene configs, evaluation tables, short report, demo video, privacy/ethics note.