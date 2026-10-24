# Robustness evaluation (Phase 6)

Detector mAP on the UA-DETRAC-10K val split under deterministic degradations.
weights: `weights/detector.pt`  seed: 42  device: mps

| condition | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| clean | 0.9702 | 0.8407 |
| night | 0.9466 | 0.8094 |
| rain | 0.9698 | 0.8324 |
| glare | 0.9694 | 0.8373 |
