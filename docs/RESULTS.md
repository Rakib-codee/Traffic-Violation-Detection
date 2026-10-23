# Results

Regenerated from real runs; do not edit by hand.

# Detector mAP

run: `smoke`
weights: `weights/detector.pt`
dataset: `data/coco8/data.yaml`
seed: 42
device: mps
val images: 4

| class | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| person | 0.5305 | 0.2627 |
| dog | 0.4975 | 0.4747 |
| horse | 0.9950 | 0.4747 |
| elephant | 0.7450 | 0.4747 |
| umbrella | 0.9950 | 0.4747 |
| potted plant | 0.4975 | 0.4747 |
| **all** | **0.7101** | **0.4747** |
