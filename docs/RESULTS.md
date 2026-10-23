# Results

Regenerated from real runs; do not edit by hand.

# Detector mAP

run: `detector`
weights: `weights/detector.pt`
dataset: `data/vehicles/data.yaml`
seed: 42
device: mps
val images: 500

| class | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| bus | 0.9725 | 0.8110 |
| car | 0.9839 | 0.8270 |
| truck | 0.9330 | 0.8222 |
| van | 0.9917 | 0.9027 |
| **all** | **0.9702** | **0.8407** |
