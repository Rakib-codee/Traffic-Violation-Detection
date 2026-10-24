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

# Helmet detector mAP

run: `helmet`
weights: `weights/helmet.pt`
dataset: `data/helmet/data.yaml`
seed: 42
device: cuda
val images: 372

| class | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| With Helmet | 0.9517 | 0.6470 |
| Without Helmet | 0.8891 | 0.5258 |
| **all** | **0.9204** | **0.5864** |

# Plate detector mAP

run: `plate`
weights: `weights/plate.pt`
dataset: `data/plates/data.yaml`
seed: 42
device: cuda
val images: 2048

| class | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| License_Plate | 0.9738 | 0.6873 |
| **all** | **0.9738** | **0.6873** |
