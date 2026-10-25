# Data sources and licenses

No datasets are committed to this repository. `data/` is gitignored.
Record every dataset used below before downloading it.

| Dataset | Purpose | License | Source | Status |
|---|---|---|---|---|
| UA-DETRAC-DATASET-10K v2 (Roboflow) | vehicle detection fine-tuning | CC BY 4.0 on Roboflow; underlying UA-DETRAC is research-use — cite the DETRAC paper (Wen et al., ECCVW 2020) | https://universe.roboflow.com/rjacaac1/ua-detrac-dataset-10k (via `scripts/download_dataset.py`) | downloaded to data/vehicles |
| coco8 | pipeline smoke tests only (8 COCO images) | COCO terms (images: Flickr terms; annotations CC BY 4.0) | https://github.com/ultralytics/assets/releases | downloaded to data/coco8 |
| Helmet Detection_YOLOv8 v3 (Roboflow) | helmet non-use detection (Phase 6) | CC BY 4.0 | https://universe.roboflow.com/learning-evidence/helmet-detection_yolov8 (via `scripts/download_dataset.py`) | downloaded to data/helmet |
| License Plate Recognition v13 (Roboflow Universe Projects) | optional ANPR plate detection (Phase 6) | CC BY 4.0 | https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e | downloaded to data/plates |
| UA-DETRAC (official, incl. videos + track annotations) | Phase 2 tracking eval (HOTA/IDF1) | research use; request form on site | https://detrac-db.rit.albany.edu/ | requested by repo owner; pending |
| BDD100K | driving scenes | BSD-style; verify terms | https://bdd-data.berkeley.edu/ | not downloaded |
| AI City Challenge | traffic analytics | per-track terms; verify | https://www.aicitychallenge.org/ | not downloaded |
| CCPD (optional) | plate detection | research use; verify | https://github.com/detectRecog/CCPD | not downloaded |

Own footage: only from public/permitted sources; document permission here before use.

| detrac_mvi20012 sample clip (from data/vehicles export) | sample/demo + event-level eval set | CC BY 4.0 (same export as above) | data/samples/ | in use |

The event-level evaluation set (small, hand-labeled) lives in `data/events/`
(gitignored) with the committed manifest `configs/event_set.yaml`. Derived
clips are constructed from the UA-DETRAC sample by `scripts/build_event_set.py`
(e.g. reversed playback => wrong-way ground truth by construction); labeling
method and caveats are documented in the manifest. Run it with
`python scripts/eval_events.py`.
