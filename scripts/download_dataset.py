"""Download a Roboflow Universe dataset in YOLOv8 format into data/.

The API key is read from the ROBOFLOW_API_KEY environment variable —
never hard-code it:

    export ROBOFLOW_API_KEY=...        # from https://app.roboflow.com/settings/api
    python scripts/download_dataset.py \\
        --workspace rjacaac1 --project ua-detrac-dataset-10k --version 2 \\
        --format yolov8 --location data/vehicles
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tvd.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--format", default="yolov8")
    parser.add_argument("--location", required=True, help="output dir under data/")
    args = parser.parse_args()

    if not os.environ.get("ROBOFLOW_API_KEY"):
        sys.exit("ROBOFLOW_API_KEY is not set; export it and retry.")

    from roboflow import Roboflow

    from tvd.detect.data import repair_downloaded_yaml

    rf = Roboflow()
    version = rf.workspace(args.workspace).project(args.project).version(args.version)
    dataset = version.download(args.format, location=args.location)
    repair_downloaded_yaml(Path(args.location))
    logger.info("dataset downloaded to %s", dataset.location)


if __name__ == "__main__":
    main()
