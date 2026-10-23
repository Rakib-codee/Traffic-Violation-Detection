"""Convert UA-DETRAC annotation XML to TrackEval MOTChallenge ground truth.

UA-DETRAC layout (after you obtain it from https://detrac-db.rit.albany.edu/):

    <annotations-dir>/MVI_20011.xml     # one XML per video sequence
    <videos-dir>/MVI_20011.avi

Output (TrackEval MOTChallenge layout):

    <out-dir>/MVI_20011/gt/gt.txt
    <out-dir>/MVI_20011/seqinfo.ini

GT rows: frame,id,x,y,w,h,conf,class,visibility
All vehicle types (sedan/suv/taxi/van/bus/truck/...) map to MOT class 3
('car'); 'others' is written with conf=0 so TrackEval treats it as a
distractor and excludes it from scoring.
"""

from __future__ import annotations

import argparse
import configparser
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from tvd.utils import get_logger

logger = get_logger(__name__)

VEHICLE_TYPES = {"sedan", "suv", "taxi", "police", "van", "bus", "truck"}
# trackeval 1.3.0 MotChallenge2DBox only scores the 'pedestrian' slot for
# custom benchmarks; class id 1 is that slot and stands for "vehicle" here.
MOT_VEHICLE_CLASS = 1


def convert_sequence(annotation_xml: Path, video_path: Path, out_dir: Path) -> Path:
    seq_name = annotation_xml.stem
    out_dir = Path(out_dir) / seq_name
    gt_dir = out_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    seq_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    tree = ET.parse(annotation_xml)
    rows: list[str] = []
    for frame in tree.getroot().iter("frame"):
        frame_num = int(frame.attrib["num"])
        for target in frame.iter("target"):
            target_id = int(target.attrib["id"])
            box = target.find("box")
            attr = target.find("attribute")
            vehicle_type = attr.attrib.get("vehicle_type", "").lower() if attr is not None else ""
            left = float(box.attrib["left"])
            top = float(box.attrib["top"])
            width = float(box.attrib["width"])
            height = float(box.attrib["height"])
            conf = 1 if vehicle_type in VEHICLE_TYPES else 0  # 'others' -> distractor
            rows.append(
                f"{frame_num},{target_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},"
                f"{conf},{MOT_VEHICLE_CLASS},1.0"
            )

    (gt_dir / "gt.txt").write_text("\n".join(rows) + "\n")

    ini = configparser.ConfigParser()
    ini["Sequence"] = {
        "name": seq_name,
        "imDir": "img1",
        "frameRate": int(fps),
        "seqLength": seq_length,
        "imWidth": im_width,
        "imHeight": im_height,
        "imExt": ".jpg",
    }
    with (out_dir / "seqinfo.ini").open("w") as f:
        ini.write(f)

    logger.info(
        "%s: %d gt rows, %d frames, %dx%d", seq_name, len(rows), seq_length, im_width, im_height
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", required=True, type=Path)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--sequences", nargs="*", default=None, help="limit to these sequence names"
    )
    args = parser.parse_args()

    xmls = sorted(args.annotations_dir.glob("*.xml"))
    if args.sequences:
        wanted = set(args.sequences)
        xmls = [x for x in xmls if x.stem in wanted]
    if not xmls:
        raise SystemExit(f"no annotation XMLs found in {args.annotations_dir}")

    for xml in xmls:
        video = args.videos_dir / f"{xml.stem}.avi"
        if not video.exists():
            logger.warning("skipping %s: video not found at %s", xml.stem, video)
            continue
        convert_sequence(xml, video, args.out_dir)


if __name__ == "__main__":
    main()
