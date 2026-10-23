from pathlib import Path

import pytest
import yaml

from tvd.detect.data import validate_dataset, validate_label_file


def _make_dataset(root: Path, labels: dict[str, str]) -> Path:
    """Create a minimal YOLO dataset; labels maps split/filename -> content."""
    names = ["car", "bus"]
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "images" / "train" / "a.jpg").write_bytes(b"fake")
    (root / "images" / "val" / "b.jpg").write_bytes(b"fake")
    for rel, content in labels.items():
        (root / "labels" / rel).write_text(content)
    dataset_yaml = root / "data.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val", "names": names}
        )
    )
    return dataset_yaml


def test_validate_dataset_ok(tmp_path):
    dataset_yaml = _make_dataset(
        tmp_path,
        {
            "train/a.txt": "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n",
            "val/b.txt": "0 0.5 0.5 0.3 0.3\n",
        },
    )
    info = validate_dataset(dataset_yaml)
    assert info.names == ["car", "bus"]
    assert info.n_train == 1
    assert info.n_val == 1


def test_label_out_of_range_class(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("5 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="out of range"):
        validate_label_file(f, n_classes=2)


def test_label_unnormalized_coords(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("0 50 50 20 20\n")
    with pytest.raises(ValueError, match="normalized"):
        validate_label_file(f, n_classes=2)


def test_label_wrong_field_count(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("0 0.5 0.5 0.2\n")
    with pytest.raises(ValueError, match="5 fields"):
        validate_label_file(f, n_classes=2)


def test_label_zero_area_box(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("0 0.5 0.5 0.0 0.2\n")
    with pytest.raises(ValueError, match="positive"):
        validate_label_file(f, n_classes=2)


def test_missing_split_dir(tmp_path):
    dataset_yaml = _make_dataset(tmp_path, {"train/a.txt": "0 0.5 0.5 0.2 0.2\n"})
    (tmp_path / "labels" / "val").rmdir()
    with pytest.raises(ValueError, match="labels dir"):
        validate_dataset(dataset_yaml)


def test_missing_yaml_key(tmp_path):
    bad = tmp_path / "data.yaml"
    bad.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "images/train"}))
    with pytest.raises(ValueError, match="missing required key"):
        validate_dataset(bad)
