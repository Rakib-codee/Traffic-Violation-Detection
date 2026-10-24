VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup lint format test run demo clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/ruff format --check src tests scripts

format:
	$(VENV)/bin/ruff check --fix src tests scripts
	$(VENV)/bin/ruff format src tests scripts

test:
	$(PYTHON) -m pytest tests/ -v

run:
	$(PYTHON) scripts/make_sample_video.py --config configs/system.yaml

# Fresh-clone demo: no trained weights or datasets needed. Uses the COCO
# yolov8n.pt weights (auto-downloaded by ultralytics) and demo configs with
# all extension modules disabled. Writes outputs under results/.
demo: run
	$(PYTHON) scripts/run_pipeline.py \
		--video data/samples/synthetic_intersection.mp4 \
		--scene configs/scene.synthetic.yaml \
		--weights yolov8n.pt \
		--extensions-config configs/extensions.demo.yaml
	$(PYTHON) scripts/eval_rules.py

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache src/*.egg-info
