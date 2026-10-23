VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup lint format test run clean

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

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache src/*.egg-info
