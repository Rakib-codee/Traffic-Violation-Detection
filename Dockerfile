# Traffic Violation Detection — CPU demo/test image (Phase 7).
# Build:  docker build -t tvd .
# Run the end-to-end demo (default CMD):
#   docker run --rm tvd
# Run the test suite instead:
#   docker run --rm tvd python -m pytest -q
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime libs needed by opencv-python-headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch/torchvision first (much smaller than the default
# CUDA wheels), then the package itself; the pinned versions in
# pyproject.toml are already satisfied, so pip keeps the CPU builds.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install torch==2.14.1 torchvision==0.29.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[dev,track]"

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

# End-to-end demo: generate the synthetic video, run the pipeline with COCO
# yolov8n weights (downloaded on first run) and extensions disabled, then the
# rule-engine eval. All outputs land under results/ inside the container.
CMD python scripts/make_sample_video.py --config configs/system.yaml \
    && python scripts/run_pipeline.py \
        --video data/samples/synthetic_intersection.mp4 \
        --scene configs/scene.synthetic.yaml \
        --weights yolov8n.pt \
        --extensions-config configs/extensions.demo.yaml \
    && python scripts/eval_rules.py
