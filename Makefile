.PHONY: install test lint serve bench docker onnx

PYTHON     ?= python3.11

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	mypy inverseops tests

serve:
	uvicorn inverseops.serving.app:app --host 0.0.0.0 --port 8000

bench:
	$(PYTHON) scripts/run_latency_bench.py

docker:
	docker-compose -f docker/docker-compose.yaml up --build

onnx:
	$(PYTHON) scripts/run_onnx_export.py

# --- V3 training targets added in Phase 1 Day 1 ---
# train:
#     $(PYTHON) scripts/run_training.py --config configs/w2s_denoise_swinir.yaml
