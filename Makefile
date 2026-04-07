.PHONY: install test lint train eval serve samples bench docker onnx \
	train-smoke train-short train-day5 compare-sigma50 compare-full

PYTHON     ?= python3.11
RUN_NAME   ?= swinir_fmd_denoise_sigma15_25_50_v1
CHECKPOINT ?= outputs/training/checkpoints/best.pt
MICRO      ?= data/raw/fmd/Confocal_FISH/gt
NATURAL    ?= data/raw/natural

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	mypy inverseops tests

train:
	$(PYTHON) scripts/run_training.py --config configs/denoise_swinir.yaml

eval:
	# Day 3 baseline evaluation
	# Usage: make eval MICRO=data/raw/fmd NATURAL=data/raw/natural
	@if [ -z "$(MICRO)" ]; then \
		echo "Usage: make eval MICRO=<microscopy_root> [NATURAL=<natural_root>]"; \
		echo "Example: make eval MICRO=data/raw/fmd NATURAL=data/raw/natural"; \
	else \
		$(PYTHON) scripts/run_evaluation.py \
			--microscopy-root $(MICRO) \
			$${NATURAL:+--natural-root $(NATURAL)}; \
	fi

serve:
	uvicorn inverseops.serving.app:app --host 0.0.0.0 --port 8000

bench:
	$(PYTHON) scripts/run_latency_bench.py

docker:
	docker-compose -f docker/docker-compose.yaml up --build

onnx:
	$(PYTHON) scripts/run_onnx_export.py

samples:
	$(PYTHON) scripts/save_sample_degradations.py $${DATA_ROOT:-data/raw/fmd}

# --- Day 4+ workflows ---

train-smoke:
	$(PYTHON) scripts/run_training.py \
		--config configs/denoise_swinir.yaml \
		--epochs 2 \
		--limit-train-samples 4 \
		--limit-val-samples 2 \
		--batch-size 1 \
		--no-wandb

train-short:
	$(PYTHON) scripts/run_training.py \
		--config configs/denoise_swinir.yaml \
		--epochs 5 \
		--limit-train-samples 32 \
		--limit-val-samples 8 \
		--batch-size 1 \
		--no-wandb

train-day5:
	$(PYTHON) scripts/run_training.py \
		--config configs/denoise_swinir.yaml \
		--run-name $(RUN_NAME)

compare-sigma50:
	$(PYTHON) scripts/run_evaluation.py \
		--microscopy-root $(MICRO) \
		--natural-root $(NATURAL) \
		--single-sigma 50 \
		--checkpoint $(CHECKPOINT) \
		--model-mode finetuned \
		--output-csv artifacts/compare_finetuned/finetuned_sigma50_metrics.csv \
		--baseline-csv artifacts/baseline/baseline_summary.csv \
		--no-wandb \
		--allow-missing-datasets

compare-full:
	$(PYTHON) scripts/run_evaluation.py \
		--microscopy-root $(MICRO) \
		--natural-root $(NATURAL) \
		--checkpoint $(CHECKPOINT) \
		--model-mode finetuned \
		--output-csv artifacts/compare_finetuned/finetuned_full_metrics.csv \
		--baseline-csv artifacts/baseline/baseline_summary.csv \
		--no-wandb \
		--allow-missing-datasets
