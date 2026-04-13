"""FastAPI application for image restoration inference.

Endpoints:
    POST /restore        — Denoise an uploaded image (V3 checkpoint registry)
    POST /super_resolve  — 2x super-resolve an uploaded image
    GET  /health         — Service health check
    GET  /metrics        — Prometheus-compatible metrics
"""

from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from inverseops.serving.qc import (
    CALIBRATED_RANGE,
    decide_denoise,
    decide_sr,
    estimate_noise_level,
    validate_input,
    validate_output,
)
from inverseops.serving.schemas import (
    HealthResponse,
    InputAnalysis,
    Metrics,
    ModelInfo,
    RestoreResponse,
)

# Limits
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# Checkpoint registry — keyed by logical model name, resolved relative to
# CHECKPOINT_ROOT. Each entry carries the task, dataset, and build config
# needed to reconstruct the model at startup. Entries MUST match the V3
# training-config shape so build_model() succeeds without re-downloading
# DIV2K pretrained weights. See docs/plans/2026-04-13-v3-serving-layer-migration.md.
CHECKPOINT_ROOT = Path(os.environ.get("CHECKPOINT_ROOT", "./checkpoints"))

CHECKPOINT_REGISTRY: dict[str, dict] = {
    "w2s_denoise_swinir": {
        "path": "w2s_denoise_swinir.pt",
        "task": "denoise",
        "dataset": "w2s",
        "build_config": {
            "model": {"name": "swinir", "pretrained": False},
            "task": "denoise",
            "data": {"dataset": "w2s"},
        },
    },
    "w2s_sr_swinir_2x": {
        "path": "w2s_sr_swinir_2x.pt",
        "task": "sr",
        "dataset": "w2s",
        "build_config": {
            "model": {"name": "swinir", "pretrained": False, "scale": 2},
            "task": "sr",
            "data": {"dataset": "w2s", "scale": 2},
        },
    },
}
DEFAULT_DENOISE_MODEL = "w2s_denoise_swinir"
DEFAULT_SR_MODEL = "w2s_sr_swinir_2x"

# Deprecation shim: first call that passes the legacy `noise_level` form
# param logs a warning. The warning fires once per process, not per request.
_noise_level_deprecation_warned = False

# Prometheus metrics
_registry = CollectorRegistry()
RESTORE_REQUESTS = Counter(
    "restore_requests_total", "Total restore requests", registry=_registry
)
RESTORE_COMPLETED = Counter(
    "restore_completed_total", "Completed restorations", registry=_registry
)
RESTORE_FAILED = Counter(
    "restore_failed_total", "Failed restorations", registry=_registry
)
RESTORE_LATENCY = Histogram(
    "restore_latency_seconds",
    "Inference latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=_registry,
)
QC_DECISION = Counter(
    "qc_decision_total",
    "QC decisions by type",
    ["decision"],
    registry=_registry,
)


def _select_model(request: Request, noise_level: float | None):
    """Select the default denoise model from the registry.

    The `noise_level` param is vestigial in V3 (see Decision 20 / README).
    First use per process logs a deprecation warning via structlog.
    """
    global _noise_level_deprecation_warned
    if noise_level is not None and not _noise_level_deprecation_warned:
        import structlog

        structlog.get_logger().warning(
            "noise_level_param_deprecated",
            message=(
                "The `noise_level` form parameter on /restore is deprecated "
                "in V3 and ignored at request time. It will be removed in V4."
            ),
        )
        _noise_level_deprecation_warned = True

    models = getattr(request.app.state, "models", None)
    if not models or DEFAULT_DENOISE_MODEL not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Default denoise model '{DEFAULT_DENOISE_MODEL}' not loaded",
        )
    return models[DEFAULT_DENOISE_MODEL], DEFAULT_DENOISE_MODEL


async def _read_upload(file: UploadFile) -> bytes:
    """Read upload with size limit to prevent OOM."""
    chunks = []
    total = 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Upload exceeds "
                    f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit"
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(models=None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        models: Optional dict mapping sigma -> model for testing.
                If None, the lifespan loads all 3 sigma checkpoints.
    """
    if models is not None:
        application = FastAPI(
            title="InverseOps",
            description="Microscopy image denoising API",
            version="0.1.0",
        )
        application.state.models = models
    else:

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            from inverseops.models import build_model

            loaded: dict[str, object] = {}
            for logical_name, entry in CHECKPOINT_REGISTRY.items():
                full_path = CHECKPOINT_ROOT / entry["path"]
                if not full_path.is_file():
                    raise RuntimeError(
                        f"Registered checkpoint not found: {full_path}. "
                        f"Set CHECKPOINT_ROOT or place the checkpoint file at "
                        f"the expected path before starting the server."
                    )
                model = build_model(entry["build_config"], device="cpu")
                ckpt = torch.load(full_path, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
                model.eval()
                loaded[logical_name] = model
            app.state.models = loaded
            yield
            app.state.models = None

        application = FastAPI(
            title="InverseOps",
            description="Microscopy image restoration API (V3)",
            version="0.1.0",
            lifespan=lifespan,
        )

    from inverseops.serving.logging import RequestIDMiddleware, configure_logging

    configure_logging()
    application.add_middleware(RequestIDMiddleware)

    _register_routes(application)
    return application


def _register_routes(application: FastAPI) -> None:
    """Register all routes on the given app instance."""

    @application.post("/restore")
    async def restore(
        request: Request,
        file: UploadFile = File(...),
        noise_level: float | None = Form(default=None),
    ) -> Response:
        """Denoise an uploaded image using the default denoise model.

        The noise_level form parameter is vestigial in V3 and ignored
        at request time; first use per process logs a deprecation
        warning. It will be removed in V4.
        """
        RESTORE_REQUESTS.inc()

        # Read with size limit
        try:
            contents = await _read_upload(file)
        except HTTPException:
            RESTORE_FAILED.inc()
            raise
        except Exception:
            RESTORE_FAILED.inc()
            raise HTTPException(
                status_code=400,
                detail="Could not read upload",
            )

        # Decode image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception:
            RESTORE_FAILED.inc()
            raise HTTPException(
                status_code=400,
                detail="Could not decode image file",
            )

        issues = validate_input(image)
        if issues:
            RESTORE_FAILED.inc()
            raise HTTPException(
                status_code=422, detail={"issues": issues}
            )

        # Select default denoise model (noise_level is vestigial in V3)
        model, logical_name = _select_model(request, noise_level)

        # Noise level analysis
        noise_level_estimated = None
        try:
            noise_level_estimated = estimate_noise_level(image)
        except ImportError:
            pass
        except Exception:
            pass

        sigma = (
            noise_level if noise_level is not None
            else noise_level_estimated
        )
        in_range = (
            CALIBRATED_RANGE[0] <= sigma <= CALIBRATED_RANGE[1]
            if sigma is not None
            else True
        )

        input_analysis = InputAnalysis(
            noise_level_source=(
                "user_supplied" if noise_level is not None
                else "estimated"
            ),
            noise_level_sigma=(
                noise_level if noise_level is not None
                else noise_level_estimated
            ),
            estimation_method=(
                None if noise_level is not None
                else (
                    "wavelet_mad"
                    if noise_level_estimated is not None
                    else None
                )
            ),
            in_calibrated_range=in_range,
        )

        # Run inference — get raw output for QC validation
        start = time.perf_counter()
        try:
            raw_output = model.predict_raw(image)
        except Exception as e:
            RESTORE_FAILED.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {e}",
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        RESTORE_LATENCY.observe(elapsed_ms / 1000)

        # Validate raw output BEFORE clipping to uint8
        output_valid, output_issues = validate_output(raw_output)
        issues.extend(output_issues)

        # Clip and convert to PIL image
        output_clipped = np.clip(
            raw_output * 255.0, 0, 255
        ).astype(np.uint8)
        result_image = Image.fromarray(output_clipped, mode="L")

        decision = decide_denoise(
            noise_level, noise_level_estimated, output_valid
        )
        QC_DECISION.labels(decision=decision).inc()
        RESTORE_COMPLETED.inc()

        meta = RestoreResponse(
            status="completed",
            decision=decision,
            metrics=Metrics(
                inference_ms=round(elapsed_ms, 1),
                output_valid=output_valid,
            ),
            input_analysis=input_analysis,
            issues=issues,
            model_info=ModelInfo(
                backend=logical_name,
            ),
        )

        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-Restore-Status": meta.status,
                "X-Restore-Decision": meta.decision,
                "X-Restore-Task": meta.task,
                "X-Restore-Inference-Ms": str(
                    meta.metrics.inference_ms
                ),
                "X-Restore-Metadata": meta.model_dump_json(),
            },
        )

    @application.post("/super_resolve")
    async def super_resolve(
        request: Request,
        file: UploadFile = File(...),
    ) -> Response:
        """Super-resolve an uploaded image (2x).

        Routes to the default SR model in CHECKPOINT_REGISTRY. The model
        was trained on clean LR input (W2S avg400 -> SIM); passing a noisy
        image may produce degraded output but will not fail.

        QC: No noise-level calibration (model trained on clean LR). QC
        checks input validity and output finiteness only. Resolution
        bounds 8x8-2048x2048 match the shared validate_input path used
        by /restore.
        """
        RESTORE_REQUESTS.inc()

        try:
            contents = await _read_upload(file)
        except HTTPException:
            RESTORE_FAILED.inc()
            raise
        except Exception:
            RESTORE_FAILED.inc()
            raise HTTPException(
                status_code=400, detail="Could not read upload"
            )

        try:
            image = Image.open(io.BytesIO(contents))
            image.load()
        except Exception:
            RESTORE_FAILED.inc()
            raise HTTPException(
                status_code=400, detail="Could not decode image file"
            )

        issues = validate_input(image)
        if issues:
            RESTORE_FAILED.inc()
            raise HTTPException(
                status_code=422, detail={"issues": issues}
            )

        models = getattr(request.app.state, "models", None)
        if not models or DEFAULT_SR_MODEL not in models:
            RESTORE_FAILED.inc()
            raise HTTPException(
                status_code=503,
                detail=f"Default SR model '{DEFAULT_SR_MODEL}' not loaded",
            )
        model = models[DEFAULT_SR_MODEL]

        lr_image = image.convert("L") if image.mode != "L" else image
        lr_arr = np.array(lr_image, dtype=np.float32) / 255.0

        from inverseops.data.w2s import W2S_MEAN, W2S_STD

        lr_z = (lr_arr * 255.0 - W2S_MEAN) / W2S_STD

        from inverseops.evaluation.stitching import sliding_window_sr

        start = time.perf_counter()
        try:
            sr_z = sliding_window_sr(model, lr_z, device="cpu", clamp=False)
        except Exception as e:
            RESTORE_FAILED.inc()
            raise HTTPException(
                status_code=500, detail=f"SR inference failed: {e}"
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        RESTORE_LATENCY.observe(elapsed_ms / 1000)

        output_valid, output_issues = validate_output(sr_z)
        issues.extend(output_issues)
        decision = decide_sr(output_valid)
        QC_DECISION.labels(decision=decision).inc()
        RESTORE_COMPLETED.inc()

        sr_denorm = sr_z * W2S_STD + W2S_MEAN
        sr_uint8 = np.clip(sr_denorm, 0, 255).astype(np.uint8)
        out_image = Image.fromarray(sr_uint8, mode="L")

        meta = RestoreResponse(
            status="completed",
            decision=decision,
            task="sr",
            metrics=Metrics(
                inference_ms=round(elapsed_ms, 1),
                output_valid=output_valid,
            ),
            input_analysis=InputAnalysis(
                noise_level_source="estimated",
                noise_level_sigma=None,
                estimation_method=None,
                in_calibrated_range=True,
            ),
            issues=issues,
            model_info=ModelInfo(backend=DEFAULT_SR_MODEL),
        )

        buf = io.BytesIO()
        out_image.save(buf, format="PNG")
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-Restore-Status": meta.status,
                "X-Restore-Decision": meta.decision,
                "X-Restore-Task": meta.task,
                "X-Restore-Inference-Ms": str(meta.metrics.inference_ms),
                "X-Restore-Metadata": meta.model_dump_json(),
            },
        )

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        models = getattr(request.app.state, "models", None)
        loaded = models is not None and DEFAULT_DENOISE_MODEL in models
        return HealthResponse(
            status="healthy" if loaded else "unhealthy",
            model_loaded=loaded,
        )

    @application.get("/metrics")
    async def metrics() -> Response:
        return Response(
            content=generate_latest(_registry),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )


# Default app instance for `uvicorn inverseops.serving.app:app`
app = create_app()
