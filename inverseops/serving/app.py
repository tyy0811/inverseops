"""FastAPI application for image restoration inference.

Endpoints:
    POST /restore  — Denoise an uploaded image
    GET  /health   — Service health check
    GET  /metrics  — Prometheus-compatible metrics
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from inverseops.serving.qc import (
    CALIBRATED_RANGE,
    decide,
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

# Supported sigma levels (matching pretrained checkpoints)
SUPPORTED_SIGMAS = (15, 25, 50)
DEFAULT_SIGMA = 25

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
    """Select the model checkpoint closest to the requested sigma."""
    models = getattr(request.app.state, "models", None)
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if noise_level is not None:
        sigma = min(
            models.keys(), key=lambda s: abs(s - noise_level)
        )
    else:
        sigma = DEFAULT_SIGMA

    model = models[sigma]
    if not model.is_loaded():
        raise HTTPException(
            status_code=503, detail=f"Model sigma={sigma} not loaded"
        )
    return model, sigma


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
            from inverseops.models.swinir import SwinIRBaseline

            loaded = {}
            for sigma in SUPPORTED_SIGMAS:
                m = SwinIRBaseline(
                    noise_level=sigma,  # type: ignore[arg-type]
                    device="cpu",
                )
                m.load()
                loaded[sigma] = m
            app.state.models = loaded
            yield
            app.state.models = None

        application = FastAPI(
            title="InverseOps",
            description="Microscopy image denoising API",
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
        """Denoise an uploaded image.

        The noise_level parameter selects the checkpoint trained for
        that sigma. If omitted, defaults to sigma=25.
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

        # Select model based on noise_level
        model, selected_sigma = _select_model(
            request, noise_level
        )

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

        decision = decide(
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
                backend=f"swinir_sigma{selected_sigma}",
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
                "X-Restore-Inference-Ms": str(
                    meta.metrics.inference_ms
                ),
                "X-Restore-Metadata": meta.model_dump_json(),
            },
        )

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        models = getattr(request.app.state, "models", None)
        loaded = models is not None and all(
            m.is_loaded() for m in models.values()
        )
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
