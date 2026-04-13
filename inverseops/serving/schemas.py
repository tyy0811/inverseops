"""Pydantic schemas for the serving API.

Defines request/response contracts for the /restore and /super_resolve endpoints.
"""

from typing import Literal

from pydantic import BaseModel


class InputAnalysis(BaseModel):
    """Analysis of the input image characteristics."""

    noise_level_source: str = "estimated"
    """How the noise level was determined: 'user_supplied' or 'estimated'."""

    noise_level_sigma: float | None = None
    """Estimated or provided noise level (sigma)."""

    estimation_method: str | None = None
    """Method used for estimation (e.g., 'wavelet_mad'). None if user-supplied."""

    in_calibrated_range: bool = True
    """Whether the noise level falls within the model's calibrated range."""


class Metrics(BaseModel):
    """Inference performance metrics."""

    inference_ms: float
    """Inference time in milliseconds."""

    output_valid: bool
    """Whether the output passed validation checks."""


class ModelInfo(BaseModel):
    """Information about the model used for restoration."""

    backend: str = "swinir_microscopy_v1"
    """Model identifier."""

    version: str = "0.1.0"
    """Model version string."""


class RestoreResponse(BaseModel):
    """Response metadata for the /restore and /super_resolve endpoints.

    Returned as JSON alongside the restored image file.
    """

    status: str = "completed"
    """Request completion state: 'completed' or 'failed'."""

    decision: str = "good"
    """Quality decision: 'good', 'review', or 'out_of_range'."""

    task: Literal["denoise", "sr"] = "denoise"
    """Task that was performed. 'denoise' for /restore, 'sr' for /super_resolve."""

    metrics: Metrics
    """Inference performance."""

    input_analysis: InputAnalysis
    """Analysis of the input image."""

    issues: list[str] = []
    """Any issues encountered during processing."""

    model_info: ModelInfo = ModelInfo()
    """Information about the model used."""


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str
    """Service status: 'healthy' or 'unhealthy'."""

    model_loaded: bool
    """Whether the model is loaded and ready."""

    version: str = "0.1.0"
