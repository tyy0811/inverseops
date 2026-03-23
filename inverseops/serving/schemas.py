"""Pydantic schemas for the serving API.

Defines request/response contracts for the /restore endpoint.
FastAPI integration will be added in a future iteration.
"""

from pydantic import BaseModel


class RestoreRequest(BaseModel):
    """Request schema for the /restore endpoint."""

    noise_level: float | None = None
    """Optional noise level hint. If not provided, will be estimated."""

    image_path: str | None = None
    """Path to the image to restore."""


class InputAnalysis(BaseModel):
    """Analysis of the input image characteristics."""

    noise_level_source: str = "estimated"
    """How the noise level was determined: 'provided' or 'estimated'."""

    noise_level_sigma: float | None = None
    """Estimated or provided noise level (sigma)."""

    in_calibrated_range: bool = True
    """Whether the noise level falls within the model's calibrated range."""


class ModelInfo(BaseModel):
    """Information about the model used for restoration."""

    backend: str = "pytorch"
    """Inference backend (e.g., 'pytorch', 'onnx', 'tensorrt')."""

    version: str = "0.1.0"
    """Model version string."""


class RestoreResponse(BaseModel):
    """Response schema for the /restore endpoint.

    Returned after processing a restore request.
    """

    status: str = "completed"
    """Request completion state: 'completed', 'pending', or 'failed'."""

    decision: str = "good"
    """Quality decision: 'good', 'review', or 'out_of_range'."""

    input_analysis: InputAnalysis
    """Analysis of the input image."""

    model_info: ModelInfo
    """Information about the model used."""
