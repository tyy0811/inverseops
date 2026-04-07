"""Test Pydantic schemas for serving API."""

from inverseops.serving.schemas import (
    HealthResponse,
    InputAnalysis,
    Metrics,
    ModelInfo,
    RestoreResponse,
)


def test_restore_response_defaults() -> None:
    """RestoreResponse has expected default values."""
    response = RestoreResponse(
        metrics=Metrics(inference_ms=42.0, output_valid=True),
        input_analysis=InputAnalysis(),
    )
    assert response.status == "completed"
    assert response.decision == "good"
    assert response.issues == []


def test_restore_response_roundtrip() -> None:
    """RestoreResponse can be serialized and deserialized."""
    response = RestoreResponse(
        status="completed",
        decision="review",
        metrics=Metrics(inference_ms=100.0, output_valid=True),
        input_analysis=InputAnalysis(
            noise_level_source="estimated",
            noise_level_sigma=15.0,
            in_calibrated_range=True,
        ),
        model_info=ModelInfo(backend="swinir_microscopy_v1", version="0.1.0"),
    )

    data = response.model_dump()
    restored = RestoreResponse.model_validate(data)

    assert restored.status == "completed"
    assert restored.decision == "review"
    assert restored.input_analysis.noise_level_sigma == 15.0
    assert restored.metrics.inference_ms == 100.0


def test_input_analysis_defaults() -> None:
    """InputAnalysis has sensible defaults."""
    analysis = InputAnalysis()
    assert analysis.noise_level_source == "estimated"
    assert analysis.in_calibrated_range is True
    assert analysis.estimation_method is None


def test_model_info_defaults() -> None:
    """ModelInfo has sensible defaults."""
    info = ModelInfo()
    assert info.backend == "swinir_microscopy_v1"
    assert info.version == "0.1.0"


def test_health_response() -> None:
    """HealthResponse works correctly."""
    h = HealthResponse(status="healthy", model_loaded=True)
    assert h.status == "healthy"
    assert h.model_loaded is True
