"""Test Pydantic schemas for serving API."""

from inverseops.serving.schemas import (
    InputAnalysis,
    ModelInfo,
    RestoreRequest,
    RestoreResponse,
)


def test_restore_request_defaults() -> None:
    """RestoreRequest can be instantiated with defaults."""
    request = RestoreRequest()
    assert request.noise_level is None
    assert request.image_path is None


def test_restore_request_with_values() -> None:
    """RestoreRequest accepts provided values."""
    request = RestoreRequest(noise_level=0.5, image_path="/tmp/image.png")
    assert request.noise_level == 0.5
    assert request.image_path == "/tmp/image.png"


def test_restore_response_defaults() -> None:
    """RestoreResponse has expected default values."""
    response = RestoreResponse(
        input_analysis=InputAnalysis(),
        model_info=ModelInfo(),
    )
    assert response.status == "completed"
    assert response.decision == "good"


def test_restore_response_roundtrip() -> None:
    """RestoreResponse can be serialized and deserialized."""
    response = RestoreResponse(
        status="completed",
        decision="review",
        input_analysis=InputAnalysis(
            noise_level_source="estimated",
            noise_level_sigma=15.0,
            in_calibrated_range=True,
        ),
        model_info=ModelInfo(backend="pytorch", version="0.1.0"),
    )

    # Roundtrip through dict
    data = response.model_dump()
    restored = RestoreResponse.model_validate(data)

    assert restored.status == "completed"
    assert restored.decision == "review"
    assert restored.input_analysis.noise_level_sigma == 15.0
    assert restored.model_info.backend == "pytorch"


def test_input_analysis_defaults() -> None:
    """InputAnalysis has sensible defaults."""
    analysis = InputAnalysis()
    assert analysis.noise_level_source == "estimated"
    assert analysis.in_calibrated_range is True


def test_model_info_defaults() -> None:
    """ModelInfo has sensible defaults."""
    info = ModelInfo()
    assert info.backend == "pytorch"
    assert info.version == "0.1.0"
