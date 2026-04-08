"""Tests for experiment tracking utilities."""

from inverseops.tracking.experiment import VALID_TAGS, make_run_name


def test_valid_tags_has_tasks():
    """Tag constants include task values."""
    assert "denoising" in VALID_TAGS["task"]
    assert "sr" in VALID_TAGS["task"]


def test_valid_tags_has_models():
    """Tag constants include model values."""
    assert "swinir" in VALID_TAGS["model"]
    assert "nafnet" in VALID_TAGS["model"]


def test_valid_tags_has_noise():
    """Tag constants include noise type values."""
    assert "synthetic" in VALID_TAGS["noise"]
    assert "real" in VALID_TAGS["noise"]


def test_make_run_name_with_sha():
    """Run name includes git SHA when available."""
    name = make_run_name("test_run", git_sha="abc1234def")
    assert name == "test_run_abc1234"


def test_make_run_name_without_sha():
    """Run name has timestamp fallback when no git SHA."""
    name = make_run_name("test_run", git_sha=None)
    assert name.startswith("test_run_")
    assert len(name) > len("test_run_")
