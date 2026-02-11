"""Example tests for ml_intro."""

import pytest

from ml_intro import __version__


def test_version():
    """Test that the package version is defined."""
    assert __version__ == "0.1.0"
