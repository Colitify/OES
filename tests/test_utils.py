import pytest
import json
import tempfile
from pathlib import Path


def test_get_git_sha():
    """get_git_sha returns a string."""
    from src.utils import get_git_sha
    sha = get_git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0
