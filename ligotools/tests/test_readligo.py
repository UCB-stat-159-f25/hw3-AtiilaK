"""
Unit tests for the readligo.py module.

These tests verify that the `loaddata` function correctly handles
edge cases involving missing or empty files. Both tests ensure that
the function fails gracefully by returning `(None, None, None)`
instead of raising exceptions.
"""

import os, sys
# Add the repository root to Python's import path so ligotools can be imported when running tests directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))



from ligotools import readligo

def test_loaddata_returns_none_for_missing_file(tmp_path):
    """
    Test that `loaddata` returns (None, None, None) when given
    an empty file that exists but has no content.
    """
    fake_file = tmp_path / "fake.hdf5"
    fake_file.write_text("")  # create an empty file
    strain, time, dq = readligo.loaddata(str(fake_file))
    assert strain is None
    assert time is None
    assert dq is None

def test_loaddata_handles_nonexistent_file():
    """
    Test that `loaddata` returns (None, None, None) when the
    file path does not exist.
    """
    strain, time, dq = readligo.loaddata("data/this_file_does_not_exist.hdf5")
    assert strain is None
    assert time is None
    assert dq is None
