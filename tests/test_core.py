"""Tests for core FracTime functionality."""

import numpy as np
import pytest
from fractime.core import *

def test_sample_data():
    """Test that sample data generation works."""
    data = np.random.randn(100)
    assert len(data) == 100
    assert isinstance(data, np.ndarray)
