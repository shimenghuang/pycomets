# To import module functions without installing the module first
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
from pycomets.utils import _safe_squeeze

## Test _safe_squeeze

@pytest.mark.parametrize("x, axis, expected", [
    (np.array([[1,2]]), 0, (2,)),
    (np.array([1,2]), 0, (2,)),
    (np.array([[1],[2]]), 0, (2,1)),
    (np.array([[1,2]]), 1, (1,2)),
    (np.array([[1],[2]]), 1, (2,)),
    (np.array([[1,2],[3,4],[5,6]]), 0, (3,2)),
    (np.array([[1,2],[3,4],[5,6]]), 1, (3,2))
])
def test_safe_squeeze_normal(x, axis, expected):
    xx = _safe_squeeze(x, axis=axis)
    assert xx.shape == expected

@pytest.mark.parametrize("x, axis, err_type, err_text", [
    (np.array([[1],[2]]), 2, IndexError, r"axis 2 exceeds the dimension of arr 2"),
    (np.array([[1],[2]]), -1, ValueError, r"axis can only be nonnegative integers, got value -1")
])
def test_safe_squeeze_exception(x, axis, err_type, err_text):
    with pytest.raises(err_type, match=err_text):
        _safe_squeeze(x, axis)

@pytest.mark.parametrize("x, axis, expected", [
    (np.array([[1,2]]), 0, np.array([[1,2]])), # shape changed and should return a copy
    (np.array([1,2]), 0, np.array([2,2])) # shape did not change and should return a view
])
def test_safe_squeeze_memory(x, axis, expected):
    xx = _safe_squeeze(x, axis=axis) 
    xx[0] = 2
    np.testing.assert_array_equal(x, expected)




