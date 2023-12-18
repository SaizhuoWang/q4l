import numpy as np
import pandas as pd
import pytest

from q4l.data.backend.compute import MyHXDFComputeBackend


@pytest.fixture
def dummy_data():
    return {
        "a": pd.DataFrame([1, 2, 3]),
        "b": pd.DataFrame([4, 5, 6]),
        "c": pd.DataFrame([7, 8, 9]),
    }


def test_my_hxdf_compute_backend(dummy_data):
    # Test a few operators in MyHXDFComputeBackend
    backend = MyHXDFComputeBackend()

    result = backend.compute(dummy_data, "a+b")
    expected_result = dummy_data["a"] + dummy_data["b"]
    r = result.values.astype(expected_result.values.dtype)
    assert np.all(r == expected_result.values), f"Expected {expected_result}, but got {result}"

    result = backend.compute(dummy_data, "a-b")
    expected_result = dummy_data["a"] - dummy_data["b"]
    r = result.values.astype(expected_result.values.dtype)
    assert np.all(r == expected_result.values), f"Expected {expected_result}, but got {result}"


def test_unsupported_node():
    data = {
        "a": pd.DataFrame([1, 2, 3]),
    }

    backend = MyHXDFComputeBackend()
    with pytest.raises(ValueError, match=r"Unsupported node type"):
        result = backend.compute(data, "lambda x: x + 1")


def test_unsupported_operator():
    data = {
        "a": pd.DataFrame([1, 2, 3]),
        "b": pd.DataFrame([4, 5, 6]),
    }

    backend = MyHXDFComputeBackend()
    with pytest.raises(ValueError, match=r"Unsupported operator type"):
        backend.compute(data, "data['a'] ** data['b']")


if __name__ == "__main__":
    pytest.main()
