import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from q4l.data.backend.storage import (
    FactorHubBackend,
    FactorRecord,
    FileSystemBackend,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def dummy_factor_data():
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    data = np.ones(shape=(len(dates), 3))
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C"])


def test_file_system_backend(temp_dir, dummy_factor_data):
    """Test FileSystemBackend functionality."""
    fs_backend = FileSystemBackend(root_dir=temp_dir, frequency="daily")

    # Save a factor CSV file in the temporary directory
    factor_name = "test_factor"
    factor_path = os.path.join(temp_dir, "daily", f"{factor_name}.csv")
    os.makedirs(os.path.dirname(factor_path))
    dummy_factor_data.to_csv(factor_path)

    # Test get_factor
    result = fs_backend.get_factor(factor_name)
    result.index = pd.to_datetime(result.index)
    assert result.equals(dummy_factor_data), "Factor data does not match."


def test_factor_hub_backend(temp_dir, dummy_factor_data, monkeypatch):
    """Test FactorHubBackend functionality."""
    logging.getLogger("q4l").setLevel(logging.DEBUG)

    def mock_get_factor(*args, **kwargs):
        return dummy_factor_data

    monkeypatch.setattr(FactorHubBackend, "download_factor", mock_get_factor)

    factor_hub_backend = FactorHubBackend(factor_cache_dir=temp_dir)

    # Test get_factor
    factor_name = "act_vol_1D"
    factor_hub_backend.get_factor(factor_name)

    # Test mem cache
    factor_hub_backend.get_factor(factor_name)

    # Test disk cache
    FactorHubBackend(factor_cache_dir=temp_dir).get_factor(factor_name)

    # Test another factor
    factor_hub_backend.get_factor("act_volume_1min")

    # Test create_toc and update_toc
    factor_hub_backend.create_toc()
    factor_hub_backend.update_toc(
        {
            "factor_name": factor_name,
            "path": os.path.join(temp_dir, f"{factor_name}.csv"),
            "start_time": "20200101",
            "end_time": "20201231",
            "file_type": "csv",
        }
    )
    assert factor_name in factor_hub_backend.factor_names, "Failed to update the factor in TOC."

    # Test check_time
    record = FactorRecord(
        factor_name=factor_name,
        path=os.path.join(temp_dir, f"{factor_name}.csv"),
        start_time="20200101",
        end_time="20201231",
        file_type="csv",
    )


# def test_factor_hub_backend_time_range(
#     temp_dir, dummy_factor_data, monkeypatch
# ):
#     """
#     Test FactorHubBackend functionality with different time ranges.
#     """

#     def mock_get_factor(*args, **kwargs):
#         return dummy_factor_data

#     monkeypatch.setattr(FactorHubBackend, "download_factor", mock_get_factor)

#     factor_hub_backend = FactorHubBackend(factor_cache_dir=temp_dir)

#     # Test get_factor with different time ranges
#     factor_name = "test_factor"

#     # Test with start and end times
#     result = factor_hub_backend.get_factor(
#         factor_name, "20200201", "20200301"
#     )
#     expected_result = dummy_factor_data.loc["20200201":"20200301"]
#     assert result.equals(
#         expected_result
#     ), f"Factor data does not match, expected:\n{expected_result}\n but got:\n{result}"

#     # Test with only start time
#     result = factor_hub_backend.get_factor(factor_name, "20200201", None)
#     expected_result = dummy_factor_data.loc["20200201":]
#     assert result.equals(
#         expected_result
#     ), f"Factor data does not match, expected:\n{expected_result}\n but got:\n{result}"

#     # Test with only end time
#     result = factor_hub_backend.get_factor(factor_name, None, "20200301")
#     expected_result = dummy_factor_data.loc[:"20200301"]
#     assert result.equals(
#         expected_result
#     ), f"Factor data does not match, expected:\n{expected_result}\n but got:\n{result}"

#     # Test with no start and end times
#     result = factor_hub_backend.get_factor(factor_name, None, None)
#     assert result.equals(
#         dummy_factor_data
#     ), f"Factor data does not match, expected:\n{dummy_factor_data}\n but got:\n{result}"
