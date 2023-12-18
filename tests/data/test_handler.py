from typing import Dict

import pandas as pd
import pytest
from test_loader import create_mock_loader_config

from q4l.config import PreprocessorConfig, TimeInterval
from q4l.data.handler import Q4LDataHandler

# Sample data for testing
instruments = "nikkei225"
start_time = "20190101"
end_time = "20200101"
fit_start_time = "20190101"
fit_end_time = "20190901"

# @pytest.fixture
def data_handler_config():
    return dict(
        instruments="nikkei225",
        start_time="20190101",
        end_time="20200101",
        fit_start_time="20190101",
        fit_end_time="20190901",
        loader_config=create_mock_loader_config(),
        preprocessor_config=PreprocessorConfig(
            **{
                "shared": [],
                "learn": [],
                "infer": [],
            }
        ),
        shm_name="test_shm_q4l_pytest",
        use_shm=False,
        cache_dir="/tmp/q4l_pytest_cache",
    )


def data_handler_config_shm() -> Dict:
    cfg = data_handler_config()
    cfg["use_shm"] = True
    return cfg


@pytest.fixture
def handler_orig():
    return Q4LDataHandler(**data_handler_config())


@pytest.fixture
def handler_shm():
    return Q4LDataHandler(**data_handler_config_shm())


def test_init(handler_orig):
    assert handler_orig.instruments == instruments
    assert handler_orig.start_time == start_time
    assert handler_orig.end_time == end_time
    assert handler_orig.fit_start_time == fit_start_time
    assert handler_orig.fit_end_time == fit_end_time


def test_fetch(handler_orig):
    data = handler_orig.fetch(
        segment=TimeInterval("20190201", "20190523"),
        col_set="example_group_2",
        data_key="infer",
    )
    assert isinstance(data, pd.DataFrame)


def test_shm(handler_shm):
    print(handler_shm)
    # Now build another shm handler to peek whether shm is ready
    another_handler = Q4LDataHandler(**data_handler_config_shm(), init_data=False)
    assert another_handler.shm_ready
    another_handler.setup_data()
    assert another_handler.raw_data.equals(handler_shm._data)
