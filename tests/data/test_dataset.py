import pandas as pd
import pytest
from hydra import compose, initialize

from q4l.config import GlobalConfig
from q4l.data.dataset import Q4LDataset
from q4l.data.handler import CS_ALL, DK_I

# You can define your own ExperimentConfig here, or create a separate function to load it from a file.
# For this example, I'll create a simple ExperimentConfig instance with minimal values.


@pytest.fixture
def exp_config():
    with initialize(config_path="../config", version_base=None):
        cfg: GlobalConfig = compose(config_name="base")
        return cfg.experiment


def test_init(exp_config):
    dataset = Q4LDataset(exp_config)

    assert dataset.handler is not None
    assert dataset.segments is not None


def test_trade_calendar(exp_config):
    dataset = Q4LDataset(exp_config)

    assert dataset.trade_calendar is not None


def test_prepare(exp_config):
    dataset = Q4LDataset(exp_config)

    partition = "train"
    col_set = CS_ALL
    data_key = DK_I

    data = dataset.prepare(partition, col_set, data_key)

    assert isinstance(data, (pd.DataFrame, list))
    assert len(data) > 0

    if isinstance(data, list):
        assert all(isinstance(df, pd.DataFrame) for df in data)


if __name__ == "__main__":
    pytest.main(["-v", "test_q4ldataset.py"])
