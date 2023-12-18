import numpy as np
import pandas as pd
import pytest
import torch.utils.data.dataloader as dataloader
from omegaconf import DictConfig
from test_handler import data_handler_config_shm

from q4l.config import ModuleConfig, TimeInterval
from q4l.data.handler import Q4LDataHandler
from q4l.data.sampler import TCDataSampler, reshape_data


@pytest.fixture
def filter_config():
    filter1 = ModuleConfig(
        name="TSNaNFilter",
        module_path="q4l.data.filter",
        kwargs={"nan_ratio": 0.2},
    )
    return [DictConfig(filter1)]


T, N, W, F = 100, 40, 5, 20


def create_sample_dataframe(nan_ratio=0.25):
    # Generate T random dates
    start_date = pd.Timestamp("2022-01-01")
    dates = pd.date_range(start=start_date, periods=T, freq="D")

    # Generate N random ticker names
    tickers = [
        "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=4)) for _ in range(N)
    ]

    index = pd.MultiIndex.from_product([dates, tickers], names=["datetime", "ticker"])
    columns = pd.MultiIndex.from_tuples(
        [*[("feature", f"x_{i+1}") for i in range(F - 1)], ("label", "y")]
    )
    data = pd.DataFrame(np.random.randn(T * N, F), index=index, columns=columns)

    # Flatten the DataFrame
    flat_data = data.stack().stack()

    # Calculate the number of elements to replace with NaN
    num_elements = flat_data.size
    num_nan = int(num_elements * nan_ratio)

    # Randomly select indices to replace with NaN
    nan_indices = np.random.choice(num_elements, size=num_nan, replace=False)

    # Replace the selected indices with NaN
    flat_data.iloc[nan_indices] = np.nan

    # Unstack the DataFrame to get back to the original shape
    data_with_nan = flat_data.unstack().unstack()

    data = pd.DataFrame(data_with_nan, index=index, columns=columns)

    return data


# Create a DataFrame with a 10% NaN ratio
sample_data = create_sample_dataframe(nan_ratio=0.1)
print(sample_data)

# Test for reshape_data function
def test_reshape_data():
    data = create_sample_dataframe()
    reshaped_data = reshape_data(data, W)
    assert reshaped_data.shape == ((T - W + 1), N, W, F)


def test_reshape_data_with_shm():
    handler_1 = Q4LDataHandler(**data_handler_config_shm())

    def worker():
        handler_2 = Q4LDataHandler(**data_handler_config_shm())
        df = handler_2.fetch(TimeInterval(start="20190401", end="20190601"))
        arr_reshaped = reshape_data(df, W)
        return arr_reshaped

    df = handler_1.fetch(TimeInterval(start="20190401", end="20190601"))
    arr_expected = reshape_data(df, W)
    assert np.array_equal(worker(), arr_expected)


# Test the sampler initialization
def test_sampler_init_time_series(filter_config):
    data = create_sample_dataframe()
    sampler = TCDataSampler(data, filter_config, window_size=W, sampling_mode="T")
    assert len(sampler) < (T - W + 1) * N


def test_sampler_init_cross_sectional(filter_config):
    data = create_sample_dataframe()
    sampler = TCDataSampler(data, filter_config, window_size=W, sampling_mode="C")
    assert len(sampler) == T - W + 1


# Test the sampler getitem method
def test_sampler_getitem_time_series(filter_config):
    data = create_sample_dataframe()
    sampler = TCDataSampler(data, filters=filter_config, window_size=W, sampling_mode="T")

    with pytest.raises(TypeError):
        sampler["string_index"]

    sample = sampler[0]
    assert "label" in sample
    assert "x" in sample
    assert "y" in sample
    assert sample["x"].shape == (W, F - 1)
    assert sample["y"].shape == (W, 1)


def test_sampler_getitem_cross_sectional(filter_config):
    data = create_sample_dataframe()
    sampler = TCDataSampler(data, filters=filter_config, window_size=W, sampling_mode="C")

    with pytest.raises(TypeError):
        sampler["string_index"]

    sample = sampler[0]
    assert "label" in sample
    assert "x" in sample
    assert "y" in sample
    assert len(sample["x"].shape) == 3
    assert len(sample["y"].shape) == 3


# Test the sampler collate method
def test_sampler_collate_ts(filter_config):
    data = create_sample_dataframe()
    sampler = TCDataSampler(data, filters=filter_config, window_size=W, sampling_mode="T")
    B = 5
    batch = [sampler[i] for i in range(B)]
    collated_batch = sampler.collate(batch)

    assert "label" in collated_batch
    assert "x" in collated_batch
    assert "y" in collated_batch
    assert collated_batch["x"].shape == (B, W, F - 1)
    assert collated_batch["y"].shape == (B, W, 1)


def test_sampler_collate_cs(filter_config):
    data = create_sample_dataframe()
    sampler = TCDataSampler(data, filters=filter_config, window_size=W, sampling_mode="C")
    loader = dataloader.DataLoader(sampler, batch_size=5, collate_fn=sampler.collate, shuffle=True)

    collated_batch = next(iter(loader))

    assert "label" in collated_batch
    assert "x" in collated_batch
    assert "y" in collated_batch
    assert "padding_mask" in collated_batch
