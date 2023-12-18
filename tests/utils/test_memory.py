import numpy as np
import pandas as pd
import pytest

from q4l.utils.memory import (
    MmapSharedMemoryNumpyArray,
    SharedMemoryDataFrame,
    SharedMemoryManager,
    monitor_memory_usage,
)

# Change this to the path of your module
from scripts.collector import get_tickers


@pytest.fixture
def large_numpy_array():
    return np.random.rand(10000, 10000)


@pytest.fixture
def large_dataframe(large_numpy_array):
    tickers = get_tickers(
        region="us", data_dir="/wsz/Data/my_data_dir", read_existing=True
    )[:100]
    ticks = pd.date_range("2019-01-01", "2019-12-31", freq="1min")[:100]
    pd_1 = pd.DataFrame(
        large_numpy_array[:100, :100], index=ticks, columns=tickers
    ).unstack()
    pd_2 = pd_1 + 1
    pd_3 = pd_1 + 2
    pd_full = pd.concat([pd_1, pd_2, pd_3], axis=1)
    pd_full.columns = ["open", "high", "low"]
    return pd_full


def test_mmap_shared_memory_numpy_array(large_numpy_array):
    name = "test_mmap_shared_memory_numpy_array"

    shm_arr = MmapSharedMemoryNumpyArray(name, large_numpy_array)
    loaded_array = shm_arr.to_numpy()

    np.testing.assert_array_equal(large_numpy_array, loaded_array)

    shm_arr.close()
    shm_arr.unlink()


def test_shared_memory_data_frame(large_dataframe):
    name = "test_shared_memory_data_frame"

    shm_df = SharedMemoryDataFrame(name, large_dataframe)
    loaded_dataframe = shm_df.to_dataframe()

    pd.testing.assert_frame_equal(large_dataframe, loaded_dataframe)

    shm_df.close()
    shm_df.unlink()


def test_shared_memory_cleanup():
    name = "test_shared_memory_cleanup"
    shm_arr = MmapSharedMemoryNumpyArray(name, np.array([1, 2, 3]))
    shm_arr.close()

    # Register a shared memory file with an invalid file path
    file_path = f"/dev/shm/{name}_invalid"
    SharedMemoryManager().register(file_path)

    # Cleanup should not raise any exceptions
    SharedMemoryManager().cleanup()


def test_memory_footprint(large_dataframe):
    def worker(name):
        monitor_memory_usage(f"Worker {name} started")
        sdf = SharedMemoryDataFrame(name)
        sdf.to_dataframe()
        monitor_memory_usage(f"Worker {name} finished")

    import loky

    executor = loky.get_reusable_executor(max_workers=1)
    monitor_memory_usage("Main process started")
    SharedMemoryDataFrame("test_memory_footprint", large_dataframe)
    monitor_memory_usage("Main process created shared memory")
    executor.submit(worker, "test_memory_footprint").result()
    monitor_memory_usage("Main process finished")
