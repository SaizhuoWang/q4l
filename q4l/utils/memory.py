"""Utilities for (shared) memory management."""
import atexit
import math
import os
import pickle
import struct
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd
import psutil

from q4l.utils.log import get_logger

from .misc import reraise_with_stack


def convert_size(size_bytes: int) -> str:
    """Converts bytes to human readable format."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


class SharedMemoryManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.no_cleanup_files = set()
            cls._instance.shm_object_mapping = {}
            atexit.register(cls._instance.cleanup)
        return cls._instance

    def register(self, name: str, size: int = 0, auto_cleanup=True):
        if name in self.shm_object_mapping:
            return self.shm_object_mapping[name]
        try:  # first see if it's already created
            shm_object = SharedMemory(name=name)
        except:  # if not, create it
            if (
                size == 0
            ):  # If we originally intend to load a shared memory file, but it does not exist, raise error
                raise FileNotFoundError(
                    f"Trying to load shared memory file {name}, but file not found."
                )
            shm_object = SharedMemory(name=name, create=True, size=size)
        self.shm_object_mapping[name] = shm_object
        if not auto_cleanup:
            self.no_cleanup_files.add(name)
        return shm_object

    def unregister(self, name: str):
        if name in self.shm_object_mapping:
            self.shm_object_mapping[name].close()
            self.shm_object_mapping.pop(name)
            if name in self.no_cleanup_files:
                self.no_cleanup_files.remove(name)

    def cleanup(self):
        return  # Do not cleanup
        # if self.sub_process:
        #     return  # Do not cleanup in sub process
        # for name in self.shm_object_mapping:
        #     if name in self.no_cleanup_files:
        #         continue
        #     self.shm_object_mapping[name].close()
        #     self.shm_object_mapping[name].unlink()
        #     self.shm_object_mapping.pop(name)

    def is_name_taken(self, name: str) -> bool:
        return name in self.shm_object_mapping

    def summary(self) -> str:
        summary_str = (
            f"Total shared memory files: {len(self.shm_object_mapping)}\n"
        )
        for name in self.shm_object_mapping:
            shm_object = self.shm_object_mapping[name]
            shm_size = convert_size(shm_object.size)
            summary_str += f"Name: {name}, Size: {shm_size} bytes\n"
        return summary_str


class MmapSharedMemoryNumpyArray:
    def __init__(self, name: str, arr: np.ndarray = None, auto_cleanup=True):
        """Create or link to a shared memory numpy array.

        Parameters
        ----------
        name : str
            The file name of the shared memory file.
        arr : np.ndarray, optional
            Source data array, if not None, write the array to shm.
            by default None
        auto_cleanup : bool, optional
            Whether to automatically cleanup this array upon exit,
            by default True

        """
        self.name = name
        self.auto_cleanup = auto_cleanup
        self.shm_mgr = SharedMemoryManager()
        if arr is not None:
            self._create_shared_memory(arr)
        else:
            self._load_shared_memory()

    def _create_shared_memory(self, arr):
        data_buffer = arr.tobytes()

        dtype_bytes = pickle.dumps(arr.dtype)
        dtype_size = len(dtype_bytes)
        dtype_buffer = struct.pack("B", dtype_size) + dtype_bytes

        shape_bytes = pickle.dumps(arr.shape)
        shape_size = len(shape_bytes)
        shape_buffer = struct.pack("B", shape_size) + shape_bytes

        total_size = len(data_buffer) + len(dtype_buffer) + len(shape_buffer)

        # Allocate shared memory chunk
        self.shm_object = self.shm_mgr.register(
            name=self.name, size=total_size, auto_cleanup=self.auto_cleanup
        )

        # Write to shared memory
        offset = 0
        self.shm_object.buf[offset : offset + len(dtype_buffer)] = dtype_buffer
        offset += len(dtype_buffer)

        self.shm_object.buf[offset : offset + len(shape_buffer)] = shape_buffer
        offset += len(shape_buffer)

        self.shm_object.buf[offset:] = data_buffer

    def _load_shared_memory(self):
        self.shm_object = self.shm_mgr.register(name=self.name)

    def to_numpy(self) -> np.ndarray:
        offset = 0

        # Decode dtype
        dtype_size = struct.unpack(
            "B", self.shm_object.buf[offset : offset + 1]
        )[0]
        dtype = pickle.loads(
            self.shm_object.buf[offset + 1 : offset + 1 + dtype_size]
        )
        offset += 1 + dtype_size

        # Decode shape
        shape_length = struct.unpack(
            "B", self.shm_object.buf[offset : offset + 1]
        )[0]
        offset += 1
        shape = pickle.loads(
            self.shm_object.buf[offset : offset + shape_length]
        )
        offset += shape_length

        # Get data buffer
        data_buffer = self.shm_object.buf[offset:]

        return np.ndarray(shape=shape, dtype=dtype, buffer=data_buffer)

    def close(self):
        self.shm_mgr.unregister(self.name)

    def unlink(self):
        self.shm_mgr.unregister(self.name)


def monitor_memory_usage(message=None):
    print(message)
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    res_memory = memory_info.rss / 1024**2
    shr_memory = memory_info.shared / 1024**2
    print(
        f"Process {os.getpid()} RES memory: {res_memory:.2f} MB, SHR memory: {shr_memory:.2f} MB\n"
    )


class SharedMemoryDataFrame:
    def __init__(self, name: str, df: pd.DataFrame = None, auto_cleanup=True):
        self.name = name
        self.auto_cleanup = auto_cleanup
        if df is not None:
            self._create_shared_memory(df)
        else:
            self.data_arr = MmapSharedMemoryNumpyArray(f"{self.name}_data")
            self.index_arr = MmapSharedMemoryNumpyArray(f"{self.name}_index")
            self.columns_arr = MmapSharedMemoryNumpyArray(
                f"{self.name}_columns"
            )
        self.logger = get_logger(self)

    def _create_shared_memory(self, df: pd.DataFrame):
        data_arr = MmapSharedMemoryNumpyArray(
            name=self.name + "_data",
            arr=df.values,
            auto_cleanup=self.auto_cleanup,
        )

        # Make index array
        index_np_arr = df.index.to_numpy()
        lv0_dtype = df.index.get_level_values(0).to_numpy().dtype
        lv1_dtype = df.index.get_level_values(1).to_numpy().astype("str").dtype
        index_np_arr = index_np_arr.astype(
            np.dtype([("tick", lv0_dtype), ("ticker", lv1_dtype)])
        )
        index_arr = MmapSharedMemoryNumpyArray(
            arr=index_np_arr,
            name=self.name + "_index",
            auto_cleanup=self.auto_cleanup,
        )

        # Make column array
        columns_np_arr = df.columns.to_numpy()
        lv0_dtype = (
            df.columns.get_level_values(0).to_numpy().astype("str").dtype
        )
        lv1_dtype = (
            df.columns.get_level_values(1).to_numpy().astype("str").dtype
        )
        columns_np_arr = columns_np_arr.astype(
            np.dtype([("group", lv0_dtype), ("feature", lv1_dtype)])
        )
        columns_arr = MmapSharedMemoryNumpyArray(
            arr=columns_np_arr,
            name=self.name + "_columns",
            auto_cleanup=self.auto_cleanup,
        )

        self.data_arr = data_arr
        self.index_arr = index_arr
        self.columns_arr = columns_arr

    def to_dataframe(self):
        data = self.data_arr.to_numpy()
        index = self.index_arr.to_numpy()
        columns = self.columns_arr.to_numpy()
        # self.logger.info(
        #     f"Memory profile after converting to numpy:\n{display_memory_tree(ProcessNode(psutil.Process()))}"
        # )

        # NOTE: An ugly trick here: Unpack and pack again, otherwise error
        ticks = [x[0] for x in index]
        tickers = [x[1] for x in index]
        tuples = list(zip(ticks, tickers))
        index = pd.MultiIndex.from_tuples(tuples, names=["tick", "ticker"])
        # self.logger.info(
        #     f"Memory profile after creating index:\n{display_memory_tree(ProcessNode(psutil.Process()))}"
        # )

        # NOTE: An ugly trick here: Unpack and pack again, otherwise error
        groups = [x[0] for x in columns]
        features = [x[1] for x in columns]
        tuples = list(zip(groups, features))
        columns = pd.MultiIndex.from_tuples(tuples, names=["group", "feature"])
        # self.logger.info(
        #     f"Memory profile after creating columns:\n{display_memory_tree(ProcessNode(psutil.Process()))}"
        # )

        df = pd.DataFrame(data, index=index, columns=columns)
        # self.logger.info(
        #     f"Memory profile after creating dataframe:\n{display_memory_tree(ProcessNode(psutil.Process()))}"
        # )
        return df

    def close(self):
        self.data_arr.close()
        self.index_arr.close()
        self.columns_arr.close()

    def unlink(self):
        self.data_arr.unlink()
        self.index_arr.unlink()
        self.columns_arr.unlink()




@reraise_with_stack
def worker(name):
    monitor_memory_usage(f"Worker {name} started")
    sdf = SharedMemoryDataFrame(name)
    sdf.to_dataframe()
    monitor_memory_usage(f"Worker {name} finished")


def test_memory_footprint():
    import loky

    executor = loky.get_reusable_executor(max_workers=1)
    monitor_memory_usage("Main process started")
    dff = large_dataframe(np.random.rand(10000, 10000))
    monitor_memory_usage("Main process created dff")
    SharedMemoryDataFrame("test_memory_footprint", dff)
    monitor_memory_usage("Main process created shared memory")
    executor.submit(worker, "test_memory_footprint").result()
    monitor_memory_usage("Main process finished")


def clear_shm(pattern: str):
    shm_files = os.listdir("/dev/shm")
    for f in shm_files:
        if pattern in f:
            os.unlink(f"/dev/shm/{f}")
