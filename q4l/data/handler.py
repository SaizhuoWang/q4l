"""Data handlers are wrappers for data loading and processing."""
import hashlib
import json
import os
import pickle
import typing as tp
from inspect import getfullargspec
from typing import List

import filelock
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..config import ExperimentConfig, GlobalConfig, JobConfig, TimeInterval
from ..qlib.data.dataset import processor as processor_module
from ..qlib.data.dataset.processor import Processor
from ..qlib.log import TimeInspector
from ..qlib.utils import lazy_sort_index
from ..qlib.utils.mod import get_callable_kwargs
from ..utils.log import get_logger, recursive_sort, serialize_value
from ..utils.memory import MmapSharedMemoryNumpyArray, SharedMemoryDataFrame
from ..utils.misc import JsonObject, create_instance, interval_contains
from .loader import Q4LDataLoader

CS_ALL = "__all"  # return all columns with single-level index column
CS_RAW = "__raw"  # return raw data with multi-level index column
# data key
DK_R = "raw"
DK_I = "infer"
DK_L = "learn"
ATTR_MAP = {DK_R: "raw_data", DK_I: "infer_data", DK_L: "learn_data"}

# process type
PTYPE_I = "independent"
PTYPE_A = "append"

# - self._infer will be processed by shared_processors + infer_processors
# - self._learn will be processed by shared_processors + infer_processors + learn_processors
#   - (e.g. self._infer processed by learn_processors )

# init type
IT_FIT_SEQ = (
    "fit_seq"  # the input of `fit` will be the output of the previous processor
)
IT_FIT_IND = "fit_ind"  # the input of `fit` will be the original df
IT_LS = "load_state"  # The state of the object has been load by pickle


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


def process_dataframe(
    df: pd.DataFrame,
    processors: List[Processor],
    with_fit: bool,
    check_for_infer: bool,
) -> tp.Tuple[pd.DataFrame, tp.Dict]:
    feedback_dict = {}
    # Avoid modifying the original data
    if not all(p.readonly() for p in processors):
        df = df.copy()
    for proc in processors:
        if check_for_infer and not proc.is_for_infer():
            raise TypeError(
                "Only processors usable for inference can be used in `infer_processors`"
            )
        with TimeInspector.logt(f"{proc.__class__.__name__}"):
            if with_fit:
                proc.fit(df)
            df = proc(df)
            # NOTE: The code below is to extract feedback from the processor
            # But this is currently incompatible with processor's implementation
            # So we add a condition to avoid errors like `insufficient values to unpack`.
            if isinstance(df, tp.Tuple):
                df, feedback = df
                if isinstance(feedback, tp.Dict):
                    feedback_dict.update(feedback)
    return df, feedback_dict


class Q4LDataHandler:
    def __init__(
        self,
        exp_config: ExperimentConfig,
        job_config: JobConfig,
        process_type: str = PTYPE_I,
    ):
        self.logger = get_logger(self)
        self.exp_config = exp_config
        self.job_config = job_config

        # NOTE: Due to refactor workload, currently this code is just a copy-paste
        instruments = exp_config.data.pool
        start_time = exp_config.time.start_time
        end_time = exp_config.time.end_time
        fit_start_time = exp_config.time.fit_start_time
        fit_end_time = exp_config.time.fit_end_time
        shm_name = exp_config.data.shm_name
        use_shm = exp_config.data.use_shm
        loader_config = exp_config.data.loader
        preprocessor_config = exp_config.data.preprocessor
        cache_dir = job_config.machine.factor_cache_dir

        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.dataloader_config = loader_config
        self.shm_prefix = shm_name
        self.use_shm = use_shm
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.process_type = process_type
        self.cache_dir = cache_dir
        self.data_already_setup = False

        self._setup_preprocessor(preprocessor_config)

        # Setup data loader
        self.data_loader = Q4LDataLoader(loader_config)
        super().__init__()

    @property
    def shm_name(self):
        return (
            None
            if self.shm_prefix is None
            else self.shm_prefix
            + "_".join(
                [
                    self.digest,
                    self.start_time.replace(":", "-"),
                    self.end_time.replace(":", "-"),
                    self.fit_start_time.replace(":", "-"),
                    self.fit_end_time.replace(":", "-"),
                ]
            )
        )

    @property
    def shm_ready_file(self) -> str:
        return f"{self.shm_name}_ready"

    @property
    def shm_ready(self):
        try:
            numpy_array = MmapSharedMemoryNumpyArray(
                name=self.shm_ready_file
            ).to_numpy()
            return numpy_array[0] == 1
        except FileNotFoundError:
            return False

    @property
    def config_dict(self) -> DictConfig:
        if not hasattr(self, "_config_dict"):
            config_dict = {
                "instruments": self.instruments,
                "region": self.exp_config.data.region,
                "dataloader_config": OmegaConf.to_container(
                    self.dataloader_config
                ),
            }
            setattr(
                self,
                "_config_dict",
                OmegaConf.create(recursive_sort(serialize_value(config_dict))),
            )
        return getattr(self, "_config_dict")

    @staticmethod
    def compute_data_config_digest(config: GlobalConfig) -> str:
        config_dict = {
            "instruments": config.experiment.data.pool,
            "region": config.experiment.data.region,
            "dataloader_config": OmegaConf.to_container(
                config.experiment.data.loader
            ),
        }
        config_dict = OmegaConf.create(
            recursive_sort(serialize_value(config_dict))
        )
        config_str = repr(config_dict)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]

    @property
    def digest(self) -> str:
        return hashlib.md5(repr(self.config_dict).encode("utf-8")).hexdigest()[
            :8
        ]

    @property
    def trade_calendar(self):
        return self.data_loader.trade_calendar

    @property
    def data_dim(self):
        return self.learn_data.shape[1]

    @property
    def cache_path(self) -> str:
        cache_dir = self.cache_dir or os.getcwd()
        return os.path.join(cache_dir, self.digest, "cache.pkl")

    def _setup_preprocessor(self, preprocessor_config):
        shared_processors = preprocessor_config.shared
        infer_processors = preprocessor_config.infer
        learn_processors = preprocessor_config.learn

        if infer_processors is None:
            infer_processors = []
        if learn_processors is None:
            learn_processors = []
        if shared_processors is None:
            shared_processors = []

        self.preprocessor_config = {
            "infer_processors": infer_processors,
            "learn_processors": learn_processors,
            "shared_processors": shared_processors,
        }

        # Setup preprocessor
        self.infer_processors = []  # for lint
        self.learn_processors = []  # for lint
        self.shared_processors = []  # for lint
        for pname in (
            "infer_processors",
            "learn_processors",
            "shared_processors",
        ):
            for proc in self.preprocessor_config[pname]:
                getattr(self, pname).append(
                    create_instance(
                        proc,
                        default_module=None
                        if (isinstance(proc, dict) and "module_path" in proc)
                        else processor_module,
                        accept_types=processor_module.Processor,
                        try_kwargs={
                            "fit_start_time": self.fit_start_time,
                            "fit_end_time": self.fit_end_time,
                            "device": self.job_config.misc.device,
                            # "fields_group": self.exp_config.data.sampler.x_group,
                        },
                        fields_group=OmegaConf.to_container(
                            self.exp_config.data.sampler.x_group
                        ),
                    )
                )

    def link_shm(self, **kwargs):
        """Link to data in shared memory.

        If shared memory is not ready, load data and put it to shared memory.

        """
        self.logger.info(
            f"Using shared memory. Trying to load data from {self.shm_name}"
        )

        if self.shm_ready:
            self.logger.info(
                "Found shared memory ready flag. Loading data from shared memory."
            )
            for key in ["raw", "infer", "learn"]:
                self.logger.info(f"Loading {key} data from shared memory")
                shm_df = SharedMemoryDataFrame(
                    f"{self.shm_name}_{key}"
                ).to_dataframe()
                setattr(self, f"{key}_data", shm_df)
        else:
            self.logger.info("Shared memory not ready. Loading data from disk.")
            self.load_data(**kwargs)
            self.logger.info(f"Putting data to shared memory {self.shm_name}")
            for key in ["raw", "infer", "learn"]:
                SharedMemoryDataFrame(
                    name=f"{self.shm_name}_{key}",
                    df=getattr(self, f"{key}_data"),
                    auto_cleanup=False,
                )
            # Create a ready flag file
            MmapSharedMemoryNumpyArray(
                name=self.shm_ready_file, arr=np.array([1]), auto_cleanup=False
            )
            self.logger.info("Shared memory ready flag set.")

        self.logger.info("Data successfully loaded from shared memory.")

    def load_data(self, use_cache: bool = True):
        def _load_data(
            instruments=self.instruments,
            start_time=self.start_time,
            end_time=self.end_time,
        ):
            return lazy_sort_index(
                self.data_loader.load(
                    instruments=instruments,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

        with TimeInspector.logt(
            "Loading data"
        ):  # Raw data loading and factor computation => disk cache
            # make sure the fetch method is based on a index-sorted pd.DataFrame
            if use_cache:
                data, time_interval, success = self._load_cache()
                if success:
                    self.raw_data = data.loc[
                        pd.to_datetime(self.start_time) : pd.to_datetime(
                            self.end_time
                        )
                    ]
                else:
                    self.logger.info(
                        "Cache not found, loading data from source..."
                    )
                    self.raw_data = _load_data(
                        start_time=time_interval.start,
                        end_time=time_interval.end,
                    )
                    self._save_cache()
            else:
                self.raw_data = _load_data()

        with TimeInspector.logt("Fit & process data"):
            self.process_data(with_fit=True)

        # FactorPlotter(self.raw_data).plot_factor_distribution('alpha191', postfix="raw")
        # FactorPlotter(self.raw_data).plot_factor_statistics('alpha191', postfix="raw")
        # FactorPlotter(self.learn_data).plot_factor_distribution('alpha191', postfix="learn")
        # FactorPlotter(self.learn_data).plot_factor_statistics('alpha191', postfix="learn")

    # NOTE: The following methods are not used in the current version
    # def _update_cache(self):
    #     self.logger.info("Updating cache...")

    #     # Load existing cache
    #     existing_cache = self._load_cache()

    #     # Merge the existing cache with the current data
    #     self.data = pd.concat(
    #         [self.data, existing_cache], axis=0
    #     ).drop_duplicates()

    #     # Update metadata to the union of current interval
    #     metadata_path = os.path.dirname(self.cache_path) + "/metadata.json"
    #     with JsonObject(metadata_path) as metadata:
    #         metadata["start_time"] = min(
    #             metadata["start_time"], self.start_time
    #         )
    #         metadata["end_time"] = max(metadata["end_time"], self.end_time)

    #     # Save the updated cache and metadata
    #     self._save_cache()

    def _load_cache(
        self,
    ) -> tp.Tuple[tp.Optional[pd.DataFrame], TimeInterval, bool]:
        """Load the cached data from disk. Will use config digest to locate disk
        cache, and perform metadata comparison.

        Returns
        -------
        tp.Tuple[tp.Optional[pd.DataFrame], TimeInterval, bool]
            The cached data, the time interval of the cached data, and whether the
            cache is valid

        """
        cache_path = self.cache_path
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self.logger.info(f"Loading cache from {cache_path} ...")
        # Check if time matches
        metadata_path = os.path.dirname(cache_path) + "/metadata.json"
        with JsonObject(metadata_path) as metadata:
            try:
                metadata_interval = (
                    metadata["start_time"],
                    metadata["end_time"],
                )
                if not interval_contains(
                    metadata_interval, (self.start_time, self.end_time)
                ):
                    return (
                        None,
                        TimeInterval(
                            start=min(self.start_time, metadata_interval[0]),
                            end=max(self.end_time, metadata_interval[1]),
                        ),
                        False,
                    )
            except (
                KeyError
            ):  # metadata.json is missing and no "start_time" or "end_time"
                pass

        try:
            with open(cache_path, "rb") as f:
                return (
                    pickle.load(f),
                    TimeInterval(
                        start=self.start_time,
                        end=self.end_time,
                    ),
                    True,
                )
        except FileNotFoundError:
            return (
                None,
                TimeInterval(start=self.start_time, end=self.end_time),
                False,
            )

    def _save_cache(self):
        """Save the data and config dict to disk."""
        cache_path = self.cache_path
        self.logger.info(f"Saving cache to {cache_path} ...")
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        # save cache data
        with filelock.FileLock(f"{cache_path}.lock"):
            with open(cache_path, "wb") as f:
                pickle.dump(self.raw_data, f)
        # save cache config
        with filelock.FileLock("{}.lock".format(cache_path + "_config.json")):
            with open(cache_path + "_config.json", "w") as f:
                json.dump(OmegaConf.to_container(self.config_dict), f)
        # save metadata
        metadata_path = os.path.join(cache_dir, "metadata.json")
        with JsonObject(metadata_path) as data:
            data["start_time"] = self.start_time
            data["end_time"] = self.end_time

    # A static version of the above method
    @staticmethod
    def save_cache(
        cache_path: str, data: pd.DataFrame, config_dict: tp.Dict, logger
    ):
        """Save the data and config dict to disk."""
        cache_dir = os.path.dirname(cache_path)
        os.path.join(cache_dir, "metadata.json")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        logger.info(f"Saving cache to {cache_path} ...")
        with filelock.FileLock(f"{cache_path}.lock"):
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        with filelock.FileLock("{}.lock".format(cache_path + "_config.json")):
            with open(cache_path + "_config.json", "w") as f:
                json.dump(OmegaConf.to_container(config_dict), f)

    def process_data(self, with_fit: bool = False):
        """Process_data data. Fun `processor.fit` if necessary.

        Notation: (data)  [processor]

        data processing flow of self.process_type == DataHandlerLP.PTYPE_I

        .. code-block:: text

            (self._data)-[shared_processors]-(_shared_df)-[learn_processors]-(_learn_df)
                                                   \\
                                                    -[infer_processors]-(_infer_df)

        data processing flow of self.process_type == DataHandlerLP.PTYPE_A

        .. code-block:: text

            (self._data)-[shared_processors]-(_shared_df)-[infer_processors]-(_infer_df)-[learn_processors]-(_learn_df)

        Parameters
        ----------
        with_fit : bool
            The input of the `fit` will be the output of the previous processor

        """
        # shared data processors
        shared_df, shared_feedback = process_dataframe(
            self.raw_data,
            self.shared_processors,
            with_fit,
            check_for_infer=True,
        )
        for k, v in shared_feedback.items():
            setattr(self, k, v)

        # data for inference
        infer_df, infer_feedback = process_dataframe(
            shared_df, self.infer_processors, with_fit, check_for_infer=True
        )
        self.infer_data = infer_df
        for k, v in infer_feedback.items():
            setattr(self, k, v)

        # data for learning
        if self.process_type == PTYPE_I:
            learn_df = shared_df
        elif self.process_type == PTYPE_A:
            learn_df = infer_df
        else:
            raise NotImplementedError(
                f"Process type {self.process_type} not supported."
            )

        learn_df, learn_feedback = process_dataframe(
            learn_df, self.learn_processors, with_fit, check_for_infer=False
        )
        self.learn_data = learn_df
        for k, v in learn_feedback.items():
            setattr(self, k, v)

    def setup_data(self, use_cache: bool = True, **kwargs):
        # Load dataframes from disk or shm
        if self.data_already_setup:
            self.logger.info("Data already setup. Skip setup.")
            return
        if self.use_shm:
            self.link_shm(**kwargs)
        else:
            self.load_data(**kwargs)
        self.nan_mask = (
            self.raw_data.isna()
        )  # Original data will not be imputed
        self.nan_mask = self.nan_mask[
            self.learn_data.columns
        ]  # Some columns may be dropped during preprocessing
        self.ticker_list = (
            self.raw_data.index.get_level_values(1).unique().tolist()
        )
        self.data_already_setup = True

    def fetch(
        self,
        segment: TimeInterval,
        col_set: str = CS_ALL,
        data_key: str = DK_I,
        # x_window: int = 1,
        # y_window: int = 1,
    ) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch data from underlying data source."""
        # self.logger.info(
        #     f"Memory profile before creating df 0x11: {display_memory_tree(ProcessNode(psutil.Process()))}"
        # )

        # Make a copy of the underlying data source
        df: pd.DataFrame = getattr(self, ATTR_MAP[data_key]).copy(deep=False)
        ticks = df.index.get_level_values(0).unique()
        # self.logger.info(
        #     f"Memory profile after creating df 0x22: {display_memory_tree(ProcessNode(psutil.Process()))}"
        # )

        # Extend interval by window size
        x_window = self.exp_config.data.sampler.x_window
        y_window = self.exp_config.data.sampler.y_window

        # Align the start and end of the segment to the trade calendar
        start = self.trade_calendar.align(segment.start, mode="backward")
        actual_start_idx = max(ticks.get_loc(start) - x_window + 1, 0)
        actual_start_tick = ticks[actual_start_idx]
        end = self.trade_calendar.align(segment.end, mode="forward")
        actual_end_idx = min(ticks.get_loc(end) + y_window, len(ticks) - 1)
        actual_end_tick = ticks[actual_end_idx]

        # self.logger.info(
        #     f"Memory profile after alignment 0x33: {display_memory_tree(ProcessNode(psutil.Process()))}"
        # )

        # Check the column set requested
        if col_set == CS_ALL:
            # If all columns are requested, set column names to level 1
            df.columns = df.columns.get_level_values(1)
        elif col_set == CS_RAW:
            # If raw columns are requested, keep the dataframe as it is
            df = df
        else:
            # If a specific column set is requested, filter the dataframe
            df = df[col_set]

        # self.logger.info(
        #     f"Memory profile after alignment but before slicing: {display_memory_tree(ProcessNode(psutil.Process()))}"
        # )
        target_df = df.loc[actual_start_tick:actual_end_tick]
        target_nan_mask = self.nan_mask.loc[actual_start_tick:actual_end_tick]
        # self.logger.info(
        #     f"Memory profile after slicing: {display_memory_tree(ProcessNode(psutil.Process()))}"
        # )

        # Return the subset of data within the specified segment
        return target_df, target_nan_mask
