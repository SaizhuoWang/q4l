import contextlib
import functools
import json
import os
import pickle
import traceback
import typing as tp
from datetime import datetime
from typing import Any, ContextManager, Dict, Iterable, Tuple, Union

import filelock
import joblib
import omegaconf
import pandas as pd
import torch.profiler as torch_profiler
from omegaconf import OmegaConf
from tqdm import tqdm

from ..config import ExperimentConfig, GlobalConfig, JobConfig
from ..constants import PROJECT_ROOT
from ..qlib.log import QlibLogger
from ..qlib.utils import init_instance_by_config
from ..qlib.workflow import R


def create_instance(
    config,
    default_module=None,
    accept_types: Union[type, Tuple[type]] = (),
    try_kwargs: Dict = {},
    **kwargs,
) -> Any:
    if isinstance(config, omegaconf.DictConfig):
        cfg = OmegaConf.to_container(config, resolve=True)
    else:
        cfg = config
    return init_instance_by_config(
        cfg, default_module, accept_types, try_kwargs, **kwargs
    )


def reraise_with_stack(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            traceback_str = traceback.format_exc()
            # print(traceback_str)
            raise Exception(
                "Error occurred. Original traceback is\n%s\n" % traceback_str
            )

    return wrapped


class MyProfiler(torch_profiler.profile):
    """A wrapper class for torch_profiler.profile.

    Some handy switches are implemented inside. Backbone is still pytorch
    profiler.

    """

    has_profiled = False

    def __init__(
        self, only_one_epoch: bool = True, activate: bool = True, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.only_one_epoch = only_one_epoch
        self.activate = activate

    def start(self):
        if not self.activate:
            return
        if self.only_one_epoch and self.has_profiled:
            return
        super().start()

    def stop(self):
        if not self.activate:
            return
        if self.only_one_epoch and self.has_profiled:
            return
        super().stop()
        self.has_profiled = True

    def step(self):
        if not self.activate:
            return
        if self.only_one_epoch and self.has_profiled:
            return
        super().step()


def get_torch_profiler(
    wait: int = 2,
    warmup: int = 2,
    active: int = 6,
    repeat: int = 1,
    only_one_epoch: bool = True,
    activate: bool = True,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H%:M%:S")

    profiler_dir = os.path.join(PROJECT_ROOT, "pytorch_profiler", timestamp)
    os.makedirs(profiler_dir, exist_ok=True)
    prof_schedule = torch_profiler.schedule(
        wait=wait, warmup=warmup, active=active, repeat=repeat
    )
    trace_handler = torch_profiler.tensorboard_trace_handler(profiler_dir)
    prof = MyProfiler(
        only_one_epoch=only_one_epoch,
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        activities=[
            torch_profiler.ProfilerActivity.CPU,
            torch_profiler.ProfilerActivity.CUDA,
        ],
        activate=activate,
    )
    return prof


def make_qlib_init_config(
    config: GlobalConfig = None,
    exp_config: ExperimentConfig = None,
    job_config: JobConfig = None,
) -> tp.Dict:
    if config is not None:
        exp_config = config.experiment
        job_config = config.job
    return {
        "region": exp_config.data.region,
        "provider_uri": os.path.join(
            job_config.machine.data_root, exp_config.data.region
        ),
        "exp_manager": {
            "class": "MLflowExpManager",
            "module_path": "q4l.qlib.workflow.expm",
            "kwargs": {
                "uri": job_config.machine.mlflow_tracking_uri,
                "default_exp_name": job_config.name.exp_name,
            },
        },
        "num_parallel": job_config.parallel.repeat
        * job_config.parallel.rolling,
        "mongo": {
            "task_url": job_config.machine.taskdb_uri,
            "task_db_name": job_config.machine.taskdb_name,
        },
    }


class JsonObject:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def __enter__(self):
        if not os.path.exists(self.filepath):
            # Create a new file with empty values
            with filelock.FileLock(self.filepath + ".lock"):
                with open(self.filepath, "w") as file:
                    json.dump({}, file)
        with filelock.FileLock(self.filepath + ".lock"):
            with open(self.filepath, "r") as file:
                self.data = json.load(file)
        self.original_data = self.data.copy()
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If not dirty, do nothing
        if self.data == self.original_data:
            return
        with filelock.FileLock(self.filepath + ".lock"):
            with open(self.filepath, "w") as file:
                json.dump(self.data, file)


def interval_contains(interval1: tuple, interval2: tuple) -> bool:
    # Convert string times to pandas.Timestamp
    start1, end1 = pd.Timestamp(interval1[0]), pd.Timestamp(interval1[1])
    start2, end2 = pd.Timestamp(interval2[0]), pd.Timestamp(interval2[1])

    # Check if interval1 contains interval2
    return start1 <= start2 and end1 >= end2


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as
    argument."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class OptionalTqdm(ContextManager):
    def __init__(self, iterable: Iterable, use_tqdm: bool) -> None:
        """Optional tqdm context manager.

        Parameters
        ----------
        iterable : Iterable
            The iterable to wrap.
        use_tqdm : bool
            True to use tqdm, False otherwise.

        Examples
        --------
        # Usage example
        data = range(10)
        use_progress_bar = True

        with OptionalTqdm(data, use_progress_bar) as progress_data:
            for item in progress_data:
                # Do something with item

        """
        self.iterable = iterable
        self.use_tqdm = use_tqdm
        self._tqdm_iterable = None

    def __enter__(self) -> Iterable:
        if not self.use_tqdm:
            return self.iterable
        self._tqdm_iterable = tqdm(self.iterable)
        return self._tqdm_iterable

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.use_tqdm and self._tqdm_iterable is not None:
            self._tqdm_iterable.close()


def load_pickle(path: str) -> tp.Any:
    """Load a pickle file.
    Parameters
    ----------
    path : str
        The path of the pickle file
    Returns
    -------
    Any
        The loaded pickle object
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def fill_backtest_config(
    collector_config: tp.Dict,
    exp_config: ExperimentConfig,
    job_config: JobConfig,
) -> tp.Dict:
    from ..data.dataset import Q4LDataModule

    if "portfolio_analysis" not in collector_config:
        return collector_config

    data = Q4LDataModule(exp_config=exp_config, job_config=job_config)
    refilled_config = collector_config["portfolio_analysis"]
    # Filling start and end time
    start_time = data.trade_calendar.align(
        exp_config.time.segments.test[0].start, mode="forward"
    )
    end_time = data.trade_calendar.align(
        exp_config.time.segments.test[0].end, mode="backward"
    )
    refilled_config["kwargs"]["config"]["backtest"]["start_time"] = start_time
    refilled_config["kwargs"]["config"]["backtest"]["end_time"] = end_time
    data_loader = data.handler.data_loader
    # Fill benchmark data
    benchmark_config = refilled_config["kwargs"]["config"]["backtest"][
        "benchmark"
    ]
    benchmark_return = data_loader.compute_alpha_expressions(
        expressions=[benchmark_config["field"]],
        instruments=[benchmark_config["ticker"]],
        compute_backend=data_loader.compute_backends[
            benchmark_config["backend"]
        ],
        start_time=start_time,
        end_time=end_time,
        group_name="benchmark",
        names=["return"],
    )
    benchmark_series = benchmark_return["return"]["data"].unstack().swaplevel()
    benchmark_series.index = benchmark_series.index.get_level_values(0)
    refilled_config["kwargs"]["config"]["backtest"]["benchmark"] = {
        "benchmark": benchmark_series
    }
    # Fill exchange config
    refilled_config["kwargs"]["config"]["backtest"]["exchange_kwargs"][
        "exchange"
    ]["kwargs"]["loader"] = data_loader

    collector_config["portfolio_analysis"] = refilled_config

    return collector_config


def generate_evaluations(
    config: GlobalConfig,
    stage_key: str,
    logger: QlibLogger,
    recorder_wrapper=R,
):
    """This function takes a configuration object, a key for the rolling
    strategy, a rolling_recorder instance and a logger to generate evaluations
    for each recorder defined in the config.

    Parameters
    ----------
    config : GlobalConfig
        The configuration dictionary containing experiment and job configurations.
    key : str
        The key to access the recorder configuration in the config dictionary.
        Can be 'rolling' or 'repeat', etc.
    rolling_recorder : object
        The rolling_recorder instance to be used.
    logger : object
        Logger object to log the information during the evaluations generation.

    """

    # Ensure the key exists in the config
    if stage_key not in config.experiment.collector:
        raise ValueError(
            f"Stage key '{stage_key}' not found in collector configuration."
        )

    # Convert backtest benchmark config into real pd.DataFrame
    eval_configs = fill_backtest_config(
        OmegaConf.to_container(
            config.experiment.collector[stage_key], resolve=True
        ),
        exp_config=config.experiment,
        job_config=config.job,
    )
    rec = recorder_wrapper.get_recorder()
    logger.info(rec)
    logger.info(f"eval_configs = {eval_configs}")
    for recorder_name, cfg in eval_configs.items():
        logger.info(f"Recorder {recorder_name} generating evaluations ...")
        recorder = create_instance(
            cfg, recorder=rec, recorder_wrapper=recorder_wrapper
        )
        logger.info(f"Succesfully created recorder {recorder_name}")
        recorder.generate()
        logger.info(f"Recorder {recorder_name} finished generating evaluations")
