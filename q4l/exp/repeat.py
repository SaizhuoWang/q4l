"""Utilities for repeated rolling-window experiments.

Repetition is implemented by multi-processing. Multiple experiments can be run
in parallel.

"""
import os
import random
import subprocess
import sys
import tempfile
import time
import typing as tp
from concurrent.futures import Future

import loky
import mlflow
import pandas as pd
import pymongo
import torch
import yaml
from loky import get_reusable_executor
from omegaconf import OmegaConf

from ..config import GlobalConfig
from ..data.dataset import Q4LDataModule
from ..data.handler import Q4LDataHandler
from ..data.loader import Q4LDataLoader
from ..eval.plotting import plot_return_curves
from ..qlib import init as qlib_init
from ..qlib.model.ens.group import RollingGroup
from ..qlib.utils import FLATTEN_TUPLE, flatten_dict, lazy_sort_index
from ..qlib.workflow import R
from ..qlib.workflow.task.collect import Collector
from ..utils.log import get_logger, redirect_logging
from ..utils.misc import (
    JsonObject,
    generate_evaluations,
    interval_contains,
    load_pickle,
    make_qlib_init_config,
)
from .db_backend import MongoDBBackend, UnQLiteBackend
from .rolling import make_rolling_runner
from .taskgen import RollingGen, generate_tasks

# Some default parameters
POLLING_INTERVAL = 30


def rec_key(subdir_name: str):
    return "repeat", subdir_name


def repeat_ensemble_func(ensemble_dict: dict, *args, **kwargs):
    """using sample:
    from qlib.model.ens.ensemble import AverageEnsemble
    pred_res['new_key_name'] = AverageEnsemble()(predict_dict)

    Parameters
    ----------
    ensemble_dict : dict
        Dictionary you want to ensemble

    Returns
    -------
    pd.DataFrame
        The dictionary including ensenbling result
    """
    # need to flatten the nested dict
    ensemble_dict = flatten_dict(ensemble_dict, sep=FLATTEN_TUPLE)
    get_logger("repeat_ensemble").info(
        f"keys in group: {list(ensemble_dict.keys())}"
    )
    values = list(ensemble_dict.values())
    # NOTE: this may change the style underlying data!!!!
    # from pd.DataFrame to pd.Series
    results = pd.concat(values, axis=1)
    results = results.groupby("datetime").apply(
        lambda df: (df - df.mean()) / df.std()
    )
    results = results.mean(axis=1)
    results = results.sort_index()
    return results


class RepeatSubdirCollector(Collector):
    def __init__(
        self,
        root_dir: str,
        process_list: tp.List[tp.Callable] = [],
        subdir_key_func: tp.Callable = None,
        artifacts_path: tp.Dict[str, str] = {"pred": "pred.pkl"},
        artifacts_key=None,
    ):
        super().__init__(process_list)
        self.logger = get_logger(self)

        self.root_dir = root_dir
        self.subdir_key_func = subdir_key_func
        self.artifacts_path = artifacts_path
        self.artifacts_key = (
            artifacts_key
            if artifacts_key is not None
            else list(artifacts_path.keys())
        )

        # Read the rolling window list
        with open(os.path.join(self.root_dir, "global_config.yaml"), "r") as f:
            global_config = yaml.safe_load(f)
            self.num_repeat = global_config["job"]["repeat"]["total_repeat"]

    def collect(self) -> dict:
        ret = {}  # {'pred': {window1: pred1, window2: pred2, ...}}

        # Collect subdir list
        subdir_list = []
        for subdir in os.listdir(self.root_dir):
            if (
                os.path.isdir(os.path.join(self.root_dir, subdir))
                and subdir.isnumeric()
            ):
                if int(subdir) >= self.num_repeat:
                    continue
                subdir_list.append(subdir)
        self.logger.info(
            f"Found {len(subdir_list)} subdirs:\n" + "\n".join(subdir_list)
        )

        for subdir in subdir_list:
            subdir_key = (
                self.subdir_key_func(subdir) if self.subdir_key_func else subdir
            )
            for key in self.artifacts_key:
                artifact = load_pickle(
                    os.path.join(
                        self.root_dir, subdir, self.artifacts_path[key]
                    )
                )
                item_dict = ret.setdefault(key, {})
                item_dict[subdir_key] = artifact
        return ret


class RepeatExpManager:
    def __init__(
        self,
        config: GlobalConfig,
        repeat_start_index: int = 0,
        exp_fn: tp.Callable = None,
    ) -> None:
        self.config = config
        config.job.misc.device = "cuda:0"
        self.exp_config = config.experiment
        self.job_config = config.job
        self.timestamp = config.job.misc.timestamp
        self.repeat_start_index = repeat_start_index
        self.exp_fn = exp_fn or self.default_exp_fn
        self.logger = get_logger(self)

    def collect(self):
        """Collect repeated experiment results and perform an ensemble
        analysis."""
        R.set_suffix("")
        R.set_tags(task_status="collecting")
        self.logger.info(f"Collecting repeated experiment results...")
        collector = RepeatSubdirCollector(
            root_dir=R.artifact_uri,
            process_list=RollingGroup(ens=repeat_ensemble_func),
            subdir_key_func=rec_key,
            artifacts_path={
                "pred": "sig_analysis/pred.pkl",
                "label": "sig_analysis/label.pkl",
            },
            artifacts_key=["pred", "label"],
        )
        repeat_results = collector()
        self.logger.info("Result collected. Now dumping.")
        artifact_prefix = {"pred": "sig_analysis", "label": "sig_analysis"}

        repeat_recorder = R.get_recorder()
        for k, v in repeat_results.items():
            key = list(v.keys())[0]
            file_suffix = os.path.join(R.suffix, artifact_prefix[k])
            repeat_recorder.save_objects(
                artifact_path=file_suffix, **{f"{k}.pkl": v[key]}
            )

        generate_evaluations(
            config=self.config,
            stage_key="repeat",
            logger=self.logger,
            recorder_wrapper=R,
        )
        portfolio_report_df = pd.read_csv(
            os.path.join(
                R.artifact_uri,
                "portfolio_analysis",
                "report_normal_1day.csv",
            ),
            index_col=0,
        )

        rolling_backtest_dfs = []
        intervals = None
        for i in range(self.job_config.repeat.total_repeat):
            cur_df = pd.read_csv(
                os.path.join(
                    R.artifact_uri,
                    str(i),
                    "portfolio_analysis",
                    "report_normal_1day.csv",
                ),
                index_col=0,
            )
            if intervals is None:
                # Do some plotting, with rolling regions
                with open(
                    os.path.join(R.artifact_uri, str(i), "rolling_list.txt"),
                    "r",
                ) as f:
                    interval_lines = f.readlines()
                    intervals = []
                    for line in interval_lines:
                        interval_ends = line.strip().split("~")
                        start, end = interval_ends[0], interval_ends[1]
                        intervals.append((start, end))
            rolling_backtest_dfs.append(cur_df)

        plot_return_curves(
            dataframes=rolling_backtest_dfs,
            ensemble_df=portfolio_report_df,
            intervals=intervals,
            artifact_uri=R.artifact_uri,
        )

        self.logger.info(f"Collect finished.")
        R.set_tags(task_status="done")
        return repeat_recorder

    def run(self):
        """Run repeated experiments (parallel or serial)
        NOTE: Number of repeats is solely determined by `total_repeat` in `job_config.repeat`.
        Splitting in the outside need to change this value before building the manager.
        """
        # Init qlib for future use
        qlib_init(
            **make_qlib_init_config(
                exp_config=self.exp_config, job_config=self.job_config
            )
        )
        # Initialize mlflow experiment tracking
        self.setup_tracking()
        # Make shared memory data handler
        self.save_job_info()
        # Prepare disk cache (factor value dataframes)
        self.prepare_disk_cache()
        # Prepare shared memory
        self.prepare_shared_memory()
        # Save code change to mlflow for reproducibility
        self.save_git_changes()
        # Run experiments
        self.run_experiment()
        # Collect repeated results
        self.collect()
        # Close tracking
        mlflow.end_run()

    def save_git_changes(self):
        """Largely copied from
        qlib.workflow.recorder.MLflowRecorder._log_uncommitted_code."""
        for cmd, fname in [
            ("git diff", "code_diff.txt"),
            ("git status", "code_status.txt"),
            ("git diff --cached", "code_cached.txt"),
        ]:
            try:
                out = subprocess.check_output(cmd, shell=True)
                mlflow.log_text(
                    out.decode(), fname
                )  # this behaves same as above
            except subprocess.CalledProcessError:
                self.logger.info(
                    f"Fail to log the uncommitted code of $CWD({os.getcwd()}) when run {cmd}."
                )

    def prepare_shared_memory(self):
        if self.exp_config.data.use_shm:
            if not self.job_config.misc.prepare_shm:
                self.logger.info(
                    "Using shared memory, but not preparing it. Please make sure the shm is ready."
                )
                return
            self.logger.info("Preparing shared memory")
            self.config.experiment.data.shm_name = self.shm_name
            self._prepare_shared_memory()
        else:
            self.logger.info("Not using shared memory")

    def save_job_info(self):
        """Dump experiment information (Job info) to a database."""

        db_backend_type, db_path = self.config.job.machine.taskdb_uri.split(
            "://"
        )[0:2]
        if db_backend_type == "mongodb":
            db_backend = MongoDBBackend(
                uri=self.config.job.machine.taskdb_uri,
                dbname="quant_exp_history",
            )
        elif db_backend_type == "unqlite":
            db_backend = UnQLiteBackend(database_path=db_path)

        # Insert job info into the specified collection
        job_id = db_backend.insert("exp", self.job_info)

        # Handling the job_id based on the backend type
        if isinstance(db_backend, MongoDBBackend):
            self.job_config.misc.mongo_job_id = str(job_id.inserted_id)
        elif isinstance(db_backend, UnQLiteBackend):
            self.job_config.misc.mongo_job_id = str(
                job_id
            )  # or any other appropriate handling

        self.logger.info(
            f"Job info saved with ID {self.job_config.misc.mongo_job_id}"
        )

    @property
    def job_info(self):
        """A quick digest of the submitted job."""
        return {
            "name": self.job_config.name.exp_name,
            "timestamp": self.timestamp,
            "repeat_start_index": self.repeat_start_index,
            "num_repeat": self.job_config.repeat.total_repeat,
            "num_cpu_per_task": self.job_config.resource.cpu_per_task,
            "num_gpu": torch.cuda.device_count(),
            "parallel_repeat": self.job_config.parallel.repeat,
            "parallel_rolling": self.job_config.parallel.rolling,
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", "<no_slurm_jobid>"),
            "SLURM_JOB_NAME": os.environ.get(
                "SLURM_JOB_NAME", "<no_slurm_jobname>"
            ),
            "SLURM_JOB_NODELIST": os.environ.get(
                "SLURM_JOB_NODELIST", "<no_slurm_job_nodelist>"
            ),
        }

    def setup_tracking(self):
        """Init mlflow experiment tracking context.

        TODO: Decouple this routine with mlflow using newly designed experiment tracking module.

        """
        # Set current experiment
        mlflow.set_tracking_uri(self.job_config.machine.mlflow_tracking_uri)
        # Here we sleep for a random time to avoid concurrent mlflow experiment creation
        sleep_time = random.randint(3, 10)
        time.sleep(sleep_time)
        mlflow.set_experiment(self.job_config.name.exp_name)
        # Set current run
        run_name = self.job_config.name.run_name
        mlflow.start_run(run_name=run_name)
        # Save experiment config file
        with tempfile.TemporaryDirectory("w+") as temp_dir:
            f = open(os.path.join(temp_dir, "global_config.yaml"), "w+")
            yaml.safe_dump(OmegaConf.to_container(self.config), f)
            mlflow.log_artifact(os.path.join(temp_dir, "global_config.yaml"))
        # Redirect logging to mlflow artifact text files
        self.recorder = R.get_recorder(
            experiment_name=self.job_config.name.exp_name,
            recorder_name=run_name,
        )
        R.set(
            experiment_name=self.job_config.name.exp_name,
            recorder_name=run_name,
        )
        R.set_tags(
            num_repeat=self.job_config.repeat.total_repeat,
            slurm_jobid=os.environ.get("SLURM_JOB_ID", "<no_slurm_jobid>"),
        )
        redirect_logging(is_debug=self.job_config.misc.debug)
        self.logger.info(
            f"Start running {self.job_config.repeat.total_repeat} repeated experiments"
        )

    def make_repeat_specific_config(self, gpu_index: int):
        """Repeated experiments may differ in their config (e.g. different
        random seeds, different GPUs, etc.). This hook is called for making each
        individual repeat experiment's config. Override this to customize your
        own repeat config.

        Changes in this config:
            - New random seed
            - Specific GPU device (for single GPU model training)

        """
        from copy import deepcopy

        new_config = deepcopy(self.config)
        # Customize seed
        random_seed = random.randint(0, 1000000)
        self.logger.info(f"Setting random seed to {random_seed}")
        new_config.job.misc.seed = random_seed
        # Set GPU index
        new_config.job.misc.device = f"cuda:{gpu_index}"

        return new_config

    def run_experiment(self):
        """Execute the experiment function, `exp_fn`, for a specified number of
        repeats in either parallel or single mode, based on the job
        configuration.

        The function iterates through the total number of repeats, invoking the
        experiment function with the appropriate arguments. In parallel mode, a
        reusable executor is created to manage the parallel execution of the
        experiment function.

        """
        parallel_repeat = self.job_config.parallel.repeat
        total_gpu = torch.cuda.device_count()

        # Initialize variables
        gpu_index = 0
        repeat_jobs = []

        # Determine if a reusable executor is needed for parallel execution
        executor = loky.get_reusable_executor(max_workers=parallel_repeat)
        R.set_tags(task_status="training")
        # Iterate through the total number of repeats
        for repeat_index in range(
            self.repeat_start_index,
            self.job_config.repeat.total_repeat + self.repeat_start_index,
        ):
            # Add new random seed for each repeat
            repeat_config = self.make_repeat_specific_config(
                gpu_index=gpu_index
            )
            # Set the arguments for the experiment function
            exp_fn_args = {
                "repeat_index": repeat_index,
                "gpu_index": gpu_index,
                "config": repeat_config,
                "is_subprocess": True,
                "timestamp": self.timestamp,
            }

            # Execute the experiment function in parallel or single mode
            self.logger.info(
                f"Submitting repeat {repeat_index} to executor with "
                f"gpu_idx = {gpu_index}..."
            )
            if parallel_repeat == 1:
                # Single mode
                self.exp_fn(**exp_fn_args)
            else:
                # Parallel mode
                job = executor.submit(self.exp_fn, **exp_fn_args)
                repeat_jobs.append(job)
                # Update the GPU index
                if total_gpu > 0:
                    gpu_index = (gpu_index + 1) % total_gpu
                else:
                    gpu_index = -1
        start_time = time.time()
        # Wait for all parallel tasks to complete, if applicable
        self.wait_for_all_jobs(repeat_jobs)

        end_time = time.time()
        self.logger.info(f"Total time spent: {end_time - start_time} seconds")

    def wait_for_all_jobs(self, job_list: tp.List[Future]):
        # Wait for all parallel tasks to complete, if applicable
        client = mlflow.MlflowClient(
            tracking_uri=self.job_config.machine.mlflow_tracking_uri
        )
        run = mlflow.active_run()
        while job_list:
            # Sleep
            time.sleep(POLLING_INTERVAL)
            done_jobs = 0
            for future in job_list:
                if future.done():
                    done_jobs += 1
                    future.result()  # See if any error occurs
            if done_jobs == len(job_list):
                self.logger.info("All repeat jobs completed")
                client.set_terminated(run.info.run_id, "FINISHED")
                break
            self.logger.info(f"{done_jobs} of {len(job_list)} repeat done.")
            client.set_terminated(run.info.run_id, "RUNNING")

            try:
                num_rolling_per_repeat = int(
                    R.get_recorder().list_tags()["num_rolling_per_repeat"]
                )
            except KeyError:
                self.logger.info("No rolling runs has started, waiting...")
                continue
            # Do some logging
            task_pool = (
                pymongo.MongoClient(self.config.job.machine.taskdb_uri)
                .get_database(self.config.job.machine.taskdb_name)
                .get_collection(self.config.job.name.run_name)
            )
            all_tasks = list(task_pool.find())
            num_tasks_in_progress = len(all_tasks)
            num_total_tasks = (
                num_rolling_per_repeat * self.job_config.repeat.total_repeat
            )
            num_completed_tasks = len(
                [1 for x in all_tasks if x["status"] == "done"]
            )
            num_waiting_tasks = len(
                [1 for x in all_tasks if x["status"] == "waiting"]
            )
            num_running_tasks = len(
                [1 for x in all_tasks if x["status"] == "running"]
            )
            num_queued_tasks = num_total_tasks - num_tasks_in_progress
            self.logger.info(
                f"Progress monitor: TOTAL {num_total_tasks} | "
                f"COMPLETED {num_completed_tasks} | WAITING "
                f" {num_waiting_tasks} | RUNNING {num_running_tasks} "
                f"| QUEUED {num_queued_tasks}"
            )

    def default_exp_fn(
        self, repeat_index: int, gpu_index: int, is_subprocess: bool, **kwargs
    ):
        make_rolling_runner(
            config=self.config,
            repeat_index=repeat_index,
            gpu_idx=gpu_index,
            timestamp=self.timestamp,
        ).start(is_subprocess=is_subprocess)

    @property
    def shm_name(self):
        return "q4l_shm_data"

    def _shm_cleaner(self, signum, frame, handler_config):
        self.logger.info(
            f"Received signal {signum}. Cleaning up shared memory data handler"
        )
        if handler_config["kwargs"]["use_shm"]:
            for shm_handle in os.listdir("/dev/shm"):
                if self.shm_name in shm_handle:
                    self.logger.info(f"Removing /dev/shm/{shm_handle}")
                    os.remove(os.path.join("/dev/shm", shm_handle))
        sys.exit(0)

    def _task_init(self, task):
        # Function to initialize the data module
        Q4LDataModule(exp_config=task, job_config=self.config.job)

    def _prepare_shared_memory(self):
        """Prepare shared memory for each rolling task.

        This function is applicable only when the repeat task is a rolling task.

        We need to first generate a series of rolling task time intervals, consisting of
        `start_time`, `end_time`, `fit_start_time`, `fit_end_time`.

        Then we need to generate data handler config for each rolling task, and uses these
        configs to initialize data handlers that prepare shared memory.

        Finally, we need to add a clean-up handler for each rolling task, which is essentially
        a signal handler that removes the shared memory data handler when the task is terminated.

        """
        self.logger.info("Making shared memory data handler")
        # Make rolling task time intervals
        rolling_generator = RollingGen(
            step=self.config.experiment.time.rolling_step,
            rtype=self.config.experiment.time.rolling_type,
        )
        tasks = generate_tasks(
            tasks=self.config.experiment, generators=rolling_generator
        )
        self.logger.info(
            f"Preparing {len(tasks)} shared memory data handlers with {5} workers..."
        )
        # Using ProcessPoolExecutor for parallel execution
        with get_reusable_executor(
            max_workers=5
        ) as executor:  # Adjust max_workers based on your needs
            jobs = []
            for task in tasks:
                jobs.append(executor.submit(self._task_init, task))
            for job in jobs:
                job.result()
            # list(executor.map(self._task_init, tasks))

    def prepare_disk_cache(self):
        """Pre-compute all factors required by the experiments, and save them to
        disk cache.

        This is because in some rolling tasks, the time window may not be big
        enough, and may cause unnecessary computation/error. So we pre-compute
        all the factors required by the experiments, and save them to disk
        beforehand.

        """
        use_disk_cache = self.config.job.misc.get(
            "use_disk_cache", True
        )  # default to True
        refresh_disk_cache = self.config.job.misc.get(
            "refresh_disk_cache", False
        )  # default to False

        if not use_disk_cache:
            self.logger.info("Not using disk cache.")
            return

        start_time = self.config.experiment.time.start_time
        end_time = self.config.experiment.time.end_time
        # Go check if cache satisfies my interval requirement. If not, recompute.
        digest = Q4LDataHandler.compute_data_config_digest(self.config)
        cache_dir = os.path.join(
            self.config.job.machine.factor_cache_dir, digest
        )
        tmp_loader = Q4LDataLoader(
            config=self.config.experiment.data.loader
        )  # A temporary data loader for computing factor values.
        metadata_path = os.path.join(cache_dir, "metadata.json")

        def cache_valid():
            if not os.path.exists(cache_dir):
                return False
            try:
                with JsonObject(metadata_path) as metadata:
                    cache_interval = (
                        metadata["start_time"],
                        metadata["end_time"],
                    )
                    return interval_contains(
                        cache_interval, (start_time, end_time)
                    )
            except KeyError:
                return False

        if cache_valid() and not refresh_disk_cache:
            self.logger.info(f"Cache {cache_dir} is valid, returning.")
            return

        # If cache is not valid, we just build a new cache that satisfies the interval requirement.
        try:
            # If there's an existing cache, we will update it.
            with JsonObject(metadata_path) as metadata:
                cache_interval = (metadata["start_time"], metadata["end_time"])
            new_interval = (
                min(start_time, cache_interval[0]),
                max(end_time, cache_interval[1]),
            )
            self.logger.info(
                f"Updating cache at {cache_dir}: {cache_interval} => {new_interval}"
            )
        except:
            # If existing cache is broken, we will just build a new cache.
            self.logger.info(
                f"Cache {cache_dir} does not exist. Creating it ..."
            )
            os.makedirs(cache_dir, exist_ok=True)
            new_interval = (start_time, end_time)

        data = lazy_sort_index(
            tmp_loader.load(
                instruments=self.config.experiment.data.pool,
                start_time=new_interval[0],
                end_time=new_interval[1],
            )
        )
        Q4LDataHandler.save_cache(
            cache_path=os.path.join(cache_dir, "cache.pkl"),
            data=data,
            config_dict=OmegaConf.create(
                {
                    "instruments": self.config.experiment.data.pool,
                    "dataloader_config": self.config.experiment.data.loader,
                }
            ),
            logger=self.logger,
        )
        with JsonObject(metadata_path) as metadata:
            metadata["start_time"] = new_interval[0]
            metadata["end_time"] = new_interval[1]
