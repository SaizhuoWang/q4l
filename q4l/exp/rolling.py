"""Forward-rolling experiment manager, which is typical for time-series tasks.

Implements a rolling task runner, which is a customized version of qlib rolling
task runner,model_rolling/task_manager_rolling.py rolling task runner instance.
RollingTaskRunner.start() is the entry point of a rolling task. It runs multiple
rolling tasks serially in a single process. A single process is essentially a
single

"""

import copy
import os
import random
import typing as tp
from typing import Dict

import fire
import pandas as pd
import yaml
from omegaconf import OmegaConf

from ..config import ExperimentConfig, GlobalConfig
from ..eval.plotting import plot_return_curve
from ..qlib.model.ens.group import RollingGroup
from ..qlib.workflow.recorder import MLflowRecorder, Recorder
from ..qlib.workflow.task.collect import Collector
from ..utils.log import get_logger
from ..utils.misc import generate_evaluations, load_pickle
from .pipeline import q4l_task_wrapper_fn
from .task_manager import TaskManager
from .taskgen import RollingGen, generate_tasks


def collect_rolling(
    exp_dir: str, fname_mapping: Dict[str, str]
) -> Dict[str, tp.Any]:
    """Given the directory of the experiment, it will read the rolling window
    list from `rolling_list.yaml` and collect results from each rolling window
    sub-directory. The file to collect in each sub-directory is specified in a
    dictionary {key: file_name}. And the collected results are stored in nested
    dictionary {window: {key: result}}. The collected results will be
    concatenated from each rolling window into a single dictionary {key:
    file_name}.

    Parameters
    ----------
    exp_dir : str
        The directory of the rolling experiment
    fname_mapping : Dict[str, str]
        The mapping from the key to the file name

    Returns
    -------
    Dict[str, Any]
        The collected results.

    """
    rolling_list_file = os.path.join(exp_dir, "rolling_list.yaml")
    with open(rolling_list_file, "r") as f:
        rolling_list = yaml.load(f, Loader=yaml.FullLoader)
    rolling_results = {}
    for window, window_dir in rolling_list.items():
        window_dir = os.path.join(exp_dir, window_dir)
        for key, fname in fname_mapping.items():
            path = os.path.join(window_dir, fname)
            if key not in rolling_results:
                rolling_results[key] = {}
            rolling_results[key][window] = load_pickle(path)
    # Concatenate the results from each rolling window


def make_rolling_runner(
    config: GlobalConfig, repeat_index: int, gpu_idx: int, timestamp: str
) -> "RollingTaskRunner":
    config_new = copy.deepcopy(config)
    config_new.job.misc.seed = random.randint(0, 100000)
    config_new.experiment.model.kwargs["GPU"] = gpu_idx
    return RollingTaskRunner(
        config=config_new,
        repeat_index=repeat_index,
        timestamp=timestamp,
    )


def subdir_contain(x: str, l: tp.List[str]) -> bool:
    return x in l


class RollingSubdirCollector(Collector):
    def __init__(
        self,
        root_dir: str,
        process_list: tp.List[tp.Callable] = [],
        subdir_key_func: tp.Callable = None,
        subdir_filter_func: tp.Callable = subdir_contain,
        artifacts_path: tp.Dict[str, str] = {"pred": "pred.pkl"},
        artifacts_key=None,
    ):
        """My own collector for a forward rolling window experiment. Given the
        directory of the experiment, it will read the rolling window list from
        `rolling_list.txt` and collect results from each rolling window sub-
        directory. The file to collect in each sub-directory is specified in a
        dictionary {key: file_name}. And the collected results are stored in
        nested dictionary {window: {key: result}}. The collected results will be
        concatenated from each rolling window into a single dictionary {key:
        file_name}.

        Parameters
        ----------
        root_dir : str
            The directory of the rolling experiment
        process_list : tp.List[tp.Callable], optional
            The list of post-processing functions, by default []
        subdir_key_func : tp.Callable, optional
            The function to get the key for each sub-directory, by default None
        subdir_filter_func : tp.Callable, optional
            The function to filter out the sub-directories, by default None
        artifacts_path : tp.Dict[str, str], optional
            The mapping from the key to the file name, by default {"pred": "pred.pkl"}
        artifacts_key : tp.List[str], optional
            The list of keys to collect, by default None

        """
        super().__init__(process_list)
        self.logger = get_logger(self)

        self.root_dir = root_dir
        self.subdir_key_func = subdir_key_func
        self.subdir_filter_func = subdir_filter_func
        self.artifacts_path = artifacts_path
        self.artifacts_key = (
            artifacts_key
            if artifacts_key is not None
            else list(artifacts_path.keys())
        )

        # Read the rolling window list
        with open(os.path.join(self.root_dir, "rolling_list.txt"), "r") as f:
            self.rolling_list = f.readlines()
        self.rolling_list = [x.strip() for x in self.rolling_list]

    def collect(self) -> tp.Dict:
        ret = {}  # {'pred': {window1: pred1, window2: pred2, ...}}
        subdir_list = [
            d
            for d in os.listdir(self.root_dir)
            if self.subdir_filter_func(d, self.rolling_list)
        ]
        self.logger.info(f"Collecting from {len(subdir_list)} sub-directories")
        self.logger.info("\n".join(subdir_list))

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


class RollingTaskRunner:
    def __init__(
        self,
        config: GlobalConfig,
        repeat_index: int,
        timestamp: str,
        recorder_wrapper=None,
    ):
        from ..qlib.workflow import R

        self.recorder_wrapper = recorder_wrapper or R
        self.config = config
        self.exp_config = config.experiment
        self.job_config = config.job
        self.repeat_index = repeat_index
        self.timestamp = timestamp
        # Setup recorder
        self.recorder = self.recorder_wrapper.get_recorder(
            experiment_name=self.job_config.name.exp_name,
            recorder_name=self.job_config.name.run_name,
        )
        self.logger = get_logger(self)
        # Create tracking subdir for current rolling experiment
        self.log_dir = os.path.join(
            self.recorder.get_artifact_uri(), str(self.repeat_index)
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger.info(f"Rolling root: {self.log_dir}")
        # Construct MongoDB task manager
        self.task_manager = TaskManager(
            config=config, task_pool=config.job.name.run_name
        )

    def start(
        self,
        is_subprocess: bool = False,
        task_func: tp.Callable = q4l_task_wrapper_fn,
    ) -> Recorder:
        self.recorder_wrapper.set(
            experiment_name=self.config.job.name.exp_name,
            recorder_name=self.config.job.name.run_name,
        )
        """Start the rolling-window experiment in a serial manner."""
        self.logger.info(f"Rolling runner #{self.repeat_index} starts.")
        # Environment setup
        self.reset()
        # Generate all the rolling window tasks
        rolling_cfgs = self._generate_tasks()
        self._run_rolling(rolling_cfgs, task_func=task_func)
        rolling_recorder = self._post_analysis()
        self.logger.info(f"Rolling runner #{self.repeat_index} all finished!")
        return rolling_recorder

    def reset(self):
        """The reset function is originally intended to look for duplicate
        experiment record in MongoDB and clean it if found.

        Here, we just log the info and do nothing since our naming convention
        prevents duplicate exp names.

        """
        # Original code
        # if isinstance(self.trainer, TrainerRM):
        #     TaskManager(task_pool=self.task_pool).remove()
        # exp = self.recorder_wrapper.get_exp(experiment_name=self.experiment_name)
        # for rid in exp.list_recorders():
        #     exp.delete_recorder(rid)
        self.logger.info("Task Reset (Actually nothing is done)")

    def collect(self):
        rec: MLflowRecorder = self.recorder_wrapper.get_recorder()
        artifact_root = rec.get_artifact_uri()
        root_dir = os.path.join(artifact_root, str(self.repeat_index))

        def rec_key(subdir_name: str):
            with open(
                os.path.join(root_dir, subdir_name, "exp_config.yaml"), "r"
            ) as f:
                OmegaConf.load(f)
            # model_key = exp_config["model"]["name"]
            return "rolling", subdir_name

        def model_rec_key(recorder):
            task_config = recorder.load_object("task.pkl")
            rolling_key = task_config["dataset"]["kwargs"]["segments"]["test"]
            return f"{str(rolling_key[0])[:10]}_{str(rolling_key[1])[:10]}"

        def my_filter(recorder):
            # Rolling run name protocol:
            # rolling_train_{repeat_index}_{interval}_{signature}
            run_tags = recorder.client.get_run(recorder.id).data.tags
            signature = run_tags.get("Signature", None)
            run_type = run_tags.get("Run type", None)
            repeat_index = run_tags.get("Repeat index", None)
            if run_type != "rolling_train":
                return False
            if repeat_index != str(self.repeat_index):
                return False
            if signature != self.signature:
                return False
            return True

        numerical_collector = RollingSubdirCollector(
            root_dir=root_dir,
            process_list=RollingGroup(),
            subdir_key_func=rec_key,
            subdir_filter_func=subdir_contain,
            artifacts_path={
                "pred": "sig_analysis/pred.pkl",
                "label": "sig_analysis/label.pkl",
            },
            artifacts_key=["pred", "label"],
        )

        model_collector = RollingSubdirCollector(
            root_dir=root_dir,
            subdir_key_func=rec_key,
            subdir_filter_func=subdir_contain,
            artifacts_path={
                "model": "model.pkl",
            },
            artifacts_key=["model"],
        )

        rolling_results = numerical_collector()
        model_results = model_collector()["model"]
        self.logger.info("Dumping rolling results")

        # Need to specify these things with prefix directory.
        # This is due to the mechanism in the class `ACRecordTemp`. It will
        # perform a check before generating any anaylsis.
        # So we need to put the things into the corresponding directories
        artifact_prefix = {"pred": "sig_analysis", "label": "sig_analysis"}

        rolling_recorder = self.recorder_wrapper.get_recorder()
        for k, v in rolling_results.items():
            key = list(v.keys())[0]
            file_suffix = os.path.join(
                self.recorder_wrapper.suffix, artifact_prefix[k]
            )
            # file_suffix = self.repeat_index + "/collect/" + artifact_prefix[k]
            rolling_recorder.save_objects(
                artifact_path=file_suffix, **{f"{k}.pkl": v[key]}
            )
        for timespan, model in model_results.items():
            file_suffix = os.path.join(self.recorder_wrapper.suffix, "model")
            rolling_recorder.save_objects(
                **{f"model-{timespan}.pkl": model}, artifact_path=file_suffix
            )
        self.logger.info(f"Collect finished.")
        return rolling_recorder

    def _generate_tasks(self) -> tp.List[ExperimentConfig]:
        self.logger.info("Task generating...")
        rolling_cfgs = generate_tasks(
            tasks=self.exp_config,
            generators=RollingGen(
                step=self.exp_config.time.rolling_step,
                rtype=self.exp_config.time.rolling_type,
            ),  # generate different date segments
        )
        self.logger.info(
            f"Rolling runner #{self.repeat_index} tasks generated."
        )
        with open(os.path.join(self.log_dir, "rolling_list.txt"), "w") as f:
            for cfg in rolling_cfgs:
                start = cfg.time.segments.test[0].start
                end = cfg.time.segments.test[0].end
                f.write(f"{start}~{end}\n")
        num_rollings = len(rolling_cfgs)
        self.recorder_wrapper.set_tags(num_rolling_per_repeat=num_rollings)
        return rolling_cfgs

    def _run_rolling(
        self, configs: tp.List[ExperimentConfig], task_func: tp.Callable
    ):
        self.logger.info(
            "Writing all rolling tasks to MongoDB "
            f"collection `{self.job_config.name.run_name}`"
        )
        _id_list = self.task_manager.create_task(
            configs,
            repeat_index=self.repeat_index,
            total_repeat=self.config.job.repeat.total_repeat,
        )
        self.logger.info(
            f"{len(configs)} tasks written to MongoDB, tasks start running"
        )
        self.task_manager.run_tasks(
            task_func=task_func,
            query={"_id": {"$in": _id_list}},
            num_workers=self.job_config.parallel.rolling,
            # task_func kwargs
            repeat_index=self.repeat_index,
            job_cfg=self.job_config,
            recorder_wrapper=self.recorder_wrapper,
        )
        self.logger.info(
            f"Rolling runner #{self.repeat_index} tasks training"
            "finished, collecting results"
        )

    def _post_analysis(self):
        self.recorder_wrapper.set_suffix(str(self.repeat_index))
        self.collect()
        rolling_recorder = self.recorder_wrapper.get_recorder()

        generate_evaluations(
            self.config,
            stage_key="rolling",
            recorder=rolling_recorder,
            logger=self.logger,
        )

        # Do some plotting, with rolling regions
        # self.plot_rolling_curve()

        self.logger.info(
            f"Rolling runner #{self.repeat_index} finished collecting"
        )

    def plot_rolling_curve(self):
        with open(
            os.path.join(
                self.recorder_wrapper.artifact_uri, "rolling_list.txt"
            ),
            "r",
        ) as f:
            interval_lines = f.readlines()
            intervals = []
            for line in interval_lines:
                interval_ends = line.strip().split("~")
                start, end = interval_ends[0], interval_ends[1]
                intervals.append((start, end))
        portfolio_report_df = pd.read_csv(
            os.path.join(
                self.recorder_wrapper.artifact_uri,
                "portfolio_analysis",
                "report_normal_1day.csv",
            ),
            index_col=0,
        )
        plot_return_curve(df=portfolio_report_df, intervals=intervals)


if __name__ == "__main__":
    ## to see the whole process with your own parameters, use the command below
    # python task_manager_rolling.py main --experiment_name="your_exp_name"
    fire.Fire(RollingTaskRunner)
