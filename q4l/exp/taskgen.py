# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""TaskGenerator module can generate many tasks based on TaskGen and some task
templates."""
import abc
import copy
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Union

import omegaconf
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..config import (
    ExperimentConfig,
    SegmentConfig,
    TimeInterval,
    create_timeinterval_from_datetime,
    hydra_datetime_formatter,
)
from ..qlib.utils import transform_end_date
from ..qlib.workflow.task.utils import TimeAdjuster
from ..utils.log import get_logger


class TaskGen(metaclass=abc.ABCMeta):
    """The base class for generating different tasks.

    Example 1:

        input: a specific task template and rolling steps

        output: rolling version of the tasks

    Example 2:

        input: a specific task template and losses list

        output: a set of tasks with different losses

    """

    @abc.abstractmethod
    def generate(self, task: dict) -> List[dict]:
        """Generate different tasks based on a task template.

        Parameters
        ----------
        task: dict
            a task template

        Returns
        -------
        typing.List[dict]:
            A list of tasks

        """

    def __call__(self, *args, **kwargs):
        """This is just a syntactic sugar for generate."""
        return self.generate(*args, **kwargs)


def handler_mod(exp_config: ExperimentConfig, rolling_gen: "RollingGen"):
    """Help to modify the handler end time when using RollingGen It try to
    handle the following case.

    - Hander's data end_time is earlier than  dataset's test_data's segments.

        - To handle this, handler's data's end_time is extended.

    If the handler's end_time is None, then it is not necessary to change it's end time.

    Args:
        task (dict): a task template
        rg (RollingGen): an instance of RollingGen

    """
    try:
        test_seg_endtime = getattr(
            exp_config.time.segments, rolling_gen.test_key
        )[0][
            1
        ]  # [0] for the first and only segment (in the context of rolling), [1] for the end time
        interval = rolling_gen.ta.cal_interval(
            exp_config.time.end_time,
            test_seg_endtime,
        )
        # if end_time < the end of test_segments, then change end_time to allow load more data
        if interval < 0:
            exp_config.time.end_time = copy.deepcopy(test_seg_endtime)
    except KeyError:
        # Maybe dataset do not have handler, then do nothing.
        pass
    except TypeError:
        # May be the handler is a string. `"handler.pkl"["kwargs"]` will raise TypeError
        # e.g. a dumped file like file:///<file>/
        pass


def trunc_segments(
    ta: TimeAdjuster, segments: Dict[str, pd.Timestamp], days, test_key="test"
):
    """To avoid the leakage of future information, the segments should be
    truncated according to the test start_time.

    NOTE:
        This function will change segments **inplace**

    """
    # adjust segment
    test_start = min(t for t in segments[test_key] if t is not None)
    for k in list(segments.keys()):
        if k != test_key:
            segments[k] = ta.truncate(segments[k], test_start, days)


def timedelta_to_int(tdelta, freq):
    seconds = tdelta.total_seconds()
    if freq == "days":
        return int(seconds // 86400)  # Convert to days
    elif freq == "hours":
        return int(seconds // 3600)  # Convert to hours
    elif freq == "minutes":
        return int(seconds // 60)  # Convert to minutes
    elif freq == "seconds":
        return int(seconds)  # Return seconds
    else:
        raise ValueError(
            "Invalid frequency specified. Valid options are: 'days', 'hours', 'minutes', 'seconds'."
        )


class RollingGen(TaskGen):
    ROLL_EX = TimeAdjuster.SHIFT_EX  # fixed start date, expanding end date
    ROLL_SD = (
        TimeAdjuster.SHIFT_SD
    )  # fixed segments size, slide it from start date
    ROLL_ON = (
        TimeAdjuster.SHIFT_ON
    )  # Following segments only contain new training data (no overlap)
    ROLL_TA = (
        TimeAdjuster.SHIFT_TA
    )  # Throw this whole segment away to the next whole segment (no overlap)

    def __init__(
        self,
        step: int = 40,
        rtype: str = ROLL_EX,
        ds_extra_mod_func: Union[None, Callable] = handler_mod,
        test_key="test",
        train_key="train",
        valid_key="valid",
        trunc_days: int = None,
        task_copy_func: Callable = copy.deepcopy,
    ):
        """Generate tasks for rolling.

        Parameters
        ----------
        step : int
            step to rolling
        rtype : str
            rolling type (expanding, sliding)
        ds_extra_mod_func: Callable
            A method like: handler_mod(task: dict, rg: RollingGen)
            Do some extra action after generating a task. For example, use ``handler_mod`` to modify the end time of the handler of a dataset.
        trunc_days: int
            trunc some data to avoid future information leakage
        task_copy_func: Callable
            the function to copy entire task. This is very useful when user want to share something between tasks

        """
        self.logger = get_logger(self)
        self.step = step
        self.rtype = rtype
        self.ds_extra_mod_func = ds_extra_mod_func
        self.ta = TimeAdjuster(future=True)

        self.test_key = test_key
        self.valid_key = valid_key
        self.train_key = train_key
        self.split_keys = [self.train_key, self.valid_key, self.test_key]

        self.trunc_days = trunc_days
        self.task_copy_func = task_copy_func

    def _update_task_segs(
        self, task: ExperimentConfig, segs: Dict[str, Tuple[datetime, datetime]]
    ):
        """Segs: {"train": (start, end), "valid": (start, end), "test": (start, end)}"""
        # update segments of this task
        def tuple_to_timeinterval(t: Tuple[datetime, datetime]):
            return TimeInterval(
                start=hydra_datetime_formatter(t[0]),
                end=hydra_datetime_formatter(t[1]),
            )

        for k, v in segs.items():
            setattr(
                task.time.segments,
                k,
                OmegaConf.create([tuple_to_timeinterval(v)]),
            )

        # update segments of handler
        segs_clone = copy.deepcopy(segs)
        task.time.fit_start_time = segs_clone["train"][0]
        task.time.fit_end_time = segs_clone["train"][1]
        task.time.start_time = segs_clone["train"][0]
        task.time.end_time = segs_clone["test"][1]

        if self.ds_extra_mod_func is not None:
            self.ds_extra_mod_func(task, self)

    def gen_following_tasks(
        self, exp_config: ExperimentConfig, test_end: pd.Timestamp
    ) -> List[dict]:
        """Generating following rolling tasks for `task` until test_end.

        Parameters
        ----------
        task : dict
            Qlib task format
        test_end : pd.Timestamp
            the latest rolling task includes `test_end`

        Returns
        -------
        List[dict]:
            the following tasks of `task`(`task` itself is excluded)

        """
        prev_seg: SegmentConfig = exp_config.time.segments

        # Sanity check: each value in segments should be a list with one only element of type TimeInterval
        for k in self.split_keys:
            if len(getattr(prev_seg, k)) != 1:
                raise ValueError(
                    "For rolling tasks, each segment should only have one element"
                )
            # if not isinstance(getattr(prev_seg, k)[0], TimeInterval):
            #     raise TypeError(
            #         "For rolling tasks, each segment should only have one element of type TimeInterval"
            #     )

        while True:
            segments = {}
            try:
                for k in self.split_keys:
                    # decide how to shift
                    # expanding only for train data, the segments size of test data and valid data won't change
                    seg = getattr(prev_seg, k)[0]
                    seg = (seg.start, seg.end)
                    if k == self.train_key and self.rtype == self.ROLL_EX:
                        rtype = self.ta.SHIFT_EX
                    elif k == self.train_key and self.rtype == self.ROLL_ON:
                        rtype = self.ta.SHIFT_ON
                    else:
                        rtype = self.ta.SHIFT_SD

                    if self.rtype == self.ROLL_TA:
                        pass

                        step = timedelta_to_int(
                            prev_seg.test[0].end - prev_seg.train[0].start,
                            freq="days",
                        )
                    else:
                        step = self.step

                    # shift the segments data
                    segments[k] = self.ta.shift(seg, step=step, rtype=rtype)
                if segments[self.test_key][0] > test_end:
                    break
                if (
                    segments[self.test_key][1] is None
                    or segments[self.test_key][1] > test_end
                ):
                    segments[self.test_key] = (
                        segments[self.test_key][0],
                        test_end,
                    )
            except KeyError:
                # We reach the end of tasks
                # No more rolling
                break

            prev_seg = SegmentConfig(
                train=OmegaConf.create(
                    [
                        create_timeinterval_from_datetime(
                            *segments[self.train_key]
                        )
                    ]
                ),
                valid=OmegaConf.create(
                    [
                        create_timeinterval_from_datetime(
                            *segments[self.valid_key]
                        )
                    ]
                ),
                test=OmegaConf.create(
                    [
                        create_timeinterval_from_datetime(
                            *segments[self.test_key]
                        )
                    ]
                ),
            )
            self.logger.info(
                f"Generating rolling task:\n{self.segment_to_string(segments=segments)}"
            )
            new_config = self.task_copy_func(
                exp_config
            )  # deepcopy is necessary to avoid replace task inplace
            self._update_task_segs(new_config, segments)
            yield new_config

    def check_segments(self, task: ExperimentConfig) -> bool:
        for split_name in self.split_keys:
            segment_list = getattr(task.time.segments, split_name)
            if len(segment_list) > 1:
                return False

    def generate(self, exp_config: ExperimentConfig) -> List[ExperimentConfig]:
        """Converting the task into a rolling task.

        Parameters
        ----------
        task: dict
            A dict describing a task. For example.

            .. code-block:: python

                DEFAULT_TASK = {
                    "model": {
                        "class": "LGBModel",
                        "module_path": "qlib.contrib.model.gbdt",
                    },
                    "dataset": {
                        "class": "DatasetH",
                        "module_path": "qlib.data.dataset",
                        "kwargs": {
                            "handler": {
                                "class": "Alpha158",
                                "module_path": "qlib.contrib.data.handler",
                                "kwargs": {
                                    "start_time": "2008-01-01",
                                    "end_time": "2020-08-01",
                                    "fit_start_time": "2008-01-01",
                                    "fit_end_time": "2014-12-31",
                                    "instruments": "csi100",
                                },
                            },
                            "segments": {
                                "train": ("2008-01-01", "2014-12-31"),
                                "valid": (
                                    "2015-01-01",
                                    "2016-12-20",
                                ),  # Please avoid leaking the future test data into validation
                                "test": ("2017-01-01", "2020-08-01"),
                            },
                        },
                    },
                    "record": [
                        {
                            "class": "SignalRecord",
                            "module_path": "qlib.workflow.record_temp",
                        },
                    ],
                }

        Returns
        ----------
        List[dict]: a list of tasks

        """

        # Sanity check
        if self.check_segments(exp_config) is False:
            raise ValueError(
                "The segments of task is not valid. Rolling task only supports one segment for each split."
            )

        res = []

        cfg_clone = self.task_copy_func(exp_config)

        # calculate segments

        # First rolling
        def to_tuple(time_interval: DictConfig) -> Tuple[datetime, datetime]:
            # Convert a TimeInterval that has been parsed by OmegaConf to a tuple of datetime.
            interval_dict = OmegaConf.to_container(time_interval, resolve=True)
            return (interval_dict["start"], interval_dict["end"])

        # 1) prepare the end point
        segments: dict = copy.deepcopy(
            self.ta.align_seg(
                {
                    self.train_key: to_tuple(exp_config.time.segments.train[0]),
                    self.valid_key: to_tuple(exp_config.time.segments.valid[0]),
                    self.test_key: to_tuple(exp_config.time.segments.test[0]),
                }
            )
        )
        test_end = transform_end_date(segments[self.test_key][1])
        # 2) and init test segments
        test_start_idx = self.ta.align_idx(segments[self.test_key][0])

        # test_end_1 = self.ta.get(test_start_idx + self.step - 1)
        # test_end_2 = test_end

        # segments[self.test_key] = TimeInterval(
        #     start=self.ta.get(test_start_idx),
        #     end=min(self.ta.get(test_start_idx + self.step - 1) or test_end, test_end),
        # )

        segments[self.test_key] = (
            self.ta.get(test_start_idx),
            min(
                self.ta.get(test_start_idx + self.step - 1) or test_end,
                test_end,
            ),
        )

        if self.trunc_days is not None:
            trunc_segments(self.ta, segments, self.trunc_days, self.test_key)

        # update segments of this task
        self._update_task_segs(cfg_clone, segments)

        # Print out the first rolling task time segments

        first_segment_info = self.segment_to_string(segments)
        self.logger.info(
            f"First rolling task time segments:\n{first_segment_info}"
        )

        res.append(cfg_clone)

        # Update the following rolling
        res.extend(self.gen_following_tasks(cfg_clone, test_end))
        return res

    def segment_to_string(self, segments):
        return (
            f"Train: {segments[self.train_key][0]} - {segments[self.train_key][1]}\n"
            f"Valid: {segments[self.valid_key][0]} - {segments[self.valid_key][1]}\n"
            f"Test: {segments[self.test_key][0]} - {segments[self.test_key][1]}\n"
        )


def generate_tasks(
    tasks: Union[ExperimentConfig, List[ExperimentConfig]],
    generators: Union[TaskGen, List[TaskGen]],
) -> List[ExperimentConfig]:
    """Use a list of TaskGen and a list of task templates to generate different
    tasks.

    For examples:

        There are 3 task templates a,b,c and 2 TaskGen A,B. A will generates 2 tasks from a template and B will generates 3 tasks from a template.
        task_generator([a, b, c], [A, B]) will finally generate 3*2*3 = 18 tasks.

    Parameters
    ----------
    tasks : List[dict] or dict
        a list of task templates or a single task
    generators : List[TaskGen] or TaskGen
        a list of TaskGen or a single TaskGen

    Returns
    -------
    list
        a list of tasks

    """

    if isinstance(tasks, omegaconf.DictConfig):
        tasks = [tasks]
    if isinstance(generators, TaskGen):
        generators = [generators]

    # generate gen_task_list
    for gen in generators:
        new_task_list = []
        for task in tasks:
            new_task_list.extend(gen.generate(task))
        tasks = new_task_list

    return tasks
