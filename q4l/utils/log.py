import logging
import logging.config
import os
import sys
import typing as tp
from datetime import datetime

import numpy as np
from pandas import Timestamp
from smart_open import smart_open as open

from ..qlib.log import QlibLogger, get_module_logger, set_log_with_config
from ..qlib.workflow.recorder import MLflowRecorder


class DualOutput:
    def __init__(self, stream_name, file_name):
        self.original_output = getattr(sys, stream_name)
        self.stream_name = stream_name
        setattr(
            sys, stream_name, self
        )  # Set the current instance as the system stream
        self.log_file = open(file_name, "a", buffering=1)  # line buffering

    @property
    def original_stream(self):
        if isinstance(self.original_output, DualOutput):
            return self.original_output.original_stream
        return self.original_output

    def write(self, data):
        self.original_stream.write(data)
        self.log_file.write(data)

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()

    def close(self):
        self.flush()
        self.log_file.close()
        setattr(sys, self.stream_name, self.original_output)


class LoggingContext:
    def __init__(self, is_debug=True, recorder_wrapper=None):
        self.is_debug = is_debug
        self.orig_loggers = {}
        self.modules_of_interest = ["q4l", "qlib"]
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        self.recorder_wrapper = recorder_wrapper
        # Store current logging configurations
        for module_name in self.modules_of_interest:
            logger = logging.getLogger(module_name)
            self.orig_loggers[module_name] = {
                "level": logger.level,
                "handlers": list(logger.handlers),
                "propagate": logger.propagate,
            }

    def __enter__(self):
        # Logging redirection setup
        redirect_logging(
            is_debug=self.is_debug, recorder_wrapper=self.recorder_wrapper
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore original stdout and stderr
        # sys.stdout = self.orig_stdout
        # sys.stderr = self.orig_stderr
        # if self.is_debug:
        sys.stdout.close()
        sys.stderr.close()

        # Reset and Restore original logging configurations
        for module_name in self.modules_of_interest:
            reset_loggers(module_name)  # Reset logger of interest
            logger = logging.getLogger(module_name)
            logger.setLevel(self.orig_loggers[module_name]["level"])
            for handler in self.orig_loggers[module_name]["handlers"]:
                logger.addHandler(handler)
            logger.propagate = self.orig_loggers[module_name]["propagate"]


def redirect_logging(is_debug: bool = False, recorder_wrapper=None):
    """Redirect stdout, stderr and logging file handler to the log file in the
    artifact directory of the current run.

    Parameters
    ----------
    recorder : MLflowRecorder
        The recorder of the current run, we need to get its artifact root.
    suffix : str
        The suffix to be appended to recorder's artifact root. The task subdirectory.
    is_debug : bool, optional
        Whether in debug mode or not. If in production mode, stdout and stderr will be redirected to the log file.

    """
    # Get the path of the log file, which is saved in the artifact directory of the current run.
    from ..qlib.workflow import R

    rec_wrapper = recorder_wrapper or R
    log_dir = rec_wrapper.artifact_uri
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")

    # if is_debug:
    sys.stderr = DualOutput("stderr", log_path)
    sys.stdout = DualOutput("stdout", log_path)

    if not os.path.exists(log_path):
        open(log_path, "w").close()

    get_logger("redirect_logging").info(f"Redirecting logging to {log_path}")

    # Redirect stdout and stderr to the log file, and add logging file handlers
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    reset_loggers("q4l")
    reset_loggers("qlib")
    logging.config.dictConfig(
        make_logconfig(logpath=log_path, is_debug=is_debug, root_module="q4l")
    )
    logging.config.dictConfig(
        make_logconfig(logpath=log_path, is_debug=is_debug, root_module="qlib")
    )


def reset_loggers(module_name: str):
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith(module_name):
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


def init_q4l_logging(logpath: str = None, is_debug_mode: bool = False):
    set_log_with_config(
        log_config=make_logconfig(logpath=logpath, is_debug=is_debug_mode)
    )


def get_logger(
    module: tp.Union[object, tp.Text, tp.Callable],
    log_level: int = logging.INFO,
    prefix: str = "q4l",
    recorder: MLflowRecorder = None,
    log_filepath: str = None,
) -> QlibLogger:
    if isinstance(module, tp.Callable):
        # Since objects with `__call__` method will also be recognized, we apply an ugly
        # hack here.
        try:
            module_name = f"{module.__module__}.{module.__name__}"
        except:
            module_name = (
                f"{module.__class__.__module__}.{module.__class__.__name__}"
            )
    elif isinstance(module, tp.Text):
        module_name = f"q4l.{module}"
    else:
        module_name = (
            f"{module.__class__.__module__}.{module.__class__.__name__}"
        )

    logger = get_module_logger(module_name, level=log_level, prefix=prefix)
    if recorder is not None:
        file_handler = logging.FileHandler(
            recorder.get_artifact_uri() + "/log.txt"
        )
        formatter = logging.Formatter(
            "[%(process)s:%(threadName)s](%(asctime)s) %(levelname)s"
            " - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.logger.addHandler(file_handler)
    if log_filepath is not None:
        file_handler = logging.FileHandler(log_filepath)
        formatter = logging.Formatter(
            "[%(process)s:%(threadName)s](%(asctime)s) %(levelname)s"
            " - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.logger.addHandler(file_handler)
    return logger


def make_logconfig(
    logpath: str = None, is_debug: bool = False, root_module: str = "q4l"
) -> tp.Dict:
    formatter_string = (
        "[%(process)s:%(threadName)s](%(asctime)s) %(levelname)s"
        " - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    # logger_handlers = ["console"]
    logger_handlers = []
    # if logpath is not None:
    #     logger_handlers.append("logfile")
    logger_handlers.append("console")
    log_level = logging.DEBUG if is_debug else logging.INFO

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"logger_format": {"format": formatter_string}},
        "filters": {
            "field_not_found": {
                "()": "q4l.qlib.log.LogFilter",
                "param": [".*?WARN: data not found for.*?"],
            }
        },
        "handlers": {
            "logfile": {
                "class": "logging.FileHandler",
                "filename": logpath or "log.txt",
                "level": logging.DEBUG,
                "formatter": "logger_format",
                "filters": ["field_not_found"],
            },
            "console": {
                "class": "logging.StreamHandler",
                "level": logging.DEBUG,
                "formatter": "logger_format",
                "filters": ["field_not_found"],
            },
        },
        "loggers": {
            root_module: {
                "level": log_level,
                "handlers": logger_handlers,
                "propagate": False,
            },
        },
    }


def serialize_value(value: tp.Any) -> tp.Any:
    def convert_datetime(obj: tp.Union[datetime, Timestamp]) -> str:
        return obj.strftime("%Y%m%d")

    def convert_ndarray(obj: np.ndarray) -> list:
        return obj.tolist()

    def convert_np_scalar(
        obj: tp.Union[np.integer, np.floating, np.bool_]
    ) -> tp.Union[int, float, bool]:
        return obj.item()

    def convert_dict(obj: dict) -> dict:
        return {k: serialize_value(v) for k, v in obj.items()}

    def convert_list(obj: list) -> list:
        return [serialize_value(v) for v in obj]

    def convert_tuple(obj: tuple) -> tuple:
        return tuple(serialize_value(v) for v in obj)

    type_conversion_map = {
        (datetime, Timestamp): convert_datetime,
        np.ndarray: convert_ndarray,
        (np.integer, np.floating, np.bool_): convert_np_scalar,
        dict: convert_dict,
        list: convert_list,
        tuple: convert_tuple,
    }

    for types, converter in type_conversion_map.items():
        if isinstance(value, types):
            return converter(value)

    return value


def recursive_sort(d: tp.Dict):
    """Recursively sort the dictionary."""
    if isinstance(d, dict):
        return {k: recursive_sort(v) for k, v in sorted(d.items())}
    elif isinstance(d, list):
        return [recursive_sort(v) for v in d]
    else:
        return d
