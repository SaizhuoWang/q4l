import os
import re

import git


def is_git_repo(path: str) -> bool:
    """Check if the given path is under a Git repository.

    :param path: The path to check.
    :return: True if the path is under a Git repository, False otherwise.

    """
    # Change the current working directory to the given path
    current_path = os.getcwd()
    os.chdir(path)

    # Check if the path is under a Git repository
    try:
        result = os.system(
            "git rev-parse --is-inside-work-tree > /dev/null 2>&1"
        )
        return result == 0
    finally:
        # Change the working directory back to the original one
        os.chdir(current_path)


# ------------------ Default paths ------------------ #\
if is_git_repo(os.getcwd()):
    PROJECT_ROOT = git.Repo(
        os.getcwd(), search_parent_directories=True
    ).git.rev_parse("--show-toplevel")
else:
    PROJECT_ROOT = os.getcwd()

DEFAULT_LOG_ROOT = os.path.join(PROJECT_ROOT, "logs")
DEFAULT_SLURM_LOGDIR = os.path.join(PROJECT_ROOT, "slurm_logs")
DEFAULT_CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

# ------------------ Default values ------------------ #
# Default rolling step for rolling retraining
DEFAULT_ROLLING_STEP = 130
FACTOR_TOKEN_RE = re.compile(r"[{]([a-zA-Z]+\:)?([a-zA-Z_][a-zA-Z0-9_]*)[}]")
FACTOR_TOKEN_SUB_RE = re.compile(r"([{]([a-zA-Z0-9_]*)[}])")
TICK_FORMAT = "%Y%m%d"
INDEX_LEVEL1_NAME = "datetime"
INDEX_LEVEL2_NAME = "instrument"
COLUMN_LEVEL1_NAME = 0
COLUMN_LEVEL2_NAME = 1
MAX_THREAD_WORKERS = 30


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_ROLLING_STEP",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_SLURM_LOGDIR",
]
