import typing as tp
from datetime import datetime

import yaml
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.config_store import ConfigStore
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import MISSING as OMEGACONF_MISSING
from omegaconf import OmegaConf

from . import constants, data, eval, exp, model, utils


class MyHydraSearchPathPlugin(SearchPathPlugin):
    """Add the default YAML file dir shipped with this library to Hydra's search
    path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="q4l", path="pkg://q4l.config.q4l_builtin")


def read_factor_list(factor_list_path: str) -> tp.Dict[str, str]:
    with open(factor_list_path, "r") as f:
        factor_list = yaml.safe_load(f)
    return factor_list


def timestamp_atinit() -> str:
    return datetime.now().isoformat()


Plugins.instance().register(MyHydraSearchPathPlugin)
OmegaConf.register_new_resolver("timestamp", datetime.fromisoformat)
OmegaConf.register_new_resolver("timestamp_now", datetime.now().strftime)
OmegaConf.register_new_resolver(
    "timestamp_atinit", datetime.now().isoformat, use_cache=True
)
OmegaConf.register_new_resolver("factor_list", read_factor_list)
