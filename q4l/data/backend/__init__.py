import abc
import importlib
import typing as tp

import pandas as pd

from ...config import GlobalConfig, ModuleConfig

class Backend:
    type: str
    message = "I am the base class `Backend`, find this method in my children."

    @abc.abstractmethod
    def get_factor(self, factor_name):
        raise NotImplementedError(self.message)

    @abc.abstractclassmethod
    def compute(self, data_dict) -> pd.DataFrame:
        raise NotImplementedError(self.message)


class BackendFactory:
    @staticmethod
    def create(config: ModuleConfig, return_class_only: bool = False) -> Backend:
        backend_name = config.name
        backend_class = getattr(importlib.import_module(config.module_path), backend_name)
        return backend_class(**config.kwargs) if not return_class_only else backend_class


def get_backend(config: GlobalConfig, factor_name: str) -> str:
    """This function automatically determines the storage backend for a specific
    factor, based on the global config information at runtime and the factor
    name itself.

    Parameters
    ----------
    config : GlobalConfig
        The global config object that is parsed at runtime.
    factor_name : str
        Factor name in stripped form, e.g. "open".

    Returns
    -------
    str
        A backend name, e.g. "disk", "mysql", "factor_hub".

    Raises
    ------
    ValueError
        If no appropriate backend is found for this factor.

    """
    if config.experiment.region.name != "cn":
        return "disk"
    factor_backend_mapping = {
        "factor_hub": ["disk", "disk", "disk"],
        "ideadata": ["disk", "disk", "disk"],
        "mysql": ["mysql", "mysql", "mysql"],
    }

    for k, v in factor_backend_mapping.items():
        if factor_name in v:
            return k

    raise ValueError(
        f"Factor {factor_name} in region {config.experiment.region.name} not found in any backend"
    )

# from . import storage, compute