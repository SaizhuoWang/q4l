from unittest.mock import MagicMock

import pandas as pd
import pytest

from q4l.config import AlphaGroupConfig, BackendConfig, ModuleConfig
from q4l.data.loader import (
    BackendFactory,
    ComputeBackend,
    LoaderConfig,
    Q4LDataLoader,
)


def create_mock_loader_config() -> LoaderConfig:
    myhxdf_config = ModuleConfig(
        name="MyHXDFComputeBackend",
        module_path="q4l.data.backend.compute",
        kwargs={},
    )
    disk_config = ModuleConfig(
        name="FileSystemBackend",
        module_path="q4l.data.backend.storage",
        kwargs={
            "root_dir": "/wsz/Data/my_data_dir/main/jp",
            "frequency": "day",
        },
    )
    backend_config = BackendConfig(
        compute={"my_hxdf": myhxdf_config}, storage={"disk": disk_config}
    )
    alpha_group_config = AlphaGroupConfig(
        name="example_group",
        compute_backend="my_hxdf",
        expressions={
            "alpha1": "${disk:open}+${disk:close}",
            "alpha2": "${disk:high}+${disk:low}",
        },
    )
    alpha_group_config_2 = AlphaGroupConfig(
        name="example_group_2",
        compute_backend="my_hxdf",
        expressions={
            "alpha1": "${disk:open}+${disk:high}",
            "alpha2": "${disk:high}+${disk:close}",
        },
    )
    loader_config = LoaderConfig(
        backend=backend_config, alpha=[alpha_group_config, alpha_group_config_2]
    )
    return loader_config


def create_mock_backend_factory() -> BackendFactory:
    backend_factory = BackendFactory()
    backend_factory.create = MagicMock(return_value=MagicMock(spec=ComputeBackend))
    return backend_factory


@pytest.fixture
def q4l_loader_wrapper() -> Q4LDataLoader:
    loader_config = create_mock_loader_config()
    q4l_loader_wrapper = Q4LDataLoader(loader_config)
    return q4l_loader_wrapper


def test_load(q4l_loader_wrapper: Q4LDataLoader):
    result = q4l_loader_wrapper.load(
        instruments="nikkei225",
        start_time="20190101",
        end_time="20200101",
    )

    assert isinstance(result, pd.DataFrame)
