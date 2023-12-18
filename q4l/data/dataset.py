"""Dataset is wrapper of data handler and data sampler."""

from copy import copy
from typing import List, Text, Union

import pandas as pd
from lightning.pytorch.core import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from omegaconf import open_dict
from torch.utils.data import DataLoader

from ..config import ExperimentConfig, JobConfig
from ..qlib.utils.serial import Serializable
from ..utils.log import get_logger
from .handler import CS_ALL, CS_RAW, DK_I, DK_L, Q4LDataHandler
from .sampler import TCDataSampler

INDEX_LEVEL1_NAME = "datetime"
INDEX_LEVEL2_NAME = "instrument"
COLUMN_LEVEL1_NAME = 0
COLUMN_LEVEL2_NAME = 1
MAX_THREAD_WORKERS = 30


class Q4LDataset(Serializable):
    """A wrapper for supporting dataset split.

    This class the the `bare-metal` version of `Q4LDataModule`. If you want
    lower-level control of your data, use this class instead of `Q4LDataModule`.

    """

    def __init__(
        self,
        exp_config: ExperimentConfig,
        job_config: JobConfig,
    ):
        self.logger = get_logger(self)
        self.handler = Q4LDataHandler(
            exp_config=exp_config,
            job_config=job_config,
        )
        # self.segments = copy(OmegaConf.to_container(cfg.time.segments))
        self.segments = copy(exp_config.time.segments)
        self.logger.info(
            f"Successfully initialized handler {self.handler}. Now setup data."
        )
        use_cache = job_config.misc.get("use_disk_cache", True)
        self.handler.setup_data(use_cache=use_cache)
        super().__init__()

    def rewrite_model_config(self, cfg: ExperimentConfig):
        cfg.model.input_size = self.handler.data_dim

    @property
    def trade_calendar(self):
        return self.handler.trade_calendar

    def prepare(
        self,
        partition: Text,
        col_set: Text = CS_ALL,
        data_key: Text = DK_I,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        # Retrieve dataframes for each segment in the specified partition
        segment_dfs = []
        for segment in getattr(
            self.segments, partition
        ):  # Each segment may contain multiple intervals
            segment_df = self.handler.fetch(
                segment=segment, col_set=col_set, data_key=data_key
            )
            segment_dfs.append(segment_df)

        # Concatenate the dataframes along the row axis (axis=0)
        concatenated_df = pd.concat(segment_dfs, axis=0)

        # Return the concatenated dataframe
        return concatenated_df


class DataModuleWrapper(LightningDataModule):
    def __init__(self, dataset: Q4LDataset):
        super().__init__()
        self.dataset = dataset

    def prepare_data(self):
        self.dataset.handler.setup_data()

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.dataset.handler.get_dataloader(partition="train")

    def val_dataloader(self):
        return self.dataset.handler.get_dataloader(partition="val")

    def test_dataloader(self):
        return self.dataset.handler.get_dataloader(partition="test")

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()


class Q4LDataModule(LightningDataModule):
    """This class is the wrapped version of `Q4LDataset`.

    It is designed to be used in cope with pytorch-lightning. If you want lower-
    level control of your data, use `Q4LDataset` instead.

    """

    def __init__(self, exp_config: ExperimentConfig, job_config: JobConfig):
        super().__init__()

        self.cfg = exp_config
        self.logger = get_logger(self)

        self.logger.info(f"Initializing data handler with config")
        self.handler = Q4LDataHandler(
            exp_config=exp_config,
            job_config=job_config,
        )
        self.segments = copy(exp_config.time.segments)

        self.logger.info(
            f"Successfully initialized handler {self.handler}. Now setup data."
        )
        self.handler.setup_data()
        self.ticker_list = self.handler.ticker_list
        self.rewrite_model_config(exp_config)

        # Setup some handy attributes
        self.batch_size = exp_config.data.sampler.batch_size

    def rewrite_model_config(self, cfg: ExperimentConfig):
        """Some fields in experiment config need to be rewritten at runtime
        after data is loaded. This method is used to do that. As for the
        specific fields, you may override this method in your own cases.

        Parameters
        ----------
        cfg : ExperimentConfig
            The experiment config to be rewritten.

        """
        # De-set the struct flag of config to allow modification
        with open_dict(cfg):
            # Overwrite model input dimension
            x_data = self.handler.learn_data[self.cfg.data.sampler.x_group]
            cfg.model.input_size = x_data.shape[1]

            # # Overwrite pre-processor's feature group
            # for processor_name in ["shared", "learn", "infer"]:
            #     processor_config_list = getattr(
            #         cfg.data.preprocessor, processor_name
            #     )
            #     for processor_config in processor_config_list:
            #         processor_config.kwargs.fields_group = (
            #             cfg.data.sampler.x_group
            #         )

    @property
    def trade_calendar(self):
        return self.handler.trade_calendar

    def prepare_data(self):
        self.handler.setup_data()

    def setup(self, stage=None):
        pass

    def _make_dataloader(self, sampler):
        return DataLoader(
            sampler,
            batch_size=self.batch_size,
            collate_fn=sampler.collate,
        )

    def train_dataloader(self):
        sampler = self.prepare(
            partition="train", data_key=DK_L, return_sampler=True
        )
        return self._make_dataloader(sampler)

    def val_dataloader(self):
        sampler = self.prepare(
            partition="valid", data_key=DK_I, return_sampler=True
        )
        return self._make_dataloader(sampler)

    def test_dataloader(self):
        sampler = self.prepare(
            partition="test", data_key=DK_I, return_sampler=True
        )
        return self._make_dataloader(sampler)

    def predict_dataloader(self):
        # sampler = self.prepare(partition="test", return_sampler=True)
        return self.test_dataloader()

    def prepare(
        self,
        partition: Text,
        col_set: Text = CS_RAW,
        data_key: Text = DK_I,
        return_sampler: bool = False,
    ) -> Union[TCDataSampler, pd.DataFrame]:
        # Retrieve dataframes for each segment in the specified partition
        self.logger.info(
            f"Fetching partition {partition}. Column set is {col_set}. Data key is {data_key}"
        )
        segment_dfs = []
        nan_masks = []
        # self.logger.info(
        #     f"Memory profile before preparation at dataset  0x00: {display_memory_tree(ProcessNode(psutil.Process()))}"
        # )

        # with TimeInspector.logt(f"Fetching partition {partition}", show_start=True):
        for segment in getattr(
            self.segments, partition
        ):  # Each segment may contain multiple intervals
            segment_df, nan_mask = self.handler.fetch(
                segment=segment, col_set=col_set, data_key=data_key
            )
            segment_dfs.append(segment_df)
            nan_masks.append(nan_mask)
        # memory_profile = display_memory_tree(ProcessNode(psutil.Process()))
        # self.logger.info(f"Memory profile after fetching: {memory_profile}")

        # with TimeInspector.logt(f"Concatenating partition {partition}", show_start=True):
        # Concatenate the dataframes along the row axis (axis=0)
        concatenated_df = pd.concat(segment_dfs, axis=0)
        nan_mask_concat = pd.concat(nan_masks, axis=0)

        # Return the concatenated dataframe
        is_inference = data_key == DK_I
        if return_sampler:
            return TCDataSampler(
                concatenated_df,
                nan_mask_concat,
                config=self.cfg.data.sampler,
                is_inference=is_inference,
            )
        else:
            return concatenated_df, nan_mask_concat
