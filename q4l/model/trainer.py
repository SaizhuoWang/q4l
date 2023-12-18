from typing import Union

import pandas as pd
from lightning.pytorch.core import LightningModule
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities.types import (
    _PREDICT_OUTPUT,
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)

from .base import NonDLModel, QuantModel


class Q4LTrainer(Trainer):
    """A wrapper around the PyTorch Lightning Trainer class in Q4L."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _update_predict_output(pred_output: _PREDICT_OUTPUT) -> pd.Series:
        """Take all `pred_series` values and concatenate them into a big series
        with sorted index."""
        pred_series_list = [output["pred_series"] for output in pred_output]
        pred_series = pd.concat(pred_series_list).sort_index()
        return pred_series

    def fit(
        self,
        model: QuantModel,
        train_dataloaders: Union[
            TRAIN_DATALOADERS, LightningDataModule, None
        ] = None,
        val_dataloaders: Union[EVAL_DATALOADERS, None] = None,
        datamodule: Union[LightningDataModule, None] = None,
        ckpt_path: Union[str, None] = None,
        use_rl: bool = False,
    ) -> None:
        # For supervised/unsupervised learning, use the original fit method
        if not isinstance(model, NonDLModel):
            return super().fit(
                model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
            )
        else:
            # For non-DL models, pytorch-lightning is not used
            return model.fit(datamodule)

    def predict(
        self,
        model: Union[LightningModule, None] = None,
        dataloaders: Union[EVAL_DATALOADERS, LightningDataModule, None] = None,
        datamodule: Union[LightningDataModule, None] = None,
        return_predictions: Union[bool, None] = None,
        ckpt_path: Union[str, None] = None,
    ) -> Union[_PREDICT_OUTPUT, None]:
        if not isinstance(model, NonDLModel):
            return super().predict(
                model, dataloaders, datamodule, return_predictions, ckpt_path
            )
        else:
            return model.predict(datamodule)
