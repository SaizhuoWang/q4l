import itertools
import os
import typing as tp
from abc import abstractmethod
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import ExperimentConfig
from ..data.dataset import Q4LDataModule
from ..eval.metrics.regression import rmse
from ..eval.metrics.signal import ic, signal_return, signal_turnover
from ..qlib.workflow import R
from ..utils.log import get_logger
from ..utils.misc import create_instance

# Some constant keys
INPUT_KEY = "x"
LABEL_KEY = "y"
TEMPORAL_EMBEDDING_KEY = "temporal_emb"
SPATIAL_EMBEDDING_KEY = "spatial_emb"
EMBEDDING_KEY = "emb"
PREDICTION_KEY = "pred"
WEIGHT_KEY = "weight"


class QuantModel(pl.LightningModule):
    def __init__(self, config: ExperimentConfig, data: Q4LDataModule):
        super().__init__()
        self.config: ExperimentConfig = config
        self.data: Q4LDataModule = data
        self.q4l_logger = get_logger(self)
        self.build_model_arch()

    # Setup
    @abstractmethod
    def build_model_arch(self):
        raise NotImplementedError("Please implement this abstract method!")

    @property
    def head_input_dim(self):
        """Compute the input dimension of the prediction head.

        Its input can come from multiple sources, e.g. temporal embedding,
        spatial embedding.

        """
        raise NotImplementedError("Please implement this abstract method.")


class NonDLModel:
    def __init__(self, config: ExperimentConfig, **kwargs) -> None:
        self.actual_model = create_instance(
            config.model.components.actual_model
        )

    def fit(self, *args, **kwargs):
        self.actual_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        pred, label = self.actual_model.predict(*args, **kwargs)
        self.dump_predict_results(pred, label)

    def dump_predict_results(self, pred_series, y_series):
        self.predict_pred_series = pred_series
        self.predict_y_series = y_series

        R.save_objects(
            artifact_path="sig_analysis",
            **{"pred.pkl": pred_series, "label.pkl": y_series},
        )
        current_logpath = R.artifact_uri
        csv_dump_path = os.path.join(current_logpath, "outputs")
        os.makedirs(csv_dump_path, exist_ok=True)
        pred_df = pred_series.unstack()
        y_df = y_series.unstack()
        pred_df.to_csv(csv_dump_path + "/pred.csv")
        y_df.to_csv(csv_dump_path + "/label.csv")


class TimeSeriesModel(QuantModel):
    """Base time-series forecasting model for quant modelling. Basic
    architecture:

        [Input] -> <Temporal Model> -> [Embedding] -> <Prediction Head> -> [Output]

    The forward process will be like:
        embedding = self.temporal_model(input)
        output = self.prediction_head(embedding)
        return output

    where both input, embedding and output are flexible dictionaries,
    containing various information.

    """

    def __init__(
        self, config: ExperimentConfig, data: Q4LDataModule, **kwargs
    ):  # Garbage bin
        self.temp = 1
        self.data = data
        super().__init__(config=config, data=data)

    # Setup
    def head_input_dim(self):
        if self.config.model.name != "MLPTSEncoder":
            return self.config.model.components.temporal.kwargs.hidden_size
        else:
            return self.config.model.components.temporal.kwargs.output_dim

    def build_model_arch(self):
        model_config = self.config.model
        component_config = model_config.components

        self.temporal_model: nn.Module = create_instance(
            component_config.temporal,
            input_size=model_config.input_size,
            try_kwargs={"data": self.data},
        )
        self.head: nn.Module = create_instance(
            component_config.head, input_dim=self.head_input_dim()
        )
        self.loss: nn.Module = create_instance(model_config.loss)

    def configure_optimizers(self) -> tp.Any:
        params = itertools.chain(
            self.temporal_model.parameters(), self.head.parameters()
        )
        optimizer: torch.optim.Optimizer = create_instance(
            self.config.model.optimizer, params=params
        )
        return optimizer

    # Data loading
    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        ret_dict = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                ret_dict[k] = torch.from_numpy(v).to(device)
            else:
                ret_dict[k] = v

        return ret_dict

    # Model pipeline
    def forward(self, batch: tp.Dict, batch_idx: int) -> tp.Dict:
        if torch.any(torch.isnan(batch[INPUT_KEY])):
            print("Nan in input")
        if torch.any(torch.isnan(batch[LABEL_KEY])):
            print("Nan in label")
        embedding = self.temporal_model(batch)
        if self.config.model.name == "MSTR_I":
            embedding[EMBEDDING_KEY] = embedding[EMBEDDING_KEY][0][
                0
            ]  # Ugly trick to make it work
        if torch.any(torch.isnan(embedding[EMBEDDING_KEY])):
            print("Nan in embedding")
            self.nan_appears = True
        pred = self.head(embedding)
        if torch.any(torch.isnan(pred[PREDICTION_KEY])):
            print("Nan in prediction")
            self.nan_appears = True
        return pred

    def on_train_epoch_end(self) -> None:
        if getattr(self, "nan_appears", False):
            self.trainer.should_stop = True

    # Training
    def training_step(self, batch: tp.Dict, batch_idx: int) -> tp.Dict:
        pred = self.forward(batch, batch_idx)
        loss = self.loss(
            pred[PREDICTION_KEY].squeeze(),
            torch.nan_to_num(batch[LABEL_KEY]).squeeze(),
        )
        self.log(name="train_loss", value=loss)
        return loss

    # Helper method
    def concat_preds(self, results):
        datetime_list = []
        ticker_list = []
        pred_list = []
        y_list = []

        for item in results:
            if len(item["label"]) != len(item["pred"]):
                item["label"] = item["label"][0]
            item["pred"] = item["pred"].squeeze()
            item["y"] = item["y"].squeeze()
            for (dt, ticker), pred, y in zip(
                item["label"], item["pred"], item["y"]
            ):
                datetime_list.append(dt)
                ticker_list.append(ticker)
                pred_list.append(pred.cpu().numpy().item())
                y_list.append(y.cpu().numpy().item())

        index = pd.MultiIndex.from_arrays(
            [datetime_list, ticker_list], names=("datetime", "instrument")
        )
        pred_series = pd.Series(pred_list, index=index)
        y_series = pd.Series(y_list, index=index)

        return pred_series, y_series

    def process_batch(self, batch: tp.Dict, batch_idx: int) -> tp.Dict:
        pred = self.forward(batch, batch_idx)
        result = {
            "pred": pred["pred"],
            "label": batch["label"],
            "y": batch["y"],
        }
        return result

    def compute_validation_metrics(self, pred_series, y_series):
        # check if both dataframes have the same index
        assert pred_series.index.equals(y_series.index)

        v_rmse = rmse(pred_series, y_series)
        v_ic = ic(pred_series, y_series)
        v_ret = signal_return(pred_series, y_series)
        v_turnover = signal_turnover(pred_series)
        valid_metric_dict = {
            "valid_rmse": v_rmse,
            "valid_ic": v_ic,
            "valid_ret": v_ret,
            "valid_turnover": v_turnover,
        }
        return valid_metric_dict

    def dump_predict_results(self, pred_series, y_series):
        self.predict_pred_series = pred_series
        self.predict_y_series = y_series

        R.save_objects(
            artifact_path="sig_analysis",
            **{"pred.pkl": pred_series, "label.pkl": y_series},
        )
        current_logpath = R.artifact_uri
        csv_dump_path = os.path.join(current_logpath, "outputs")
        os.makedirs(csv_dump_path, exist_ok=True)
        pred_df = pred_series.unstack()
        y_df = y_series.unstack()
        pred_df.to_csv(csv_dump_path + "/pred.csv")
        y_df.to_csv(csv_dump_path + "/label.csv")

    # Validation
    def on_validation_epoch_start(self) -> None:
        self.validation_preds = []

    def validation_step(self, batch: tp.Dict, batch_idx: int):
        result = self.process_batch(batch, batch_idx)
        self.validation_preds.append(result)
        return result

    def on_validation_epoch_end(self) -> None:
        pred_series, y_series = self.concat_preds(self.validation_preds)
        valid_metric_dict = self.compute_validation_metrics(
            pred_series, y_series
        )
        self.log_dict(valid_metric_dict)
        self.q4l_logger.info(
            f"Validation epoch {self.current_epoch}: {valid_metric_dict}"
        )

    # Inference
    def on_predict_epoch_start(self) -> None:
        self.predict_result_list = []

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        result = self.process_batch(batch, batch_idx)
        self.predict_result_list.append(result)
        return result

    def on_predict_epoch_end(self) -> None:
        pred_series, y_series = self.concat_preds(self.predict_result_list)
        self.dump_predict_results(pred_series, y_series)


class SpatiotemporalModel(QuantModel):
    """Base spatiotemporal forecasting model for quant modelling.

    Basic architecture:
        [Input] -> <Temporal Model> -> [TEmbedding] -> <Prediction Head> -> [Output]
           \                        /               /
                    <Spatial Model> -> [SEmbedding]

    The forward process will be like:
        temporal_embedding = self.temporal_model(input)
        spatial_embedding = self.spatial_model(input, temporal_embedding)
        prediction = self.prediction_head(spatial_embedding, temporal_embedding)
        return prediction

    The model can be rewired to support pre-training.

    """

    def __init__(
        self,
        config: ExperimentConfig,
        data: Q4LDataModule,
        **kwargs,
    ):
        # For type hints
        self.temporal_model: nn.Module = None
        self.spatial_model: nn.Module = None
        self.head: nn.Module = None

        # Model rewiring switches
        self.use_temporal_model = True
        self.use_spatial_model = True
        self.ensemble_st_info = False
        self.st = config.model.basic_info.get("st", False)

        # Initialization
        super().__init__(config=config, data=data)

    # @property
    def head_input_dim(self):
        model_config = self.config.model.components
        head_dim = model_config.spatial.kwargs.node_emb_dim
        if self.st is not True and self.ensemble_st_info:
            head_dim += model_config.temporal.kwargs.hidden_size
        return head_dim

    def build_model_arch(self):
        model_config = self.config.model.components
        try_kwargs = {"data": self.data}
        # Build spatial and temporal models
        self.temporal_model = create_instance(
            model_config.temporal,
            input_size=self.config.model.input_size,
            try_kwargs=try_kwargs,
        )
        self.spatial_model = self._build_spatial_model()
        self.head = create_instance(
            model_config.head,
            input_dim=self.head_input_dim(),
        )

        # Construct (maybe-customized) loss function
        self.loss = create_instance(self.config.model.loss)

    def _build_spatial_model(self):
        raise NotImplementedError("Please implement this abstract method!")

    def configure_optimizers(self) -> tp.Any:
        params_list = [
            self.temporal_model.parameters(),
            self.spatial_model.parameters(),
        ]
        if self.head is not None:
            params_list.append(self.head.parameters())

        params = itertools.chain(*params_list)

        optimizer: torch.optim.Optimizer = create_instance(
            self.config.model.optimizer, params=params
        )
        return optimizer

    # Data loading
    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        ret_dict = {
            "label": batch["label"],
            "x": torch.from_numpy(batch["x"]).to(device),
            "y": torch.from_numpy(batch["y"]).to(device),
        }
        for k, v in batch.items():
            if k not in ret_dict:
                ret_dict[k] = v
        return ret_dict

    # Training
    def training_step(self, batch: tp.Dict, batch_idx: int) -> tp.Dict:
        pred = self.forward(batch, batch_idx)
        if torch.any(torch.isnan(pred[PREDICTION_KEY])):
            print("Nan in prediction")
            self.nan_appears = True
        loss = self.loss(
            pred[PREDICTION_KEY].squeeze(), batch[LABEL_KEY].squeeze()
        )
        return loss

    def on_train_epoch_end(self) -> None:
        if getattr(self, "nan_appears", False):
            self.trainer.should_stop = True

    # Helper method
    def concat_preds(self, results):
        datetime_list = []
        ticker_list = []
        pred_list = []
        y_list = []

        for item in results:
            if len(item["label"]) != len(item["pred"]):
                item["label"] = item["label"][0]
            item["pred"] = item["pred"].squeeze()
            item["y"] = item["y"].squeeze()
            for (dt, ticker), pred, y in zip(
                item["label"], item["pred"], item["y"]
            ):
                datetime_list.append(dt)
                ticker_list.append(ticker)
                pred_list.append(pred.cpu().numpy().item())
                y_list.append(y.cpu().numpy().item())

        index = pd.MultiIndex.from_arrays(
            [datetime_list, ticker_list], names=("datetime", "instrument")
        )
        pred_series = pd.Series(pred_list, index=index)
        y_series = pd.Series(y_list, index=index)

        return pred_series, y_series

    def process_batch(self, batch: tp.Dict, batch_idx: int) -> tp.Dict:
        pred = self.forward(batch, batch_idx)
        result = {
            "pred": pred["pred"],
            "label": batch["label"],
            "y": batch["y"],
        }
        return result

    def compute_validation_metrics(self, pred_series, y_series):
        # check if both dataframes have the same index
        assert pred_series.index.equals(y_series.index)

        v_rmse = rmse(pred_series, y_series)
        v_ic = ic(pred_series, y_series)
        v_ret = signal_return(pred_series, y_series)
        v_turnover = signal_turnover(pred_series)
        valid_metric_dict = {
            "valid_rmse": v_rmse,
            "valid_ic": v_ic,
            "valid_ret": v_ret,
            "valid_turnover": v_turnover,
        }
        return valid_metric_dict

    def dump_predict_results(self, pred_series, y_series):
        self.predict_pred_series = pred_series
        self.predict_y_series = y_series

        R.save_objects(
            artifact_path="sig_analysis",
            **{"pred.pkl": pred_series, "label.pkl": y_series},
        )
        current_logpath = R.artifact_uri
        csv_dump_path = os.path.join(current_logpath, "outputs")
        os.makedirs(csv_dump_path, exist_ok=True)
        pred_df = pred_series.unstack()
        y_df = y_series.unstack()
        pred_df.to_csv(csv_dump_path + "/pred.csv")
        y_df.to_csv(csv_dump_path + "/label.csv")

    # Validation
    def on_validation_epoch_start(self) -> None:
        self.validation_preds = []

    def validation_step(self, batch: tp.Dict, batch_idx: int):
        result = self.process_batch(batch, batch_idx)
        self.validation_preds.append(result)
        return result

    def on_validation_epoch_end(self) -> None:
        pred_series, y_series = self.concat_preds(self.validation_preds)
        valid_metric_dict = self.compute_validation_metrics(
            pred_series, y_series
        )
        self.log_dict(valid_metric_dict)
        self.q4l_logger.info(
            f"Validation epoch {self.current_epoch}: {valid_metric_dict}"
        )

    # Inference
    def on_predict_epoch_start(self) -> None:
        self.predict_result_list = []

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        result = self.process_batch(batch, batch_idx)
        self.predict_result_list.append(result)
        return result

    def on_predict_epoch_end(self) -> None:
        pred_series, y_series = self.concat_preds(self.predict_result_list)
        self.dump_predict_results(pred_series, y_series)

    def forward(self, batch: tp.Dict, batch_idx: int) -> tp.Any:
        if self.st is not True:
            temporal_info = self.get_temporal_info(batch)
            if "y_as_x" in batch:
                temporal_info["y_as_x"] = batch["y_as_x"]
            spatial_info = self.get_spatial_info(batch, temporal_info)
            intermediate = self.fuse_info(temporal_info, spatial_info)
            output = self.head(intermediate)
        else:
            self.temporal_model.keep_seq = True
            temporal_info = self.get_temporal_info(batch)

            # For-loop iterate over the temporal sequence
            spatial_embs = []
            _, seq_len, _ = temporal_info["emb"].shape  # (B, S, D)
            for i in range(seq_len):
                temporal_emb_dict = {EMBEDDING_KEY: temporal_info["emb"][:, i]}
                if "y_as_x" in batch:
                    temporal_emb_dict["y_as_x"] = batch["y_as_x"]
                spatial_info = self.get_spatial_info(temporal_emb_dict)
                spatial_embs.append(spatial_info[EMBEDDING_KEY])
            spatial_info = {EMBEDDING_KEY: torch.stack(spatial_embs, dim=1)}

            new_spatial = self.get_temporal_info(spatial_info)
            new_spatial["emb"] = new_spatial["emb"][:, -1, :]
            output = new_spatial
        return output

    # The following 3 methods are for model rewiring.
    def get_temporal_info(self, batch: tp.Dict) -> tp.Dict:
        return self.temporal_model(batch)

    def get_spatial_info(
        self, batch: tp.Dict, temporal_info: tp.Dict
    ) -> tp.Dict:
        spatial_info = self.spatial_model(batch, temporal_info)
        ret = batch.copy()
        if self.use_spatial_model:
            ret.update(spatial_info)
        return ret

    def fuse_info(
        self, temporal_info: tp.Dict, spatial_info: tp.Dict
    ) -> tp.Dict:
        if self.ensemble_st_info:
            concat_emb = torch.cat(
                [temporal_info["emb"], spatial_info["emb"]], dim=-1
            )
            return {"emb": concat_emb}
        elif self.use_spatial_model:
            return spatial_info
        else:
            return temporal_info
