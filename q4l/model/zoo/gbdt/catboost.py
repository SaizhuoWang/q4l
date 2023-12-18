# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Text, Union

import pandas as pd
from catboost import CatBoost, Pool
from catboost.utils import get_gpu_device_count

from ....data.dataset import DK_I, DK_L, Q4LDataModule
from ...base import NonDLModel
from .utils import get_data


class CatBoostModel(NonDLModel):
    """CatBoost Model."""

    def __init__(self, loss="RMSE", **kwargs):
        # There are more options
        if loss not in {"RMSE", "Logloss"}:
            raise NotImplementedError
        self._params = {"loss_function": loss}
        self._params.update(kwargs)
        self.model = None

    def fit(
        self,
        data: Q4LDataModule,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=dict(),
        reweighter=None,
        **kwargs,
    ):
        x_train, y_train = get_data(data, partition="train", data_key=DK_L)
        x_valid, y_valid = get_data(data, partition="valid", data_key=DK_L)

        # x_train, x_valid, y_train, y_valid = get_data(data)

        # for valid_ticks in train_sampler.valid_indices:
        #     if valid_ticks not in train_sampler.ticks:
        #         raise ValueError(f"Valid ticks {valid_ticks} not in train ticks {train_sampler.ticks}")

        # train_df = train_sampler.df

        # if train_df.empty or valid_df.empty:
        #     raise ValueError("Empty data from dataset, please check your dataset config.")
        # feature_group = data.cfg.data.sampler.x_group
        # label_group = data.cfg.data.sampler.y_group
        # x_train, y_train = train_df[feature_group], train_df[label_group]
        # x_valid, y_valid = valid_df[feature_group], valid_df[label_group]

        # # CatBoost needs 1D array as its label
        # if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
        #     y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        # else:
        #     raise ValueError("CatBoost doesn't support multi-label training")

        if reweighter is None:
            w_train = None
            w_valid = None
        # elif isinstance(reweighter, Reweighter):
        #     w_train = reweighter.reweight(df_train).values
        #     w_valid = reweighter.reweight(df_valid).values
        else:
            raise ValueError("Unsupported reweighter type.")

        train_pool = Pool(data=x_train, label=y_train, weight=w_train)
        valid_pool = Pool(data=x_valid, label=y_valid, weight=w_valid)

        # Initialize the catboost model
        self._params["iterations"] = num_boost_round
        self._params["early_stopping_rounds"] = early_stopping_rounds
        self._params["verbose_eval"] = verbose_eval
        self._params["task_type"] = (
            "GPU" if get_gpu_device_count() > 0 else "CPU"
        )
        self.model = CatBoost(self._params, **kwargs)

        # train the model
        self.model.fit(
            train_pool, eval_set=valid_pool, use_best_model=True, **kwargs
        )

        evals_result = self.model.get_evals_result()
        evals_result["train"] = list(evals_result["learn"].values())[0]
        evals_result["valid"] = list(evals_result["validation"].values())[0]

    def predict(
        self, data: Q4LDataModule, segment: Union[Text, slice] = "test"
    ):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test, y_test, test_index = get_data(
            data, partition="test", data_key=DK_I, return_labels=True
        )
        test_pool = Pool(data=x_test, label=y_test)
        return pd.Series(
            self.model.predict(test_pool), index=test_index
        ), pd.Series(y_test, index=test_index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """Get feature importance.

        Notes
        -----
            parameters references:
            https://catboost.ai/docs/concepts/python-reference_catboost_get_feature_importance.html#python-reference_catboost_get_feature_importance

        """
        return pd.Series(
            data=self.model.get_feature_importance(*args, **kwargs),
            index=self.model.feature_names_,
        ).sort_values(ascending=False)


if __name__ == "__main__":
    cat = CatBoostModel()
