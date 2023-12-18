# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from typing import Text, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from ....data.dataset import DK_I, DK_L, Q4LDataModule
from ....data.handler import DK_L
from ....qlib.workflow import R
from ...base import NonDLModel
from .utils import get_data


class XGBModel(NonDLModel):
    """XGBModel Model."""

    def __init__(self, **kwargs):
        self._params = {}
        self._params.update(kwargs)
        self.model = None

    def fit(
        self,
        dataset: Q4LDataModule,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=dict(),
        reweighter=None,
        **kwargs,
    ):
        x_train, y_train = get_data(dataset, partition="train", data_key=DK_L)
        x_valid, y_valid = get_data(dataset, partition="valid", data_key=DK_L)
        # Fill nan with 0
        x_train = np.nan_to_num(x_train)
        x_valid = np.nan_to_num(x_valid)
        y_train = np.nan_to_num(y_train)
        y_valid = np.nan_to_num(y_valid)

        if reweighter is None:
            w_train = None
            w_valid = None
        # elif isinstance(reweighter, Reweighter):
        #     w_train = reweighter.reweight(df_train)
        #     w_valid = reweighter.reweight(df_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train)
        dvalid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)
        self.model = xgb.train(
            self._params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=evals_result,
            **kwargs,
        )
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]
        with open(os.path.join(R.artifact_uri, "train_results.json"), "w") as f:
            json.dump(evals_result, f, indent=4)

    def predict(
        self, data: Q4LDataModule, segment: Union[Text, slice] = "test"
    ):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test, y_test, test_index = get_data(
            data, partition="test", data_key=DK_I, return_labels=True
        )
        x_test = np.nan_to_num(x_test)
        y_test = np.nan_to_num(y_test)
        dtest = xgb.DMatrix(x_test)
        return pd.Series(
            self.model.predict(dtest), index=test_index
        ), pd.Series(y_test, index=test_index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """Get feature importance.

        Notes
        -------
            parameters reference:
                https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score

        """
        return pd.Series(self.model.get_score(*args, **kwargs)).sort_values(
            ascending=False
        )
