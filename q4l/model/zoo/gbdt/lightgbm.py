# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from typing import Text, Union

import lightgbm as lgb
import pandas as pd

from ....data.dataset import DK_I, Q4LDataModule
from ....data.handler import DK_L
from ....qlib.workflow import R
from .utils import get_data


class LGBModel:
    """LightGBM Model."""

    def __init__(
        self,
        loss="mse",
        early_stopping_rounds=50,
        num_boost_round=1000,
        **kwargs,
    ):
        if loss not in {"mse", "binary"}:
            raise NotImplementedError
        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None

    def fit(
        self,
        data: Q4LDataModule,
        num_boost_round=None,
        early_stopping_rounds=None,
        verbose_eval=20,
        evals_result=None,
        reweighter=None,
        **kwargs,
    ):
        if evals_result is None:
            evals_result = {}  # in case of unsafety of Python default values
        x_train, y_train = get_data(data, partition="train", data_key=DK_L)
        x_valid, y_valid = get_data(data, partition="valid", data_key=DK_L)
        # x_train, y_train, x_valid, y_valid = get_data(data)

        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_valid, label=y_valid)
        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds
            if early_stopping_rounds is None
            else early_stopping_rounds
        )
        # NOTE: if you encounter error here. Please upgrade your lightgbm
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(evals_result)
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=self.num_boost_round
            if num_boost_round is None
            else num_boost_round,
            valid_sets=[valid_data],
            callbacks=[
                early_stopping_callback,
                verbose_eval_callback,
                evals_result_callback,
            ],
            **kwargs,
        )
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
        return pd.Series(
            self.model.predict(x_test), index=test_index
        ), pd.Series(y_test, index=test_index)
