#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Interfaces to interpret models."""

from abc import abstractmethod

import pandas as pd


class FeatureInt:
    """Feature (Int)erpreter."""

    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance.

        Returns
        -------
            The index is the feature name.

            The greater the value, the higher importance.

        """


class LightGBMFInt(FeatureInt):
    """LightGBM (F)eature (Int)erpreter."""

    def __init__(self):
        self.model = None

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """Get feature importance.

        Notes
        -----
            parameters reference:
            https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html?highlight=feature_importance#lightgbm.Booster.feature_importance

        """
        return pd.Series(
            self.model.feature_importance(*args, **kwargs),
            index=self.model.feature_name(),
        ).sort_values(  # pylint: disable=E1101
            ascending=False
        )
