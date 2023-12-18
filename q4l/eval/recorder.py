import json
import os
from typing import List, Optional

import pandas as pd

from ..qlib.workflow import R
from ..qlib.workflow.record_temp import ACRecordTemp
from ..utils.log import get_logger
from .metrics.ir import f1_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k
from .metrics.regression import mae, mape, mda, mse, r2_score, rmse


class PredAnaRecorder(ACRecordTemp):
    artifact_path = "sig_analysis"

    def __init__(self, recorder=None, skip_existing=False):
        """The recorder for the prediction analysis. It will compute metrics
        related to prediction accuracy, including:

        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Mean Absolute Percentage Error (MAPE)
        - Mean Directional Accuracy (MDA)
        - R-squared Score

        """
        super().__init__(recorder)
        self.logger = get_logger(self)

    def _generate(self, label: Optional[pd.Series] = None, **kwargs):
        pred = self.load("pred.pkl")
        if label is None:
            label = self.load("label.pkl")
        if label is None or label.empty:
            self.logger.warning(f"Empty label.")
            return

        # Compute these metrics using q4l util functions
        result = {
            "mae": mae(pred, label),
            "mse": mse(pred, label),
            "rmse": rmse(pred, label),
            "mape": mape(pred, label),
            "mda": mda(pred, label),
            "r2_score": r2_score(pred, label),
        }

        with open(
            os.path.join(
                R.artifact_uri, self.artifact_path, "pred_analysis_metric.json"
            ),
            "w",
        ) as f:
            json.dump(result, f)


class RankAnaRecorder(ACRecordTemp):
    artifact_path = "sig_analysis"

    def __init__(
        self,
        recorder=None,
        skip_existing=False,
        k_list: List[int] = [10, 20, 30, 50],
    ):
        """The recorder for the ranking analysis. It will compute metrics
        related to ranking accuracy, including:

        - Precision@K
        - Recall@K
        - F1@K
        - Normalized Discounted Cumulative Gain (NDCG)
        - Mean Reciprocal Rank (MRR)

        """
        super().__init__(recorder)
        self.logger = get_logger(self)
        self.k_list = k_list

    def _generate(self, label: Optional[pd.Series] = None, **kwargs):
        pred = self.load("pred.pkl")
        if label is None:
            label = self.load("label.pkl")
        if label is None or label.empty:
            self.logger.warning(f"Empty label.")
            return

        result = {"mrr": mrr(pred, label)}
        for k in self.k_list:
            result.update(
                {
                    f"prec@{k}": precision_at_k(pred, label, k),
                    f"recall@{k}": recall_at_k(pred, label, k),
                    f"f1@{k}": f1_at_k(pred, label, k),
                    f"ndcg@{k}": ndcg_at_k(pred, label, k),
                }
            )

        with open(
            os.path.join(
                R.artifact_uri, self.artifact_path, "rank_analysis_metric.json"
            ),
            "w",
        ) as f:
            json.dump(result, f)
