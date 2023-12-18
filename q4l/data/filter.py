"""Filter functions for disqualified data samples."""

import typing as tp
from abc import abstractmethod

import numpy as np
import pandas as pd


class Filter:
    @abstractmethod
    def filter(
        self,
        x: np.ndarray,
        x_nan_mask: np.ndarray,
        y: np.ndarray,
        y_nan_mask: np.ndarray,
        ticks: tp.List,
        tickers: tp.List,
    ) -> np.ndarray:
        raise NotImplementedError


class PoolFilter(Filter):
    """Filter stocks based on the stock pool at that time step."""

    def __init__(self, pool_filepath: str) -> None:
        super().__init__()
        self.pool_history = pd.read_csv(
            pool_filepath, index_col=0, header=None, delimiter="\t"
        )

    def filter(
        self,
        x: np.ndarray = None,
        x_nan_mask: np.ndarray = None,
        y: np.ndarray = None,
        y_nan_mask: np.ndarray = None,
        ticks: tp.List = [],
        tickers: tp.List = [],
    ) -> np.ndarray:
        """Filter window samples by stock pool at that time step.

        Parameters
        ----------
        data : np.ndarray
            input data of shape (T, N, W, F)

        Returns
        -------
        np.ndarray
            valid mask of shape (T, N). True means the sample is valid.

        """

        if len(ticks) == 0 or len(tickers) == 0:
            raise ValueError(
                "ticks and tickers must be non-empty. got ticks = {}, tickers = {}".format(
                    ticks, tickers
                )
            )

        # Initialize the mask with False values
        valid_mask = np.zeros((len(ticks), len(tickers)), dtype=bool)

        # Iterate over each ticker and its corresponding date range(s)
        for idx, ticker in enumerate(tickers):
            ticker_data_rows = self.pool_history[
                self.pool_history.index == ticker
            ]
            for _, (start_date, end_date) in ticker_data_rows.iterrows():
                start_date = pd.Timestamp(start_date)
                end_date = pd.Timestamp(end_date)
                valid_dates_mask = np.array(
                    [(start_date <= tick <= end_date) for tick in ticks],
                    dtype=bool,
                )
                valid_mask[:, idx] |= valid_dates_mask

        return valid_mask


class TSNaNFilter(Filter):
    """NaN sample filter for time-series data."""

    def __init__(
        self,
        nan_ratio: float = 0.2,
    ):
        self.nan_ratio = nan_ratio

    def filter(
        self,
        x: np.ndarray,
        x_nan_mask: np.ndarray,
        y: np.ndarray,
        y_nan_mask: np.ndarray,
        ticks: tp.List,
        tickers: tp.List,
    ) -> np.ndarray:
        """Filter window samples by total nan ratio in that window.

        Parameters
        ----------
        data : np.ndarray
            input data of shape (T, N, W, F)

        Returns
        -------
        np.ndarray
            valid mask of shape (T, N). True means the sample is valid.

        """
        # In this function, we calulate the ratio of NaN values in each sample.
        # A True entry in nan mask indicates an invalid sample.
        x_nan = x_nan_mask
        sample_nan_ratio = x_nan.sum(axis=-1).sum(axis=-1) / (
            x_nan.shape[-1] * x_nan.shape[-2]
        )
        x_valid_mask = sample_nan_ratio < self.nan_ratio
        y_valid_mask = ~np.squeeze(y_nan_mask)
        return np.logical_and(x_valid_mask, y_valid_mask)
