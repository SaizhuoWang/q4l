import abc
import typing as tp

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import mstats

from ..qlib.data.dataset.processor import (
    FilterCol,
    Processor,
    get_group_columns,
)


class ColumnNaNFilter(Processor):
    def __init__(self, nan_threshold: float, fields_group="feature"):
        self.nan_threshold = nan_threshold
        self.fields_group = fields_group

    def __call__(self, df):
        """Filter the columns with NaN ratio greater than the threshold."""
        # get_group_columns(df, self.fields_group)
        sub_df = df[self.fields_group]

        columns_to_drop = []
        for col in sub_df.columns:
            if sub_df[col].isnull().sum() / len(sub_df) > self.nan_threshold:
                columns_to_drop.append(col)
        columns_to_keep = list(set(sub_df.columns) - set(columns_to_drop))

        return FilterCol(
            fields_group=self.fields_group, col_list=columns_to_keep
        )(df)


class Columnwise2DProcessor(Processor):
    def __init__(self, fields_group="feature"):
        super().__init__()
        self.fields_group = fields_group

    @abc.abstractmethod
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def __call__(self, df: pd.DataFrame):
        columns = get_group_columns(df, self.fields_group)
        col_multiindex = pd.MultiIndex.from_tuples(columns)

        df_unstacked = df.unstack("instrument")
        df_processed = df.copy()

        processed_factor_dfs = Parallel(n_jobs=100, backend="multiprocessing")(
            delayed(self._process)(df_unstacked[col]) for col in columns
        )

        # with TimeInspector.logt(name="flatten_array", show_start=True):
        numpy_array_list = []
        for adf in processed_factor_dfs:
            numpy_array_list.append(adf.values.flatten())
        big_np_array = np.stack(numpy_array_list, axis=1)
        new_df = pd.DataFrame(
            big_np_array, index=df.index, columns=col_multiindex
        )

        # df_processed = pd.DataFrame(big_np_array, index=df.index, columns=df.columns)
        # return df_processed
        # with TimeInspector.logt(name="backfill_df"):
        #     for col, col_array in zip(columns, numpy_array_list):
        #         df_processed[col] = col_array

        # with TimeInspector.logt(name="backfill_df"):
        df_processed.update(new_df)

        return df_processed


class Winsorizer(Columnwise2DProcessor):
    def __init__(self, lower_limit=0.01, upper_limit=0.01, axis=1, **kwargs):
        """Initialize Winsorizer with lower and upper limits.

        Parameters
        ----------
        lower_limit : float
            Lower limit for winsorization. Defaults to 0.01.
        upper_limit : float
            Upper limit for winsorization. Defaults to 0.01.

        """
        super().__init__(**kwargs)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.axis = axis

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Winsorize the input dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            Winsorized dataframe

        """
        # with TimeInspector.logt(name="winsorize"):
        df_np = df.values.copy()
        df_np = mstats.winsorize(
            df_np,
            limits=(self.lower_limit, self.upper_limit),
            axis=self.axis,
            nan_policy="omit",
        )
        return pd.DataFrame(df_np, index=df.index, columns=df.columns)


class DataClipper(Columnwise2DProcessor):
    def __init__(self, clipper_type="MAD", multiplier=1.5, axis=1, **kwargs):
        """Initialize DataClipper with clipper type, multiplier and axis.

        Parameters
        ----------
        clipper_type : str
            Type of clipper to use. Either 'MAD' or 'STD'. Defaults to 'MAD'.

        multiplier : float
            Multiplier for clipper. Defaults to 1.5.

        axis : int
            Axis to calculate clipper along. Defaults to 1.

        """
        super().__init__(**kwargs)
        if clipper_type not in ["MAD", "STD"]:
            raise ValueError("clipper_type must be either 'MAD' or 'STD'")

        self.clipper_type = clipper_type
        self.multiplier = multiplier
        self.axis = axis

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        df_np = df.values.copy()

        if self.clipper_type == "MAD":
            pivot = np.nanmedian(df_np, axis=self.axis, keepdims=True)
            distance = np.nanmedian(
                np.abs(df_np - pivot), axis=self.axis, keepdims=True
            )
        elif self.clipper_type == "STD":
            pivot = np.nanmean(df_np, axis=self.axis, keepdims=True)
            distance = np.nanstd(df_np, axis=self.axis, keepdims=True)

        lower_limit = pivot - self.multiplier * distance
        upper_limit = pivot + self.multiplier * distance
        df_np = np.clip(df_np, lower_limit, upper_limit)

        return pd.DataFrame(df_np, index=df.index, columns=df.columns)


class Imputer(Columnwise2DProcessor):
    def __init__(self, method="mean", axis=0, **kwargs):
        """Initialize Imputer with the imputation method and axis.

        Parameters
        ----------
        method : str
            Imputation method. Options are 'mean', 'median', 'zero', etc. Defaults to 'mean'.
        axis : int
            The axis along which to compute the imputation statistics. Defaults to 0.

        """
        super().__init__(**kwargs)
        self.method = method
        self.axis = axis

    def _process(self, df: pd.DataFrame):
        """Impute the missing values in the data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        """
        # with TimeInspector.logt(name="impute"):
        if self.method == "mean":
            return df.fillna(df.mean(axis=self.axis))
        elif self.method == "median":
            return df.fillna(df.median(axis=self.axis))
        elif self.method == "zero":
            return df.fillna(0)
        else:
            raise ValueError(f"Unsupported imputation method: {self.method}")


class Array3DProcessor(Processor):
    def __init__(
        self, fields_group: tp.List[str] = "feature", device: str = "cpu"
    ):
        super().__init__()
        self.fields_group = fields_group
        self.device = device

    @abc.abstractmethod
    def _process(self, arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, df: pd.DataFrame):
        for grp in self.fields_group:
            sub_df = df[grp]
            num_ticks = len(sub_df.index.unique(0))
            num_tickers = len(sub_df.index.unique(1))
            big_np_narray = sub_df.values.copy()
            big_np_narray.shape = (num_ticks, num_tickers, -1)  # (T, N, F)
            new_array = self._process(big_np_narray)
            if "cuda" in self.device:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            new_array.shape = (num_ticks * num_tickers, -1)
            df[grp] = new_array
        return df


class Winsorizer3D(Array3DProcessor):
    def __init__(self, lower_limit=0.01, upper_limit=0.01, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.axis = axis

    def _process(self, arr: np.ndarray) -> np.ndarray:
        # with TimeInspector.logt(name="winsorize"):
        arr_tensor = torch.from_numpy(arr).to(self.device)
        left = torch.nanquantile(
            input=arr_tensor,
            q=self.lower_limit,
            dim=self.axis,
            keepdim=True,
        )
        right = torch.nanquantile(
            input=arr_tensor,
            q=1 - self.upper_limit,
            dim=self.axis,
            keepdim=True,
        )
        result = torch.clip(arr_tensor, left, right)
        result_np = result.cpu().numpy()
        return result_np


class Imputer3D(Array3DProcessor):
    def __init__(self, method="mean", axis=0, **kwargs):
        """Initialize Imputer with the imputation method and axis.

        Parameters
        ----------
        method : str
            Imputation method. Options are 'mean', 'median', 'zero', etc. Defaults to 'mean'.
        axis : int
            The axis along which to compute the imputation statistics. Defaults to 0.

        """
        super().__init__(**kwargs)
        self.method = method
        self.axis = axis

    def _process(self, arr: np.ndarray) -> np.ndarray:
        # Compute the imputation statistics and then use mask to impute
        # with TimeInspector.logt("Impute3D"):
        arr_tensor = torch.from_numpy(arr).to(self.device)
        nan_mask = torch.isnan(arr_tensor)
        inf_mask = torch.isinf(arr_tensor)
        fill_mask = torch.logical_or(nan_mask, inf_mask)
        if self.method == "mean":
            fill_value = torch.nanmean(
                input=arr_tensor, dim=self.axis, keepdim=True
            )
            # fill_value = np.nanmean(arr, axis=self.axis, keepdims=True)
        elif self.method == "median":
            fill_value = torch.nanmedian(
                input=arr_tensor, dim=self.axis, keepdim=True
            )
            # fill_value = np.nanmedian(arr, axis=self.axis, keepdims=True)
        elif self.method == "zero":
            fill_value = 0
        else:
            raise ValueError(f"Unsupported imputation method: {self.method}")
        a = torch.where(fill_mask, fill_value, arr_tensor)
        a_np = a.cpu().numpy()
        return a_np


class ZScore3D(Array3DProcessor):
    def __init__(
        self,
        fit_start_time,
        fit_end_time,
        fields_group=None,
        axis: int = 0,
        robust: bool = False,
        clip_outlier: bool = False,
        outlier_threshold: tp.Optional[float] = 3.0,
        **kwargs,
    ):
        super().__init__(fields_group=fields_group, **kwargs)
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.axis = axis  # If axis=-1, then compute statistics on the whole factor values
        self.robust = robust  # If robust, use MAD instead of std
        self.clip_outlier = (
            clip_outlier  # If clip_outlier, then clip the outliers
        )
        self.outlier_threshold = outlier_threshold

    def _process(self, arr: np.ndarray) -> np.ndarray:
        axis = self.axis if self.axis != -1 else (0, 1)

        def nanstd(x, axis=None, keepdims=False, unbiased=True):
            mean = torch.nanmean(x, dim=axis, keepdim=True)
            x_diff = x - mean
            squared_diff = x_diff**2
            var = torch.nanmean(squared_diff, dim=axis, keepdim=keepdims)

            if unbiased:
                n_valid_elements = torch.sum(
                    ~torch.isnan(x), dim=axis, keepdim=keepdims
                )
                divisor = n_valid_elements - 1
                var *= n_valid_elements.float() / divisor.float()

            std = torch.sqrt(var)
            return std

        def nanmad(x, axis=None, keepdims=False):
            median, median_idx = torch.nanmedian(x, dim=axis, keepdim=True)
            x_diff = torch.abs(x - median)
            mad, mad_index = torch.nanmedian(x_diff, dim=axis, keepdim=keepdims)
            return mad

        # with TimeInspector.logt("ZScore3D"):
        arr_tensor = torch.from_numpy(arr).to(self.device)
        if not self.robust:
            mean = torch.nanmean(input=arr_tensor, dim=axis, keepdim=True)
            std = nanstd(x=arr_tensor, axis=axis, keepdims=True)
        else:
            mean, _ = torch.nanmedian(input=arr_tensor, dim=axis, keepdim=True)
            std = nanmad(x=arr_tensor, axis=axis, keepdims=True) * 1.4826
        result = (arr_tensor - mean) / (std + 1e-8)
        if self.clip_outlier:
            result = torch.clip(
                result,
                -self.outlier_threshold,
                self.outlier_threshold,
            )

        result_np = result.cpu().numpy()
        return result_np
