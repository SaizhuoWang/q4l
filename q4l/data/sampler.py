import typing as tp
from abc import abstractmethod
from typing import List

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as window_trick

from ..config import ModuleConfig, SamplerConfig
from ..qlib.log import TimeInspector
from ..utils.log import get_logger
from ..utils.misc import create_instance
from .filter import Filter


def reshape_data(data: pd.DataFrame, window_size: int) -> np.ndarray:
    arr = data.values  # (T*N, F)
    num_ticks = len(data.index.unique(0))
    num_tickers = len(data.index.unique(1))
    arr.shape = (num_ticks, num_tickers, -1)  # without copy
    arr_window = window_trick(arr, window_size, axis=0)
    arr = np.transpose(arr_window, (0, 1, 3, 2))  # (T, N, W, F)
    return arr


class BaseSampler:
    """The base abstract class that defines interfaces for samplers."""

    def __init__(self, data: pd.DataFrame, filters: List[ModuleConfig] = None):
        pass

    @abstractmethod
    def _make_valid_indices(self):
        """Make index for quick sampling."""

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> tp.Dict:
        """Given an int index, return a sample dict."""

    @abstractmethod
    def collate(self, batch: tp.List[tp.Dict]) -> tp.Dict:
        """Given a list of sample dicts, return a batch dict."""


class TCDataSampler(BaseSampler):
    """The base sampler that supports time-series and cross-sectional
    sampling."""

    def __init__(
        self,
        data: pd.DataFrame,
        nan_mask: pd.DataFrame,
        config: SamplerConfig,
        is_inference: bool = False,  # If for inference, we only need to sample X
    ):
        """The input dataframe is of shape (T*N, F), where T is the number of
        ticks, N is the number of tickers, and F is the number of features.

        A cross-section is a slice of the dataframe with a fixed datetime, and a
        time-series is a time window for a specific ticker at a specific tick.
        The sampler should first filter out those invalid samples, and then
        sample from the valid ones. We may use the window trick of numpy to make
        the sampling process easier. Before applying window trick, the
        underlying numpy array need to be reshaped. Please reshape it without
        making new copies.

        """
        # Step 1: Do some initializations
        self.logger = get_logger(self)
        self.data = data[config.y_group + config.x_group + config.keep_group]
        self.nan_mask = nan_mask
        self.config = config
        self.is_inferece = is_inference

        self.filters: tp.Dict[str, Filter] = {
            name: create_instance(config)
            for name, config in config.filters.items()
        }

        self.x_window_size = config.x_window
        self.y_window_size = config.y_window

        self.sampling_mode = config.sample_mode
        if not self.is_inferece:
            self.num_ticks = (
                len(data.index.unique(0))
                - (self.x_window_size + self.y_window_size)
                + 1
            )
            self.ticks = data.index.unique(0)[
                self.x_window_size - 1 : -self.y_window_size
            ]
        else:
            self.num_ticks = len(data.index.unique(0)) - self.x_window_size + 1
            self.ticks = data.index.unique(0)[self.x_window_size - 1 :]

        self.tickers = data.index.unique(1)
        self.num_tickers = len(data.index.unique(1))
        # self.logger.info(
        #     f"Memory profile after counting ticks: {display_memory_tree(ProcessNode(psutil.Process()))}"
        # )

        # Reshape data into (T, N, W, F)
        with TimeInspector.logt(name="Reshape data", show_start=True):
            self.x = reshape_data(data[config.x_group], self.x_window_size)
            self.x_nan_mask = reshape_data(
                nan_mask[config.x_group], self.x_window_size
            )
            self.y = reshape_data(data[config.y_group], self.y_window_size)
            self.y_nan_mask = reshape_data(
                nan_mask[config.y_group], self.y_window_size
            )

            if config.sample_y_as_x:
                self.y_as_x = reshape_data(
                    data[config.y_group], self.x_window_size
                )
                self.y_as_x_nan_mask = reshape_data(
                    nan_mask[config.y_group], self.x_window_size
                )
        # self.logger.info(
        #     f"Memory profile after reshaping data: {display_memory_tree(ProcessNode(psutil.Process()))}"
        # )

        # Need alignment before making valid mask: head of X and tail of Y
        self.y = self.y[self.x_window_size :]
        self.y_nan_mask = self.y_nan_mask[self.x_window_size :]
        if not self.is_inferece:
            self.x = self.x[: -self.y_window_size]
            self.x_nan_mask = self.x_nan_mask[: -self.y_window_size]
        else:
            fake_y_tails = np.ones(
                shape=(self.y_window_size, *self.y.shape[1:])
            )
            self.y = np.concatenate([self.y, fake_y_tails], axis=0)
            fake_y_nan_mask_tails = np.zeros(
                shape=(self.y_window_size, *self.y_nan_mask.shape[1:]),
                dtype=self.y_nan_mask.dtype,
            )
            self.y_nan_mask = np.concatenate(
                [self.y_nan_mask, fake_y_nan_mask_tails],
                axis=0,
            )
        # self.logger.info(
        #     f"Memory profile after alignment: {display_memory_tree(ProcessNode(psutil.Process()))}"
        # )
        # Make data valid masks
        self.logger.info(f"Making valid mask for {self.sampling_mode} sampling")
        with TimeInspector.logt(name="Make valid mask", show_start=True):
            total_mask = np.ones((self.num_ticks, self.num_tickers), dtype=bool)
            for name, filter in self.filters.items():
                # For a valid mask, `True` means the sample is valid.
                self.logger.info(f"Applying filter {name}")
                filter_mask = filter.filter(
                    self.x,
                    self.x_nan_mask,
                    self.y,
                    self.y_nan_mask,
                    self.ticks,
                    self.tickers,
                )
                total_mask = np.logical_and(total_mask, filter_mask)
            self.valid_mask = total_mask
        # self.logger.info(
        #     f"Memory profile after valid mask: {display_memory_tree(ProcessNode(psutil.Process()))}"
        # )
        total_num_samples = self.valid_mask.size
        num_valid_samples = np.sum(self.valid_mask)
        valid_rate = num_valid_samples / total_num_samples
        self.logger.info(f"Valid mask shape: {self.valid_mask.shape}.")
        self.logger.info(
            f"Valid rate: {num_valid_samples}/{total_num_samples} = {valid_rate}"
        )

        # Based on valid mask and sampling mode, make valid indices
        with TimeInspector.logt(name="Make valid indices", show_start=True):
            self.valid_labels, self.valid_indices = self._make_valid_indices()

    def get_all_valid_data(
        self,
    ) -> tp.Tuple[np.ndarray, np.ndarray, pd.MultiIndex]:
        """Get all valid data as numpy arrays."""
        if self.sampling_mode == "T":
            x_valid = self.x[self.valid_indices]
            y_valid = self.y[self.valid_indices]
            valid_labels = self.valid_labels
            valid_series = pd.MultiIndex(valid_labels)
        elif self.sampling_mode == "C":
            x_list = []
            y_list = []
            for i, valid_index in enumerate(self.valid_indices):
                x_list.append(self.x[i][valid_index])
                y_list.append(self.y[i][valid_index])
            x_valid = np.concatenate(x_list, axis=0)
            y_valid = np.concatenate(y_list, axis=0)
            valid_labels = np.concatenate(self.valid_labels_np, axis=0).tolist()
            valid_series = pd.MultiIndex.from_tuples(
                valid_labels, names=["datetime", "instrument"]
            )
        return x_valid, y_valid, valid_series

    def _make_valid_indices(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        """A `valid index` contains two fields: `label` and `np_index`. Label is
        the timestamp and ticker of the sample, and `np_index` is the sample's
        index in the big numpy array. For time-series sampling, each sample
        corresponds to such an index. For cross-sectional sampling, each
        sample's corresponding index has its field values being lists that
        contain all the valid samples for that cross-section.

        Returns
        -------
        tp.Dict
            Valid label and indices organized in numpy array.

        Raises
        ------
        ValueError
            if the sampling mode is invalid.

        """
        if self.sampling_mode == "T":
            return self._get_time_series_indices()
        elif self.sampling_mode == "C":
            return self._get_cross_sectional_indices()
        else:
            raise ValueError(f"Invalid sampling mode: {self.sampling_mode}")

    def _get_time_series_indices(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        self.data: (T*N, F)
        """
        labels = self.data.index.to_numpy().reshape(-1, self.num_tickers)[
            self.x_window_size - 1 : -self.y_window_size
        ]
        valid_labels = labels[np.nonzero(self.valid_mask)]
        valid_indices = np.argwhere(self.valid_mask)
        return valid_labels, valid_indices

    # def _get_cross_sectional_indices(self) -> tp.Tuple[np.ndarray, np.ndarray]:
    #     valid_columns = [np.argwhere(x) for x in self.valid_mask]
    #     offset = (self.x_window_size - 1) * self.num_tickers
    #     labels = (
    #         self.data.index[offset:].to_numpy().reshape(-1, self.num_tickers)
    #     )
    #     valid_labels = [
    #         l[np.nonzero(x)] for l, x in zip(labels, self.valid_mask)
    #     ]

    #     self.valid_labels_np = valid_labels
    #     self.valid_indices_np = valid_columns

    #     return valid_labels, valid_columns

    def _get_cross_sectional_indices(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        # Extract indices where valid_mask is True
        valid_columns = [np.argwhere(x).flatten() for x in self.valid_mask]

        # Filter out rows where valid_columns is empty
        valid_rows = [i for i, x in enumerate(valid_columns) if x.size > 0]

        # Adjust valid_columns and labels accordingly
        valid_columns = [valid_columns[i] for i in valid_rows]
        offset = (self.x_window_size - 1) * self.num_tickers
        labels = (
            self.data.index[offset:].to_numpy().reshape(-1, self.num_tickers)
        )
        valid_labels = [
            labels[i][np.nonzero(self.valid_mask[i])] for i in valid_rows
        ]

        self.valid_rows = valid_rows

        self.valid_labels_np = valid_labels
        self.valid_indices_np = valid_columns

        return valid_labels, valid_columns

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise TypeError(f"Index must be int, got {type(index)}: {index}")
        label = self.valid_labels[index]
        if self.sampling_mode == "T":
            sample_idx = self.valid_indices[index]
            sample_idx = sample_idx[0], sample_idx[1]
            x = self.x[sample_idx]
            y = self.y[sample_idx]
            if self.config.sample_y_as_x:
                y_as_x = self.y_as_x[sample_idx]
        else:
            actual_index = self.valid_rows[index]
            cs_valid_indices = self.valid_indices[index]
            x = self.x[actual_index, cs_valid_indices.squeeze()]
            y = self.y[actual_index, cs_valid_indices.squeeze()]
            if self.config.sample_y_as_x:
                y_as_x = self.y_as_x[actual_index, cs_valid_indices.squeeze()]

        ret = {"label": label, "x": x, "y": y}
        if self.config.sample_y_as_x:
            ret["y_as_x"] = y_as_x
        return ret

    def __getitems__(self, indices: tp.List[int]) -> tp.Dict:
        if self.sampling_mode == "T":
            labels = self.valid_labels[indices]
            data_indices = self.valid_indices[indices]
            array_indices = data_indices.T[0], data_indices.T[1]
            x = self.x[array_indices]
            y = self.y[array_indices]
            return {
                "label": labels.tolist(),
                "x": np.ascontiguousarray(x),
                "y": np.ascontiguousarray(y),
            }
        else:
            return [
                self.__getitem__(i) for i in indices
            ]  # Uglier but easier to implement

    def collate(self, batch: tp.Union[tp.List[tp.Dict], tp.Dict]) -> tp.Dict:
        # NOTE: Potential caveat here! Be careful!
        if isinstance(batch, dict) and hasattr(self, "__getitems__"):
            return batch  # No need to collate if we are using __getitems__

        if self.sampling_mode == "T":
            label_list = [item["label"] for item in batch]
            x_concat = np.stack([item["x"] for item in batch], axis=0)
            y_concat = np.stack([item["y"] for item in batch], axis=0)
            ret = {
                "label": label_list,
                "x": np.ascontiguousarray(x_concat),
                "y": np.ascontiguousarray(y_concat),
            }
            if self.config.sample_y_as_x:
                y_as_x_concat = np.stack(
                    [item["y_as_x"] for item in batch], axis=0
                )
                ret["y_as_x"] = np.ascontiguousarray(y_as_x_concat)
            # return ret
        else:
            # Do padding for x and y of shape (B, N, W, F), where N is max number of tickers
            # in the match. Also return padding mask of shape (B, N)
            label_list = [item["label"] for item in batch]

            def pad_arrays_with_mask(arr_list):
                B = len(arr_list)
                N = max(
                    [arr.shape[0] for arr in arr_list]
                )  # Find the maximum n_i value
                other_dims = arr_list[0].shape[
                    1:
                ]  # Get the remaining dimensions

                # Create a new array with the desired shape, initialized with zeros (or any padding value)
                padded_array = np.zeros(
                    (B, N) + other_dims, dtype=arr_list[0].dtype
                )

                # Create a padding mask initialized with zeros
                padding_mask = np.zeros((B, N), dtype=np.int8)

                # Copy the data from each original array into the new array and update the padding mask
                for i, arr in enumerate(arr_list):
                    padded_array[i, : arr.shape[0], ...] = arr
                    padding_mask[i, : arr.shape[0]] = 1
                return padded_array, padding_mask

            x_batch, x_mask = pad_arrays_with_mask(
                [item["x"] for item in batch]
            )
            y_batch, _ = pad_arrays_with_mask([item["y"] for item in batch])

            ret = {
                "label": label_list,
                "x": np.ascontiguousarray(x_batch),
                "y": np.ascontiguousarray(y_batch),
                "padding_mask": np.ascontiguousarray(x_mask),
            }

            if self.config.sample_y_as_x:
                y_as_x_batch, _ = pad_arrays_with_mask(
                    [item["y_as_x"] for item in batch]
                )
                ret["y_as_x"] = np.ascontiguousarray(y_as_x_batch)

        return ret
