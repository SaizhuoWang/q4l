import abc
import os
import typing as tp

import numpy as np
import pandas as pd

from ...utils.trade_cal import TradeCalendarFactory
from . import Backend


class StorageBackend(Backend):
    backend_type: str = "storage"

    @property
    def trade_calendar(self):
        return self.calendar

    @abc.abstractmethod
    def get_factor(self, factor_name):
        raise NotImplementedError(
            "StorageBackend.get_factor() is an abstract method."
        )

    @abc.abstractmethod
    def get_ticker_list(self, pool: str) -> tp.List[str]:
        raise NotImplementedError(
            f"Not implemented yet in this class: {self.__class__.__name__}"
        )

    def slice_df_by_time(
        self, factor: pd.DataFrame, start_time: str, end_time: str
    ) -> pd.DataFrame:
        factor.index = pd.to_datetime(factor.index)
        ticks: tp.List[pd.Timestamp] = factor.index.to_list()
        start_idx = (
            0
            if start_time is None
            else ticks.index(
                pd.Timestamp(self.calendar.align(start_time, mode="forward"))
            )
        )
        end_idx = (
            len(ticks) - 1
            if end_time is None
            else ticks.index(
                pd.Timestamp(self.calendar.align(end_time, mode="backward"))
            )
        )
        return factor.iloc[start_idx : end_idx + 1]  # Rows represent time.


class FileSystemBackend(StorageBackend):
    factor_name_map = {}
    mem_cache = {}

    def __init__(
        self,
        root_dir: str,
        frequency: str,
    ) -> None:
        self.root_dir = root_dir
        self.frequency = frequency
        # Now calendar is hard-coded into the backend.
        self.calendar = TradeCalendarFactory.get_calendar(
            cal_type="q4l",
            calendar_path=os.path.join(
                root_dir, "calendars", f"{frequency}.txt"
            ),
        )

    @staticmethod
    def _repeat_across_columns(
        factor: pd.Series, reference_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Repeat a vector factor across columns so that it becomes a dataframe
        The index and columns of the returned dataframe are the same as the
        reference dataframe."""
        # We need to align the datetime between two things. Intersection is used.
        target_index = reference_df.index.intersection(factor.index)
        factor = factor.loc[target_index]

        # Repeat the factor values across columns.
        factor_repeated = np.tile(
            factor.values.reshape(-1, 1), (1, reference_df.shape[1])
        )

        # Create a new DataFrame with repeated values.
        return pd.DataFrame(
            data=factor_repeated,
            index=target_index,
            columns=reference_df.columns,
        )

    def benchmarkindex(self, attr: str, benchmark: str):
        df = pd.read_csv(
            os.path.join(
                self.root_dir, "features", self.frequency, f"{attr}.csv"
            )
        )
        # df = pd.read_csv(os.path.join(self.root_dir, "features", self.frequency, f"{attr}.csv"), index_col=0)
        # df.index = pd.to_datetime(df.index)
        bm = df[benchmark]
        return self._repeat_across_columns(bm, df)

    def get_factor(
        self,
        factor_name: str,  # Name of the factor
        start_time: str = None,  # Start time to slice the data
        end_time: str = None,  # End time to slice the data
        instruments: tp.Optional[
            tp.Union[str, tp.List[str]]
        ] = None,  # Instruments to slice the data
        benchmark: str = "000852_XSHG",  # Benchmark index
    ):
        factor_name_mapped = self.factor_name_map.get(factor_name, factor_name)
        if "benchmarkindex" in factor_name_mapped:
            part = factor_name_mapped[len("benchmarkindex") + 1 :]
            df = self.benchmarkindex(part, benchmark)
        else:
            # Construct the path to the factor file
            factor_path = os.path.join(
                self.root_dir, "features", self.frequency, f"{factor_name}.csv"
            )
            # Check if the factor file exists
            if not os.path.exists(factor_path):
                raise FileNotFoundError(f"Factor {factor_name} does not exist.")

            # Read the factor file as a dataframe, remove unnamed columns, and sort by index
            df = pd.read_csv(factor_path, index_col=0).sort_index()
            df.index = pd.to_datetime(df.index.astype("str"))

        # for col in df.columns:
        #     if "Unnamed: " in col:
        #         df = df.drop(col, axis=1)

        # # Make index a datetime index
        # df.index = pd.to_datetime(df.index)

        # Slice the dataframe by time
        df = self.slice_df_by_time(
            df,
            start_time=start_time,
            end_time=end_time,
        )

        # Slice the dataframe by instruments if provided
        if instruments is not None:
            if isinstance(instruments, str):
                ticker_list = self.get_ticker_list(instruments)
                targets = set.intersection(
                    set(df.columns.to_list()), set(ticker_list)
                )
                df = df.loc[:, list(targets)]
            elif isinstance(instruments, tp.List):
                targets = set.intersection(
                    set(df.columns.to_list()), set(instruments)
                )
                df = df.loc[:, list(targets)]

        # Return the sliced dataframe
        return df

    def get_ticker_list(self, pool: str) -> tp.List[str]:
        pool_filepath = os.path.join(
            self.root_dir, "instruments", f"{pool}.txt"
        )
        if not os.path.exists(pool_filepath):
            raise FileNotFoundError(f"Pool {pool} does not exist.")
        pool_df = pd.read_csv(
            pool_filepath, index_col=0, keep_default_na=False, delimiter="\t"
        )
        ticker_list = sorted(set(pool_df.index.to_list()))
        return ticker_list


class IdeadataBackend(StorageBackend):
    # TODO: Integrate `ideadata` functions into loader
    def get():
        pass


class DatabaseBackend(StorageBackend):
    # TODO: Integrate direct IDEA database access into loader
    def get():
        pass
