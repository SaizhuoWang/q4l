"""Load data from external storage (e.g. database, file system, etc.) into
memory."""
import re
import traceback
import typing as tp
import warnings
from datetime import datetime
from typing import List, Union

import loky
import numpy as np
import omegaconf
import pandas as pd
from omegaconf import OmegaConf

from ..config import LoaderConfig
from ..constants import FACTOR_TOKEN_RE
from ..utils.log import get_logger
from ..utils.trade_cal import TradeCalendar
from .backend import BackendFactory
from .backend.compute import ComputeBackend
from .backend.storage import StorageBackend


class Q4LDataLoader:
    def __init__(self, config: LoaderConfig) -> None:
        self.config: LoaderConfig = config
        self.storage_backends: tp.Dict[str, StorageBackend] = {
            k: BackendFactory.create(v)
            for k, v in config.backend.storage.items()
        }
        self.compute_backends: tp.Dict[str, ComputeBackend] = {
            k: BackendFactory.create(v, return_class_only=True)
            for k, v in config.backend.compute.items()
        }
        self.logger = get_logger(self)

    @property
    def trade_calendar(self) -> TradeCalendar:
        """Get trade calendar from one of your storage backend."""
        return list(self.storage_backends.values())[0].trade_calendar

    def load(
        self,
        instruments: Union[str, List[str]] = None,
        start_time: Union[str, datetime] = None,
        end_time: Union[str, datetime] = None,
    ) -> pd.DataFrame:
        # Step 1: Create an empty dictionary to store the loaded data
        result_df = {}

        # Step 2: Iterate over each alpha group defined in the configuration
        for group_name, alpha_group in self.config.alpha.items():
            self.logger.info(f"Loading group {group_name}")
            # Step 3: Extract the name and expressions for the current alpha group
            if group_name in result_df:
                warnings.warn(
                    f"Duplicate alpha group name: {group_name}. The previous one will be overwritten."
                )

            expr_dict = alpha_group.expressions
            if isinstance(alpha_group.expressions, omegaconf.Container):
                expr_dict = OmegaConf.to_container(expr_dict)

            expressions = list(expr_dict.values())
            names = list(expr_dict.keys())

            compute_backend = alpha_group.get(
                "compute_backend", self.config.default_compute_backend
            )

            # Step 4: Load the data for the current alpha group using the load_group_df method
            group_df = self.load_group_df(
                instruments=instruments,
                expressions=expressions,
                names=names,
                group_name=group_name,
                start_time=start_time,
                end_time=end_time,
                compute_backend=compute_backend,
            )
            self.logger.info(
                f"Finished loading group {group_name} with shape {group_df.shape}"
            )

            # Step 5: Add the loaded data to the result DataFrame, using the alpha group name as a column
            result_df[group_name] = group_df

        # Step 6: Concatenate the data for all alpha groups along the columns (axis=1) and return the result
        concat_df = pd.concat(result_df, axis=1)
        self.logger.info(
            f"Finished loading all groups, resulting dataframe has shape {concat_df.shape}"
        )
        return concat_df

    def load_group_df(
        self,
        instruments: tp.List[str],  # List of instrument names
        expressions: tp.List[str],  # List of expressions to compute
        names: tp.List[str],  # List of names for the computed expressions
        start_time: str,  # Start time for data retrieval
        end_time: str,  # End time for data retrieval
        compute_backend: str,  # Compute backend to use
        group_name: str = "default",  # Name of the alpha group
    ) -> pd.DataFrame:
        # Step 1: Call the compute function to get a dictionary of dataframes
        self.logger.info(
            f"Loading group {group_name} with {len(expressions)} expressions, computing in parallel using {compute_backend}"
        )
        factor_df_dict = self.compute_alpha_expressions(
            expressions=expressions,
            names=names,
            instruments=instruments,
            group_name=group_name,
            start_time=start_time,
            end_time=end_time,
            compute_backend=self.compute_backends[compute_backend],
        )

        # Step 2: Stack each dataframe and save the series with the corresponding name
        self.logger.info(
            f"Finished computing {len(expressions)} expressions, now stacking the dataframes"
        )
        success_factor_names = []
        df_series_values = []
        for name, res in factor_df_dict.items():
            df = res["data"]
            df_series = df.stack(dropna=False)
            df_indices = df_series.index
            df_indices.set_names(["datetime", "instrument"], inplace=True)
            df_series_values.append(df_series.values)
            success_factor_names.append(name)
        big_array = np.stack(df_series_values, axis=1)
        concat_df = pd.DataFrame(
            big_array, index=df_indices, columns=success_factor_names
        )
        self.logger.info(
            f"Finished stacking the dataframes, resulting dataframe has shape {concat_df.shape}"
        )

        return concat_df

    # TODO: This method is too long in its implementation, refactor it.
    def compute_alpha_expressions(
        self,
        expressions: List[str],
        names: List[str],
        group_name: str,
        instruments: List[str],
        compute_backend: ComputeBackend,
        num_workers: int = 100,
        start_time: str = None,
        end_time: str = None,
        dtype: np.dtype = np.float32,
    ) -> tp.Dict[str, pd.DataFrame]:
        """Takes in a list of alpha expressions, and compute them using backend
        compute engine (e.g. numpy, pandas, etc.). A list of expressions are
        taken in at once to avoid potential overhead caused by duplication in
        expression operands.

        All operands (base factors) involved in the expressions are first
        extracted, and then loaded into a dictionary which serves as the context.
        Then the context dictionary are passed into parallel compute routines.

        Parameters
        ----------
        expression : List[str]
            A list of alpha expressions.

        Returns
        -------
        tp.Dict[str, pd.DataFrame]
            A list of alpha results.

        """
        # Extract all operands from the expressions.
        factor_names = {
            (
                backend[:-1],
                factor_name,
            )  # Remove the trailing ':' from backend name
            for expression in expressions
            for backend, factor_name in set(FACTOR_TOKEN_RE.findall(expression))
        }  # {('disk', 'open'), ('factorhub', 'act_vol_d1'), ...}

        def replace_fn(match: re.Match) -> str:
            """Change from "${bkd:factor_name}" to "factor_name"."""
            return match[0][2:-1].split(":")[1]

        factor_name_list = "\n".join(
            [
                f"{{{backend}:{factor_name}}}"
                for backend, factor_name in factor_names
            ]
        )
        self.logger.info(
            f"Loading from storage backends for {len(factor_names)} factors:\n{factor_name_list}"
        )

        # Initialize a dict for storing data
        data_dict: tp.Dict[str, pd.DataFrame] = {}

        # Initialize a list to record the failure factors
        failed_factors = []

        # Iterate over each backend and factor_name pair in factor_names
        for backend, factor_name in factor_names:
            # Try to get the factor data
            try:
                # Use _get_factor method to retrieve the data for the current factor and backend
                factor_data = self._get_factor(
                    factor_name,
                    start_time,
                    end_time,
                    backend=backend,
                    instruments=instruments,
                )
                # If successful, add the factor data to the data_dict with factor_name as the key
                data_dict[factor_name] = factor_data
            except Exception:
                # If _get_factor raises an error, skip this factor and record the failure
                failed_factors.append((factor_name, traceback.format_exc()))

        # Calculate the success rate
        success_rate = (len(factor_names) - len(failed_factors)) / len(
            factor_names
        )

        # Log the failure cases and success rate
        self.logger.info(f"Failure factors: {[f[0] for f in failed_factors]}")
        self.logger.info(f"Success rate: {success_rate * 100}%")

        # If verbose, print the actual traceback for each failure
        if self.config.verbose:
            for factor, tb in failed_factors:
                self.logger.info(f"Failure on factor: {factor}\n{tb}")

        def align_dataframes(
            dfs: tp.Dict[str, pd.DataFrame], join_method: str = "outer"
        ) -> tp.Dict[str, pd.DataFrame]:
            # Depending on the join_method, we'll either union or intersection the indices and columns
            if join_method == "outer":
                indexer = lambda a, b: a.union(b)
            elif join_method == "inner":
                indexer = lambda a, b: a.intersection(b)
            else:
                raise ValueError(
                    f"Unsupported join method: {join_method}. Use 'outer' or 'inner'."
                )

            all_indices = pd.Index([])
            all_columns = pd.Index([])

            for df in dfs.values():
                all_indices = indexer(all_indices, df.index)
                all_columns = indexer(all_columns, df.columns)

            # align each dataframe with all_indices and all_columns
            for key in dfs.keys():
                dfs[key] = dfs[key].reindex(
                    index=all_indices, columns=all_columns, fill_value=0
                )

            return dfs

        data_dict = align_dataframes(data_dict, join_method="outer")
        self.logger.info("Factor aligned")

        for k, v in data_dict.items():
            self.logger.info(
                f"Factor {k} has shape {v.values.shape}, dtype {v.values.dtype}"
            )
        self.logger.info("Changing all dataframes to dtype float32")
        for k, v in data_dict.items():
            data_dict[k] = v.astype(np.float32)

        # Align the rows and columns of dataframes in the context dictionary.
        tick_list = sorted(
            set.intersection(*[set(df.index) for df in data_dict.values()])
        )
        for df in data_dict.values():
            df = df.loc[tick_list]

        ticker_list = sorted(
            set.intersection(*[set(df.columns) for df in data_dict.values()])
        )
        for df in data_dict.values():
            df = df.loc[:, ticker_list]

        # Compute alpha expressions in parallel.
        executor = loky.get_reusable_executor(max_workers=num_workers)
        self.logger.info(f"Computing {group_name} with {num_workers} workers")

        jobs = []
        for i, expression in enumerate(expressions):
            jobs.append(
                executor.submit(
                    compute_fn,
                    data_dict=data_dict,
                    backend=compute_backend,
                    expression=FACTOR_TOKEN_RE.sub(replace_fn, expression),
                    index=i,
                )
            )

        results = {}
        failure_cnt = 0
        for i, job in enumerate(jobs):
            result = job.result()
            status = result["status"]
            if status == "success":
                result["data"] = result["data"].astype(dtype)
                results[names[i]] = result
                if self.config.verbose:
                    self.logger.info(f"Finished alpha {i}: success")
            else:
                failure_cnt += 1
                if self.config.verbose:
                    self.logger.info(f"Finished alpha {i}: failure")
                    self.logger.info(f"Traceback: {result['message']}")

        self.logger.info(
            f"Computing {group_name}, total failures: {failure_cnt}"
        )
        self.logger.info(
            f"{len(expressions)} expressions computed, {failure_cnt} failures, success rate {1 - failure_cnt / len(expressions)}"
        )
        return results

    def _get_factor(
        self,
        factor_name: str,
        start_time: str,
        end_time: str,
        backend: str,
        instruments: str,
    ) -> pd.DataFrame:
        return self.storage_backends[backend].get_factor(
            factor_name, start_time, end_time, instruments
        )


def compute_fn(
    expression: str,
    data_dict: tp.Dict[str, pd.DataFrame],
    backend: ComputeBackend,
    index: int = None,
) -> tp.Dict:
    ret = {}
    try:
        factor_df = backend.compute(
            data=data_dict, expr=expression, index=index
        )
        ret = {
            "status": "success",
            "message": "factor computed successfully",
            "data": factor_df,
        }
    except Exception:
        ret = {
            "status": "failure",
            "message": traceback.format_exc(),
            "data": None,
        }
    return ret
