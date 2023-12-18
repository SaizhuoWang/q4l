# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Generator, Optional, Tuple, Union, cast

import pandas as pd

from .decision import BaseTradeDecision
from .report import Indicator

if TYPE_CHECKING:
    from ..strategy.base import BaseStrategy
    from ..backtest.executor import BaseExecutor

from ..utils.time import Freq

PORT_METRIC_TYPE = Dict[str, Tuple[pd.DataFrame, dict]]
INDICATOR_METRIC_TYPE = Dict[str, Tuple[pd.DataFrame, Indicator]]
PORT_METRIC_KEY = "portfolio_dict"
INDICATOR_KEY = "indicator_dict"


def backtest_loop(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    trade_strategy: BaseStrategy,
    trade_executor: BaseExecutor,
) -> Tuple[PORT_METRIC_TYPE, INDICATOR_METRIC_TYPE]:
    """Backtest function for the interaction of the outermost strategy and
    executor in the nested decision execution.

    please refer to the docs of `collect_data_loop`

    Returns
    -------
    portfolio_dict: PORT_METRIC
        it records the trading portfolio_metrics information
    indicator_dict: INDICATOR_METRIC
        it computes the trading indicator

    """
    return_value: dict = {}
    for _decision in collect_data_loop(
        start_time, end_time, trade_strategy, trade_executor, return_value
    ):
        pass

    portfolio_dict = cast(PORT_METRIC_TYPE, return_value.get(PORT_METRIC_KEY))
    indicator_dict = cast(
        INDICATOR_METRIC_TYPE, return_value.get(INDICATOR_KEY)
    )

    return portfolio_dict, indicator_dict


def collect_data_loop(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    trade_strategy: BaseStrategy,
    trade_executor: BaseExecutor,
    return_value: dict | None = None,
) -> Generator[BaseTradeDecision, Optional[BaseTradeDecision], None]:
    """Generator for collecting the trade decision data for rl training.

    Parameters
    ----------
    start_time : Union[pd.Timestamp, str]
        closed start time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
    end_time : Union[pd.Timestamp, str]
        closed end time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
        E.g. Executor[day](Executor[1min]), setting `end_time == 20XX0301` will include all the minutes on 20XX0301
    trade_strategy : BaseStrategy
        the outermost portfolio strategy
    trade_executor : BaseExecutor
        the outermost executor
    return_value : dict
        used for backtest_loop

    Yields
    -------
    object
        trade decision

    """
    trade_executor.reset(start_time=start_time, end_time=end_time)
    trade_strategy.reset(level_infra=trade_executor.get_level_infra())

    # with tqdm(total=trade_executor.trade_calendar.get_trade_len(), desc="backtest loop") as bar:
    #     _execute_result = None
    #     while not trade_executor.finished():
    #         # Get today's trading decision
    #         _trade_decision: BaseTradeDecision = trade_strategy.generate_trade_decision(_execute_result)
    #         # Execute all orders in the decision for 1 trading tick
    #         _execute_result = yield from trade_executor.collect_data(_trade_decision, level=0)
    #         # Update the strategy's state with the execution result
    #         trade_strategy.post_exe_step(_execute_result)
    #         # Update progress bar
    #         bar.update(1)
    #     trade_strategy.post_upper_level_exe_step()

    import time

    trade_executor.trade_calendar.get_trade_len()
    iteration = 0

    _execute_result = None
    while not trade_executor.finished():
        iteration += 1
        # print(f"Running iteration {iteration}/{trade_len}")

        # Get today's trading decision
        start_time = time.time()
        _trade_decision: BaseTradeDecision = (
            trade_strategy.generate_trade_decision(_execute_result)
        )
        end_time = time.time()
        # print(f"Step 1 - generate_trade_decision took: {end_time - start_time:.4f} seconds")

        # Execute all orders in the decision for 1 trading tick
        start_time = time.time()
        _execute_result = yield from trade_executor.collect_data(
            _trade_decision, level=0
        )
        end_time = time.time()
        # print(f"Step 2 - collect_data took: {end_time - start_time:.4f} seconds")

        # Update the strategy's state with the execution result
        start_time = time.time()
        trade_strategy.post_exe_step(_execute_result)
        end_time = time.time()
        # print(f"Step 3 - post_exe_step took: {end_time - start_time:.4f} seconds")

    trade_strategy.post_upper_level_exe_step()

    if return_value is not None:
        all_executors = trade_executor.get_all_executors()
        all_portfolio_metrics = {
            "{}{}".format(
                *Freq.parse(_executor.time_per_step)
            ): _executor.trade_account.get_portfolio_metrics()
            for _executor in all_executors
            if _executor.trade_account.is_port_metr_enabled()
        }
        all_indicators = {}
        for _executor in all_executors:
            key = "{}{}".format(*Freq.parse(_executor.time_per_step))
            all_indicators[key] = (
                _executor.trade_account.get_trade_indicator().generate_trade_indicators_dataframe(),
                _executor.trade_account.get_trade_indicator(),
            )
            # all_indicators[
            #     key
            # ] = _executor.trade_account.get_trade_indicator().generate_trade_indicators_dataframe()
            # all_indicators[key + "_obj"] = _executor.trade_account.get_trade_indicator()
        return_value.update(
            {
                PORT_METRIC_KEY: all_portfolio_metrics,
                INDICATOR_KEY: all_indicators,
            }
        )
