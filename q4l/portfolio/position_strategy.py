# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
from abc import ABC

import pandas as pd

from ..qlib.backtest.decision import TradeDecisionWO
from ..qlib.backtest.position import Position
from ..qlib.backtest.signal import create_signal_from
from ..qlib.contrib.strategy.order_generator import OrderGenWOInteract
from ..qlib.strategy.base import BaseStrategy


class BasePositionStrategy(BaseStrategy, ABC):
    def __init__(
        self,
        *,
        position: pd.DataFrame = None,
        risk_degree: float = 0.95,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        signal :
            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`
            the decision of the strategy will base on the given signal
        risk_degree : float
            position percentage of total value.
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:
                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.

        """
        super().__init__(
            level_infra=level_infra,
            common_infra=common_infra,
            trade_exchange=trade_exchange,
            **kwargs,
        )

        self.risk_degree = risk_degree
        self.order_generator = OrderGenWOInteract()

        self.signal: pd.DataFrame = create_signal_from(position)

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(
            trade_step
        )
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(
            trade_step, shift=1
        )
        pred_score = self.signal.get_signal(
            start_time=pred_start_time, end_time=pred_end_time
        )

        target_position = pred_score.dropna().to_dict()

        current_temp: Position = copy.deepcopy(self.trade_position)
        order_list = self.order_generator.generate_order_list_from_target_weight_position(
            current=current_temp,
            trade_exchange=self.trade_exchange,
            target_weight_position=target_position,
            risk_degree=self.risk_degree,
            pred_start_time=pred_start_time,
            pred_end_time=pred_end_time,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )
        print(f"Number of orders: {len(order_list)}")
        return TradeDecisionWO(order_list=order_list, strategy=self)
