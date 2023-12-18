import abc
import bisect

import pandas as pd


class TradeCalendar:
    @abc.abstractmethod
    def is_trade_day(self, trade_date: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_next_trade_day(self, trade_date: str) -> pd.Timestamp:
        raise NotImplementedError

    @abc.abstractmethod
    def get_pre_trade_day(self, trade_date: str) -> pd.Timestamp:
        raise NotImplementedError

    def align(self, date: str, mode: str = "forward") -> pd.Timestamp:
        """Align the date to the nearest trading day.

        Parameters
        ----------
        date : str
            The date to be aligned. Format: YYYYMMDD
        mode : str, optional
            The mode of alignment, by default 'forward'

        Returns
        -------
        str
            The aligned date.

        """
        if self.is_trade_day(date):
            return pd.Timestamp(date)

        if mode == "forward":
            return self.get_next_trade_day(date)
        elif mode == "backward":
            return self.get_pre_trade_day(date)
        else:
            raise ValueError(f"Unknown alignment mode: {mode}.")


class Q4LTradeCalendar(TradeCalendar):
    def __init__(self, calendar_path) -> None:
        with open(calendar_path, "r") as f:
            self.ticks = [pd.Timestamp(tick.strip()) for tick in f.readlines()]

    def is_trade_day(self, trade_date: str) -> bool:
        return pd.Timestamp(trade_date) in self.ticks

    def get_next_trade_day(self, trade_date: str) -> pd.Timestamp:
        trade_date = pd.Timestamp(trade_date)
        if trade_date in self.ticks:
            return trade_date
        next_tick_idx = bisect.bisect_right(self.ticks, trade_date)
        if next_tick_idx >= len(self.ticks):
            raise ValueError("No next trading tick found")
        return self.ticks[next_tick_idx]

    def get_pre_trade_day(self, trade_date: str) -> pd.Timestamp:
        trade_date = pd.Timestamp(trade_date)
        if trade_date in self.ticks:
            return trade_date
        pre_tick_idx = bisect.bisect_left(self.ticks, trade_date) - 1
        if pre_tick_idx < 0:
            raise ValueError("No previous trading tick found")
        return self.ticks[pre_tick_idx]


class TradeCalendarFactory:
    @staticmethod
    def get_calendar(cal_type: str, **kwargs) -> TradeCalendar:
        class_map = {"q4l": Q4LTradeCalendar, }
        return class_map[cal_type](**kwargs)
