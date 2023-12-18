import argparse

import pandas as pd


class Q4LBacktester:
    def __init__(self) -> None:
        pass

    def backtest_alpha(self, alpha: pd.DataFrame, **kwargs):
        pass

    def backtest_position(self, position: pd.DataFrame, **kwargs):
        pass


def grouped_backtest(pred: pd.Series, label: pd.Series, N: int) -> pd.DataFrame:
    # Ensure pred and label have the same index
    assert pred.index.equals(label.index), "Indexes of prediction and label must be the same"

    # Create a DataFrame to hold predictions and labels
    data = pd.concat([pred, label], axis=1)
    data.columns = ["pred", "label"]

    # Create a column to hold the group number
    data["group"] = data.groupby(level=0)["pred"].transform(
        lambda x: pd.qcut(x, N, labels=False, duplicates="drop")
    )

    # Calculate average return for each group
    group_returns = data.groupby(["group", data.index.get_level_values(0)])["label"].mean()

    # Calculate cumulative returns for each group
    cumulative_returns = group_returns.groupby(level=0).cumsum()

    # # Add 'long_short' and 'long_only' strategies
    # cumulative_returns['long_short'] = cumulative_returns.loc[N-1] - cumulative_returns.loc[0]
    # cumulative_returns['long_only'] = cumulative_returns.loc[N-1]

    return cumulative_returns

