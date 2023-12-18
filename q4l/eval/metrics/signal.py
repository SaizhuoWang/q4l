import numpy as np
import pandas as pd


def ic(signal: pd.Series, ret: pd.Series) -> float:
    """Calculate the IC using pure Pandas."""
    aligned = pd.concat([signal, ret], axis=1, keys=["signal", "ret"]).dropna()
    return aligned["signal"].corr(aligned["ret"])


def icir(signal: pd.Series, ret: pd.Series) -> float:
    """Calculate the ICIR using pure Pandas."""
    aligned = pd.concat([signal, ret], axis=1, keys=["signal", "ret"]).dropna()
    ic_values = aligned.groupby(level=0).apply(
        lambda x: x["signal"].corr(x["ret"])
    )
    return ic_values.mean() / ic_values.std()


def rankic(signal: pd.Series, ret: pd.Series) -> float:
    """Calculate the Rank IC using pure Pandas."""
    aligned = pd.concat([signal, ret], axis=1, keys=["signal", "ret"]).dropna()
    return aligned["signal"].rank().corr(aligned["ret"].rank())


def icwin(signal: pd.Series, ret: pd.Series) -> float:
    """Calculate the IC win rate using pure Pandas."""
    aligned = pd.concat([signal, ret], axis=1, keys=["signal", "ret"]).dropna()
    ic_values = aligned.groupby(level=0).apply(
        lambda x: x["signal"].corr(x["ret"])
    )
    return (ic_values > 0).mean()


def signal_return(signal: pd.Series, ret: pd.Series) -> float:
    """Compute the return of vectorized backtest. Signals are normalized to be
    regarded as portfolio weights.

    Parameters
    ----------
    signal : pd.Series
        The alpha signal organized as multi-indexed series (datetime, instrument).
    ret : pd.Series
        The return series organized as multi-indexed series (datetime, instrument).

    Returns
    -------
    float
        The return of vectorized backtest.

    """
    # Align signal and ret series and drop NaNs
    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    # Apply the softmax function to the signals at each time point to get the portfolio weights
    weights = signal_clean.groupby(level=0, group_keys=False).apply(
        lambda x: np.exp(x) / np.sum(np.exp(x))
    )

    # Compute the weighted returns
    weighted_returns = weights * ret_clean

    # Compute the portfolio return at each time point
    portfolio_returns = weighted_returns.groupby(
        level=0, group_keys=False
    ).sum()

    # Compute and return the total return of the portfolio
    total_return = portfolio_returns.sum()
    return total_return


def signal_turnover(signal: pd.Series) -> float:
    """Compute the turnover of an alpha factor. Signals are normalized to be
    regarded as portfolio weights.

    Parameters
    ----------
    signal : pd.Series
        The alpha signal organized as multi-indexed series (datetime, instrument).

    Returns
    -------
    float
        The turnover of an alpha factor.

    """
    # Apply the softmax function to the signals at each time point to get the portfolio weights
    weights = (
        signal.groupby(level=0, group_keys=False)
        .apply(lambda x: np.exp(x) / np.sum(np.exp(x)))
        .copy()
    )

    # Drop NaNs
    weights.dropna(inplace=True)

    # Sort the index to ensure that the dates are in ascending order
    weights.sort_index(inplace=True)

    # Compute the absolute changes in weights between each time period
    weight_changes = weights.groupby(level=1, group_keys=False).diff().abs()

    # Compute and return the turnover
    turnover = weight_changes.sum()
    return turnover
