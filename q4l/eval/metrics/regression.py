import pandas as pd


def mae(
    signal: pd.Series, ret: pd.Series, dimension: str = "instrument"
) -> float:
    """Calculate the MAE of the signal.

    Parameters
    ----------
    signal : pd.Series
        The alpha signal organized as multi-indexed series (datetime, instrument).
    ret : pd.Series
        The return series organized as multi-indexed series (datetime, instrument).
    dimension: str
        The dimension to compute the metric along first.

    Returns
    -------
    float
        The MAE of the signal.

    """
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "Invalid dimension. Expected 'datetime' or 'instrument'."
        )

    # Align signal and ret series and drop NaNs
    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    if dimension == "datetime":
        # Swap levels to have 'datetime' as the inner index
        signal_clean = signal_clean.swaplevel()
        ret_clean = ret_clean.swaplevel()

    # Calculate the MAE
    mae = abs(signal_clean - ret_clean).groupby(level=0).mean().mean()
    return mae


def mse(
    signal: pd.Series, ret: pd.Series, dimension: str = "instrument"
) -> float:
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "Invalid dimension. Expected 'datetime' or 'instrument'."
        )

    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    if dimension == "datetime":
        signal_clean = signal_clean.swaplevel()
        ret_clean = ret_clean.swaplevel()

    mse = ((signal_clean - ret_clean) ** 2).groupby(level=0).mean().mean()
    return mse


def rmse(
    signal: pd.Series, ret: pd.Series, dimension: str = "instrument"
) -> float:
    """Calculate the RMSE of the signal.

    Parameters
    ----------
    signal : pd.Series
        The alpha signal organized as multi-indexed series (datetime, instrument).
    ret : pd.Series
        The return series organized as multi-indexed series (datetime, instrument).
    dimension: str
        The dimension to compute the metric along first.

    Returns
    -------
    float
        The RMSE of the signal.

    """
    import numpy as np

    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "Invalid dimension. Expected 'datetime' or 'instrument'."
        )

    # Align signal and ret series and drop NaNs
    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    if dimension == "datetime":
        # Swap levels to have 'datetime' as the inner index
        signal_clean = signal_clean.swaplevel()
        ret_clean = ret_clean.swaplevel()

    # Calculate the RMSE
    rmse = np.sqrt(
        ((signal_clean - ret_clean) ** 2).groupby(level=0).mean().mean()
    )
    return rmse


def mape(
    signal: pd.Series, ret: pd.Series, dimension: str = "instrument"
) -> float:
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "Invalid dimension. Expected 'datetime' or 'instrument'."
        )

    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    if dimension == "datetime":
        signal_clean = signal_clean.swaplevel()
        ret_clean = ret_clean.swaplevel()

    mape = (
        (abs((signal_clean - ret_clean) / ret_clean))
        .groupby(level=0)
        .mean()
        .mean()
    )
    return mape


def mda(
    signal: pd.Series, ret: pd.Series, dimension: str = "instrument"
) -> float:
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "Invalid dimension. Expected 'datetime' or 'instrument'."
        )

    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    if dimension == "datetime":
        signal_clean = signal_clean.swaplevel()
        ret_clean = ret_clean.swaplevel()

    signal_diff = signal_clean.diff()
    ret_diff = ret_clean.diff()

    mda = ((signal_diff * ret_diff) > 0).groupby(level=0).mean().mean()
    return mda


def r2_score(
    signal: pd.Series, ret: pd.Series, dimension: str = "instrument"
) -> float:
    """Calculate the R-squared of the signal.

    Parameters
    ----------
    signal : pd.Series
        The alpha signal organized as multi-indexed series (datetime, instrument).
    ret : pd.Series
        The return series organized as multi-indexed series (datetime, instrument).
    dimension: str
        The dimension to compute the metric along first.

    Returns
    -------
    float
        The R-squared of the signal.

    """
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "Invalid dimension. Expected 'datetime' or 'instrument'."
        )

    # Align signal and ret series and drop NaNs
    signal, ret = signal.align(ret, join="inner")
    signal_clean = signal.dropna()
    ret_clean = ret.dropna()

    if dimension == "datetime":
        # Swap levels to have 'datetime' as the inner index
        signal_clean = signal_clean.swaplevel()
        ret_clean = ret_clean.swaplevel()

    # Calculate the R-squared
    ss_res = ((signal_clean - ret_clean) ** 2).groupby(level=0).sum()
    ss_tot = (
        ((signal_clean - signal_clean.groupby(level=0).mean()) ** 2)
        .groupby(level=0)
        .sum()
    )
    r2 = 1 - ss_res.sum() / ss_tot.sum()
    return r2
