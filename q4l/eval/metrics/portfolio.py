import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series) -> float:
    """Compute annualized return.

    Parameters
    ----------
    returns : pd.Series
        Series of daily returns.

    Returns
    -------
    float
        The annualized return.

    """
    return returns.add(1).prod() ** (252 / returns.count()) - 1


def sharpe_ratio(returns: pd.Series) -> float:
    """Compute Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Series of daily returns.

    Returns
    -------
    float
        The Sharpe ratio.

    """
    return returns.mean() / returns.std() * np.sqrt(252)


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown.

    Parameters
    ----------
    returns : pd.Series
        Series of daily returns.

    Returns
    -------
    float
        The maximum drawdown.

    """
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = cumulative_returns / rolling_max - 1
    return drawdowns.min()


def calmar_ratio(returns: pd.Series) -> float:
    """Compute Calmar ratio.

    Parameters
    ----------
    returns : pd.Series
        Series of daily returns.

    Returns
    -------
    float
        The Calmar ratio.

    """
    ann_ret = annualized_return(returns)
    mdd = max_drawdown(returns)
    return -ann_ret / mdd


def sortino_ratio(returns: pd.Series, target_return=0) -> float:
    """Compute Sortino ratio.

    Parameters
    ----------
    returns : pd.Series
        Series of daily returns.
    target_return : float
        The target or required rate of return.

    Returns
    -------
    float
        The Sortino ratio.

    """
    diff = returns - target_return
    downside_risk = diff[diff < 0].std()
    return returns.mean() / downside_risk * np.sqrt(252)


def portfolio_turnover_rate(
    purchases: pd.Series, sales: pd.Series, total_assets: pd.Series
) -> float:
    """Compute portfolio turnover rate.

    Parameters
    ----------
    purchases : pd.Series
        Series of total purchases each day.
    sales : pd.Series
        Series of total sales each day.
    total_assets : pd.Series
        Series of total asset value each day.

    Returns
    -------
    float
        The portfolio turnover rate.

    """
    avg_total_assets = total_assets.mean()
    total_purchases = purchases.sum()
    total_sales = sales.sum()
    return min(total_purchases, total_sales) / avg_total_assets
