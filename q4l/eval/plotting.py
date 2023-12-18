import os
import random
import typing as tp

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil import parser
from joblib import Parallel, delayed
from matplotlib.ticker import MultipleLocator
from scipy.stats import kurtosis, skew

from ..qlib.workflow import R
from ..utils.log import get_logger


def plot_return_curve(
    df: pd.DataFrame,
    intervals: tp.List[tp.Tuple] = None,
    artifact_uri: str = ".",
):
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    df["portfolio_cum"] = (df["return"] + 1.0).cumprod() - 1.0
    df["bm_cum"] = (df["bench"] + 1.0).cumprod() - 1.0
    df["excess_cum"] = df["portfolio_cum"] - df["bm_cum"]
    # import pdb; pdb.set_trace()
    ticks = pd.to_datetime(df.index)
    ax.plot(ticks, df["portfolio_cum"], label="cumulative return")
    ax.plot(ticks, df["bm_cum"], label="benchmark")
    (excess_return_line,) = ax.plot(
        ticks, df["excess_cum"], label="excess return"
    )
    # Fill area under the excess return curve
    ax.fill_between(
        ticks,
        df["excess_cum"],
        color=excess_return_line.get_color(),
        alpha=0.3,
    )

    # Set the x-ticks to be at regular intervals
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=3)
    )  # major ticks every month
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d")
    )  # format the ticks to show year and month
    ax.xaxis.set_minor_locator(
        mdates.MonthLocator()
    )  # minor ticks every Monday

    # Eliminate margins on the x-axis
    ax.margins(x=0)

    # Highlight specified time ranges
    if intervals is not None:
        for i, interval in enumerate(intervals):
            ax.axvspan(
                pd.Timestamp(parser.parse(interval[0])),
                pd.Timestamp(parser.parse(interval[1])),
                facecolor=(
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                ),
                alpha=0.1,
            )

    # Set the y-ticks to be at regular intervals
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Set grid
    ax.grid(True, which="both")

    # Set minor grid lines
    ax.grid(which="minor", alpha=0.3)
    ax.grid(which="major", alpha=1)

    # Set the legend, title, and labels.
    ax.legend(loc="upper left")
    ax.set_title("Portfolio and Benchmark Cumulative Returns")
    ax.set_xlabel("Trading Ticks")
    ax.set_ylabel("Returns")

    # Show the plot
    fig.tight_layout()
    fig.savefig(
        os.path.join(artifact_uri, "portfolio_analysis", "return_curve.png"),
        format="png",
    )


def plot_return_curves(
    dataframes: tp.List[pd.DataFrame],
    ensemble_df: pd.DataFrame,
    intervals: tp.List[tp.Tuple] = None,
    artifact_uri: str = ".",
):
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    # Calculate the portfolio cumulatives for each dataframe and store
    portfolio_cums = []
    for df in dataframes:
        df["portfolio_cum"] = (df["return"] + 1.0).cumprod() - 1.0
        portfolio_cums.append(df["portfolio_cum"].values)

    # Convert to numpy for easier calculations
    portfolio_cums = np.array(portfolio_cums)

    # Calculate mean and std
    mean_portfolio_cum = np.mean(portfolio_cums, axis=0)
    std_portfolio_cum = np.std(portfolio_cums, axis=0)

    # Plot mean curve
    ticks = pd.to_datetime(dataframes[0].index)
    ax.plot(ticks, mean_portfolio_cum, label="Mean Cumulative Return")

    # Shade 1 std region
    ax.fill_between(
        ticks,
        mean_portfolio_cum - std_portfolio_cum,
        mean_portfolio_cum + std_portfolio_cum,
        alpha=0.3,
        label="1-std deviation",
    )

    # Plot ensemble curve
    ensemble_df["ensemble_cum"] = (ensemble_df["return"] + 1.0).cumprod() - 1.0
    ax.plot(
        ticks,
        ensemble_df["ensemble_cum"],
        label="Ensemble Cumulative Return",
        linestyle="--",
    )

    # ... [rest of the function remains unchanged]
    # Set the x-ticks to be at regular intervals
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=3)
    )  # major ticks every month
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d")
    )  # format the ticks to show year and month
    ax.xaxis.set_minor_locator(
        mdates.MonthLocator()
    )  # minor ticks every Monday

    # Eliminate margins on the x-axis
    ax.margins(x=0)

    # Highlight specified time ranges
    if intervals is not None:
        for i, interval in enumerate(intervals):
            ax.axvspan(
                pd.Timestamp(parser.parse(interval[0])),
                pd.Timestamp(parser.parse(interval[1])),
                facecolor=(
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                ),
                alpha=0.1,
            )

    # Set the y-ticks to be at regular intervals
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Set grid
    ax.grid(True, which="both")

    # Set minor grid lines
    ax.grid(which="minor", alpha=0.3)
    ax.grid(which="major", alpha=1)

    # Set the legend, title, and labels.
    ax.legend(loc="upper left")
    ax.set_title("Portfolio and Benchmark Cumulative Returns")
    ax.set_xlabel("Trading Ticks")
    ax.set_ylabel("Returns")

    # Show the plot
    fig.tight_layout()
    fig.savefig(
        os.path.join(artifact_uri, "portfolio_analysis", "return_curve.png"),
        format="png",
    )


def plot_grouped_backtest(backtest_result: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    ticks = pd.to_datetime(backtest_result.index)
    for col in backtest_result.columns:
        plt.plot(ticks, backtest_result[col], label=col)

    plt.legend(loc="best")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.title("Grouped Backtest Results")
    plt.grid(True)
    plt.savefig(f"ag.png", format="png")


class FactorPlotter:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.logger = get_logger(self)

    def plot_factor_distribution(
        self,
        factor_group: str,
        save_dir: tp.Optional[str] = None,
        postfix: str = "",
    ):
        """For each factor in the factor group, plot its overall value
        distribution."""
        if save_dir is None:
            save_dir = os.path.join(
                R.artifact_uri,
                "plots",
                "factor_distributions",
                postfix,
            )
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info(
            f"Plotting distribution of {factor_group} into {save_dir}"
        )

        # Get the factors in the group
        factors = self.data[factor_group]

        def plot_fn(factor_name: str, factors, save_dir):
            # Filter the data for the specific factor
            factor_data = factors[factor_name]

            # Create the figure and the axes
            fig, ax = plt.subplots()

            # Plot the distribution with a histogram and KDE
            sns.histplot(factor_data.values, kde=False, ax=ax)

            # Set the title and show the plot
            ax.set_title(f"Distribution of {factor_name}")
            filename = os.path.join(save_dir, f"{factor_name}_distribution.png")
            plt.savefig(filename)
            plt.close(fig)

        Parallel(n_jobs=100)(
            delayed(plot_fn)(factor_name, factors, save_dir)
            for factor_name in factors.columns
        )

    def plot_factor_statistics(
        self,
        factor_group: str,
        save_dir: tp.Optional[str] = None,
        postfix: str = "",
    ):
        """For each factor in the factor group, plot value distribution of
        statistics over time or across different stocks."""
        if save_dir is None:
            save_dir = os.path.join(
                R.artifact_uri, "plots", "factor_statistics", postfix
            )
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info(
            f"Plotting distribution of {factor_group} into {save_dir}"
        )

        # Get the factors in the group
        factors = self.data[factor_group]

        # Prepare a DataFrame to store the statistics
        stats_df = pd.DataFrame()

        def stat_fn(factor_name: str, factors, save_dir):
            # Compute the statistics along the time dimension
            stats = factors[factor_name].groupby(level=0).describe()
            stats["skew"] = factors[factor_name].groupby(level=0).apply(skew)
            stats["kurtosis"] = (
                factors[factor_name].groupby(level=0).apply(kurtosis)
            )
            stats["NaN_ratio"] = factors[factor_name].isna().sum() / len(
                factors[factor_name]
            )

            # stats_df = pd.concat([stats_df, stats], axis=1, keys=[factor_name])

            # For each statistic, create a plot
            for statistic in [
                "mean",
                "std",
                "50%",
                "min",
                "max",
                "skew",
                "kurtosis",
                "NaN_ratio",
            ]:
                # Create the figure and the axes
                fig, ax = plt.subplots()

                # Plot the distribution of the statistic with a histogram and KDE
                sns.histplot(stats[statistic], kde=True, ax=ax)

                # Set the title and show the plot
                ax.set_title(
                    f"Distribution of {statistic} of {factor_name} Across Stocks"
                )
                filename = os.path.join(
                    save_dir, f"{factor_name}_{statistic}_distribution.png"
                )
                plt.savefig(filename)
                plt.close(fig)

                self.logger.info(
                    f"Saved {statistic} distribution plot of {factor_name} at {filename}"
                )
            return stats

        stats_list = Parallel(n_jobs=100)(
            delayed(stat_fn)(factor_name, factors, save_dir)
            for factor_name in factors.columns
        )
        stats_df = pd.concat(stats_list, axis=1, keys=factors.columns)

        # Save the statistics DataFrame to a CSV file
        stats_filename = os.path.join(save_dir, f"{factor_group}_stats.csv")
        stats_df.to_csv(stats_filename)
        self.logger.info(
            f"Saved statistics of {factor_group} at {stats_filename}"
        )
