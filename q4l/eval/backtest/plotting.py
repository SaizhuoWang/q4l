# This code is deprecated, it is now just for reference.
import json
import os
from datetime import datetime
from typing import Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from ..constants import PROJECT_ROOT

table_header_template = "{:10s} {:10s} {:10s} {:12s} {:12s} {:10s} {:10s} {:10s}"
table_row_template = "{:10s} {:7.4f} {:7.4f} {:7.4f} {:7.1f} {:7.4f} {:7.4f} {:7.4f}"


def plot_single_ax(data: Dict, ax: plt.Axes, title: str = None, plot_variance: bool = True):
    """Plot_single_ax.

    Parameters
    ----------
    data : Dict
        {
            "alpha1": {
                "ensemble": {
                    "code": 200,
                    "msg": "Success",
                    "data": {...}
                },
                "0": {
                    ...
                },
                ...
            },
            "alpha2": {
                ...
            },
        }
    ax : plt.Axes
        _description_
    title : str, optional
        _description_, by default None

    """
    metric_keys = [
        "ret",
        "sp",
        "maxdown",
        "max_drawback",
        "score",
        "sortino",
        "win_rate",
    ]
    textstr = table_header_template.format("factor", *metric_keys)
    for factor_name, factor_results in data.items():
        all_results = []
        for ens_idx, result in factor_results.items():
            ticks = result["data"]["alpha0"]["main"]["ticks"]
            ticks = [datetime.strptime(t, "%Y%m%d") for t in ticks]
            if (
                ens_idx != "ensemble"
            ):  # Append the backtest result and do not plot this single trial to the axis
                all_results.append(result["data"]["alpha0"]["main"]["e_net_values"])
                continue
            ax.plot(
                ticks,
                result["data"]["alpha0"]["main"]["e_net_values"],
                label=f"{factor_name}_ensemble",
            )
            textstr += "\n" + table_row_template.format(
                factor_name,
                *[result["data"]["alpha0"]["main"]["metrics"]["exceed"][kk] for kk in metric_keys],
            )
        if len(all_results) > 0 and plot_variance:  # Plot the mean and variance shaded lines
            all_results_np = np.array(all_results)
            all_mean = np.mean(all_results_np, axis=0)
            all_std = np.std(all_results_np, axis=0)
            ax.plot(ticks, all_mean, label=f"{factor_name}_all")
            ax.fill_between(
                ticks,
                all_mean - all_std,
                all_mean + all_std,
                alpha=0.2,
                facecolor=ax.get_lines()[-1].get_color(),
            )

    # Set y-axis interval to 0.1
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # Major ticks every 3 months.
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    # Minor ticks every month.
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    # Text in the x axis will be displayed in 'YYYY-mm' format.
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # add grid
    ax.grid(ls="--", which="major", alpha=1)
    ax.grid(ls="--", which="minor", alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Excessive Net Value")

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(
        0.47,
        0.2,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax.legend()
    ax.set_title(title)


def plot(data: Dict, exp_name: str, plot_variance: bool, mode: str = "alpha_sell") -> str:
    axes_per_row = 3
    fig, axes = plt.subplots((len(data) // axes_per_row), axes_per_row, figsize=(50, 30))
    fig.suptitle("Backtest Result for {}".format(exp_name), fontsize=20)
    for i, (k, v) in enumerate(data.items()):
        plot_single_ax(
            v,
            axes[i // axes_per_row, i % axes_per_row],
            title=k,
            plot_variance=plot_variance,
        )

    # Save figure
    fig_savedir = os.path.join(PROJECT_ROOT, "figures", mode)
    os.makedirs(fig_savedir, exist_ok=True)
    fig_savepath = os.path.join(fig_savedir, f"{exp_name}.pdf")
    fig.savefig(fig_savepath, format="pdf")
    return fig_savepath


def plot_backtest_curves(result_path, plot_variance) -> str:
    mode = result_path.split("/")[-2]  # alpha-sell/xd-optimizer
    exp_name = result_path.rsplit("/")[-1].split(".")[0]
    with open(result_path, "r") as f:
        data = json.load(f)

    return plot(data, exp_name, plot_variance)


if __name__ == "__main__":
    # Get result path and parse it
    result_path = input("Result path: ")
    plot_backtest_curves(result_path)
