import networkx as nx
import numpy as np
import pandas as pd
import pyvis
import yaml

from ...qlib import init as qlib_init
from ...qlib.utils import init_instance_by_config

# Load the config file from yaml
with open("config/analyze.yaml", "r") as f:
    config = yaml.safe_load(f)

qlib_init(**config["qlib_init"])
dataset = init_instance_by_config(config["dataset"])

ll_df = dataset.handler._data  # Low-Level DataFrame


def compute_corr(config, df: pd.DataFrame) -> pd.DataFrame:
    """Given the dataframe of raw factor + return, compute the correlation
    matrix of returns among stocks in terms of time steps."""
    returns = df["label"]
    returns.columns = ["returns"]

    time_steps = set(returns.index.get_level_values(0))

    all_step_returns = [returns.loc[step] for step in time_steps]
    return_df = pd.concat(
        all_step_returns, axis=1, keys=time_steps
    ).T.sort_index()
    nan_mask = return_df.isnull()
    nan_cnt = nan_mask.sum()
    mask = nan_cnt < config["nan_threshold"]

    filtered_df = return_df[return_df.columns[mask]]
    corr = filtered_df.corr()
    return corr


def filter_and_discretize(corr: pd.DataFrame, config):
    corr_df = corr.copy()
    mask = corr_df > config["corr_threshold"]
    mask = mask & ~(np.eye(mask.shape[0], dtype=bool))
    return mask.astype(np.int32)


def visualize(nx_g: nx.Graph):
    nt = pyvis.network.Network(height="100%", width="100%")
    nt.from_nx(nx_g)
    nt.show("corr_graph.html")


corr = compute_corr(config, ll_df)
adj_matrix = filter_and_discretize(corr, config)
stock_codes = list(set(corr.columns))

nodes = [(i, {"id": stock_codes[i]}) for i in range(len(stock_codes))]
edges = np.array(np.nonzero(adj_matrix.values)).T

nx_g = nx.Graph()
nx_g.add_nodes_from(nodes)
nx_g.add_edges_from(edges.tolist())

visualize(nx_g)
