import os
import typing as tp
from typing import Dict

import dgl
import jsonlines
import pandas as pd
import torch
from dgl import DGLGraph

from q4l.config import ExperimentConfig
from q4l.data.dataset import Q4LDataModule
from q4l.utils.misc import create_instance

from ....base import SpatiotemporalModel


def read_ticker_list(fpath: str) -> tp.List[str]:
    record_df = pd.read_csv(
        fpath,
        header=None,
        delimiter="\t",
        index_col=0,
        keep_default_na=False,
        na_values=["_"],
    )
    tickers = record_df.index.to_list()
    return tickers


def read_wikidata(
    data_dir: str, ticker_index_map: tp.Dict[str, int]
) -> tp.Dict:
    stock_qid_map = {}

    with jsonlines.open(
        os.path.join(data_dir, "stock_records.jsonl"), "r"
    ) as reader:
        for stock_record in reader:
            stock_qid_map[stock_record["qid"]] = stock_record["symbol"]

    ret_data_dict = {}
    with jsonlines.open(
        os.path.join(data_dir, "intra_stock_relations.jsonl"), "r"
    ) as reader:
        for rel in reader:
            src_symbol = stock_qid_map[rel["qid"]]
            dst_symbol = stock_qid_map[rel["value"]]
            if (
                src_symbol not in ticker_index_map
                or dst_symbol not in ticker_index_map
            ):
                continue
            src_node_idx = ticker_index_map[stock_qid_map[rel["qid"]]]
            rel_type = rel["property_id"]
            dst_node_idx = ticker_index_map[stock_qid_map[rel["value"]]]
            eid = f"wiki_{rel_type}"
            if eid not in ret_data_dict:
                ret_data_dict[eid] = [[], []]
            # Undirected, add twice
            ret_data_dict[eid][0].append(src_node_idx)
            ret_data_dict[eid][1].append(dst_node_idx)
            ret_data_dict[eid][0].append(dst_node_idx)
            ret_data_dict[eid][1].append(src_node_idx)

    maxn = 0
    with jsonlines.open(
        os.path.join(data_dir, "intermediate_nodes.jsonl"), "r"
    ) as reader:
        for entry in reader:
            entry["qid"]
            relations = entry["neighbors"]
            if len(relations) > maxn:
                maxn = len(relations)
            for i in range(len(relations)):
                for j in range(i + 1, len(relations)):
                    qid_i, qid_j = relations[i][0], relations[j][0]
                    p1, p2 = relations[i][1], relations[j][1]

                    symbol_i = stock_qid_map[qid_i]
                    symbol_j = stock_qid_map[qid_j]
                    if (
                        symbol_i not in ticker_index_map
                        or symbol_j not in ticker_index_map
                    ):
                        continue

                    nid_i, nid_j = (
                        ticker_index_map[symbol_i],
                        ticker_index_map[symbol_j],
                    )
                    # No self-loop
                    if qid_i == qid_j:
                        continue

                    # Forward
                    eid = f"wiki_{p1}_{p2}"
                    if eid not in ret_data_dict:
                        ret_data_dict[eid] = [[], []]
                    ret_data_dict[eid][0].append(nid_i)
                    ret_data_dict[eid][1].append(nid_j)

                    # Backward
                    eid = f"wiki_{p2}_{p1}"
                    if eid not in ret_data_dict:
                        ret_data_dict[eid] = [[], []]
                    ret_data_dict[eid][0].append(nid_j)
                    ret_data_dict[eid][1].append(nid_i)

    return ret_data_dict


def read_industry(
        data_dir: str, ticker_index_map: tp.Dict[str, int]
) -> tp.Dict:
        ret_data_dict = {}
        df = pd.read_csv(data_dir)
        df["STOCK"] = df["STOCK"].str.replace(".N$", "")
        for industry, group in df.groupby("INDUSTRY_GICS"):
            stocks = group["STOCK"].tolist()
            key = f"industry_{industry}"
            ret_data_dict[key] = [[], []]
            for i in range(len(stocks)):
                for j in range(len(stocks)):
                    if i != j:  # 避免自连接
                        ret_data_dict[key][0].append(ticker_index_map[stocks[i]])
                        ret_data_dict[key][1].append(ticker_index_map[stocks[j]])

        return ret_data_dict


class StockKG:
    def __init__(self, config: ExperimentConfig, data: Q4LDataModule) -> None:
        self.config = config
        train_df = data.prepare("train")[0]
        tickers = train_df.index.get_level_values(1).unique().to_list()
        self.ticker_index_map = {x: i for i, x in enumerate(tickers)}
        data_dict = {}
        wiki_data_dir = os.path.join(
            self.config.data.graph.wikidata_dir, self.config.data.region
        )
        industry_data_dir = os.path.join(
            self.config.data.graph.industry_dir, f"{self.config.data.region}.csv"
        )
        if config.data.graph.use_wikidata:
            data_dict.update(
                read_wikidata(
                    data_dir=wiki_data_dir,
                    ticker_index_map=self.ticker_index_map,
                )
            )
        if config.data.graph.use_industry:
            data_dict.update(
                read_industry(
                    data_dir=industry_data_dir,
                    ticker_index_map=self.ticker_index_map,
                
                )
           )

        # Construct DGL graph
        num_nodes_dict = {"stock": len(self.ticker_index_map)}
        new_data_dict = {}
        for k, v in data_dict.items():
            canonical_etype = ("stock", k, "stock")
            src_tensor = torch.tensor(v[0], dtype=torch.long)
            dsr_tensor = torch.tensor(v[1], dtype=torch.long)
            new_data_dict[canonical_etype] = (src_tensor, dsr_tensor)
        self.big_dgl_graph: DGLGraph = dgl.heterograph(
            new_data_dict, num_nodes_dict
        )

    def get_node_subgraph(self, batch: Dict) -> DGLGraph:
        batch_stock_labels = [l[1] for l in batch["label"][0]]
        node_idx_list = [self.ticker_index_map[x] for x in batch_stock_labels]
        subgraph = dgl.node_subgraph(
            self.big_dgl_graph, {"stock": node_idx_list}
        )
        return subgraph

    def get_info_subgraph(self, info_type: str) -> DGLGraph:
        proper_etype_list = []
        for edge_type in self.big_dgl_graph.etypes:
            if edge_type.startswith(info_type):
                proper_etype_list.append(edge_type)
        return dgl.edge_type_subgraph(
            graph=self.big_dgl_graph, etypes=proper_etype_list
        )


class KGModel(SpatiotemporalModel):
    def __init__(self, config: ExperimentConfig, data: Q4LDataModule):
        self.data = data
        self.kg = StockKG(config=config, data=data)
        super().__init__(config)

    def _build_spatial_model(self):
        return create_instance(
            self.config.model.components.spatial, stock_kg=self.kg
        )

    def get_spatial_info(self, batch: Dict, temporal_info: Dict) -> Dict:
        device = batch["x"].device
        current_hg = self.kg.get_node_subgraph(batch).to(device)
        spatial_info = self.spatial_model(current_hg, temporal_info)
        return {"emb": spatial_info}


if __name__ == "__main__":
    read_wikidata(
        data_dir="/student/wangsaizhuo/Codes/q4l/examples/benchmark/data/wikidata/stock_graph/us",
        ticker_index_map=read_ticker_list(
            "/student/wangsaizhuo/Codes/q4l/examples/benchmark/data/market_data/us/instruments/all.txt"
        ),
    )
    read_industry(
        data_dir="/student/wangsaizhuo/q4l_fengrui/wszlib/examples/benchmark/data/industry/us.csv",
        ticker_index_map=read_ticker_list(
            "/student/wangsaizhuo/Codes/q4l/examples/benchmark/data/market_data/us/instruments/all.txt"
       ),
    )


# class StockHypergraph:
#     def __init__(self, config: ExperimentConfig) -> None:
#         pass

#     def get_subgraph(
#         self, batch: Dict, return_type: str = "matrix"
#     ) -> Union[DGLGraph, np.ndarray]:
#         if return_type == "matrix":
#             return self.get_subgraph_matrix(batch)
#         elif return_type == "graph":
#             return self.get_subgraph_graph(batch)
#         else:
#             raise ValueError(f"Unknown return type {return_type}")


# class HypergraphModel(SpatiotemporalModel):
#     def __init__(self, config: ExperimentConfig):
#         super().__init__(config)
#         self.hypergraph = StockHypergraph(config=config)

#     def get_spatial_info(self, batch: Dict, temporal_info: Dict) -> Dict:
#         current_hg = self.hypergraph.get_subgraph(batch)
#         spatial_info = self.hgnn(current_hg, temporal_info)
#         return {"emb": spatial_info}
