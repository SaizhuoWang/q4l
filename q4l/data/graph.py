import typing as tp
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import dgl
import numpy as np
import torch
from attr import dataclass
from dateutil import rrule
from dgl import DGLGraph


@dataclass
class Edge:
    source: str
    destination: str
    start_time: datetime
    end_time: datetime
    info_source: str = None
    edge_type: str = None


@dataclass
class HyperEdge:
    nodes: list
    start_time: datetime
    end_time: datetime
    info_source: str = None
    edge_type: str = None


class StockGraph:
    def __init__(
        self, start_time: datetime, end_time: datetime, resolution: str
    ):
        self.nodes = list()

        self.edge_dict = defaultdict(list)  # List for simple edges
        self.all_edges = []  # List for all edges

        self.hyperedge_dict = defaultdict(list)  # List for hyperedges
        self.all_hyperedges = []  # List for all hyperedges

        # Node-edge adjacency list index
        self.node_src_edges = defaultdict(list)
        self.node_dst_edges = defaultdict(list)
        self.node_hyperedges = defaultdict(list)
        self.start_time = start_time
        self.end_time = end_time
        self.resolution = resolution
        self.ticks = self._generate_ticks()

        # Edge types
        self.edge_types = []

        self.hyperedge_incimatrix_dict = {}

        self.is_dirty = True  # Dirty flag

    @property
    def num_edge_types(self):
        return len(self.edge_types)

    def _generate_ticks(self):
        """Generate a range of time ticks based on the specified resolution."""
        rule_map = {
            "year": rrule.YEARLY,
            "month": rrule.MONTHLY,
            "week": rrule.WEEKLY,
            "day": rrule.DAILY,
            "hour": rrule.HOURLY,
            "minute": rrule.MINUTELY,
            "second": rrule.SECONDLY,
        }

        if self.resolution not in rule_map:
            raise ValueError("Unsupported resolution")

        return list(
            rrule.rrule(
                freq=rule_map[self.resolution],
                dtstart=self.start_time,
                until=self.end_time,
            )
        )

    def add_node(self, node: str):
        self.nodes.append(node)
        self.is_dirty = True

    def add_nodes(self, nodes: tp.List[str]):
        self.nodes.extend(nodes)
        self.is_dirty = True

    def add_edge(
        self,
        source: str,
        destination: str,
        start_time: datetime = None,
        end_time: datetime = None,
        info_source: str = None,
        edge_type: str = None,
    ):
        """Add a simple edge to the graph."""
        if source not in self.nodes or destination not in self.nodes:
            raise ValueError(
                f"Source `{source}` or destination `{destination}` not found in graph"
            )
        if (start_time is not None and start_time >= self.end_time) or (
            end_time is not None and end_time <= self.start_time
        ):
            warnings.warn("Edge is out of graph time range, skipping.")
            return

        edge = Edge(
            source=source,
            destination=destination,
            start_time=start_time or self.start_time,
            end_time=end_time or self.end_time,
            info_source=info_source,
            edge_type=edge_type,
        )
        edge_type = f"{info_source}_{edge_type}"
        if edge_type not in self.edge_types:
            self.edge_types.append(edge_type)

        edge_index = len(self.all_edges)
        self.all_edges.append(edge)

        # Add edge index to aux info
        self.edge_dict[info_source].append(edge_index)
        self.node_src_edges[source].append(edge_index)
        self.node_dst_edges[destination].append(edge_index)

        self.is_dirty = True

    def add_hyperedge(
        self,
        nodes: List[str],
        start_time: int,
        end_time: int,
        info_source: Optional[str] = None,
        edge_type: Optional[str] = None,
    ) -> None:
        """Add a hyperedge to the graph.

        This method creates and adds a hyperedge to the graph. A hyperedge is defined
        by its nodes, start and end times, information source, and edge type.

        Parameters:
        nodes (List[Any]): A list of nodes that the hyperedge connects.
        start_time (int): The start time of the hyperedge.
        end_time (int): The end time of the hyperedge.
        info_source (Optional[str]): The source of information for the hyperedge, default is None.
        edge_type (Optional[str]): The type of the hyperedge, default is None.

        Raises:
        ValueError: If one or more nodes are not found in the graph.

        Returns:
        None

        """
        if not all(node in self.nodes for node in nodes):
            raise ValueError("One or more nodes not found in graph")

        hyperedge = HyperEdge(
            nodes=nodes,
            start_time=start_time or self.start_time,
            end_time=end_time or self.end_time,
            info_source=info_source,
            edge_type=edge_type,
        )

        # Put the hyperedge into the graph
        hyperedge_index = len(self.all_hyperedges)
        self.all_hyperedges.append(hyperedge)
        self.hyperedge_dict[info_source].append(hyperedge_index)
        # Update the node-hyperedge incidence list
        for node in nodes:
            self.node_hyperedges[node].append(hyperedge_index)

        self.is_dirty = True

    def add_hyperedge_matrix(
        self,
        inci_matrix: np.ndarray,  # (N, T, D)
        start_time: datetime,
        end_time: datetime,
        info_source: Optional[str] = None,
    ) -> None:
        """Aligns the given incidence matrix with the graph's time range and
        adds it to the graph.

        Parameters
        ----------
        inci_matrix : np.ndarray
            Incidence matrix of shape (N, T, D).
        start_time : datetime
            Start time corresponding to the first time tick in the incidence matrix.
        end_time : datetime
            End time corresponding to the last time tick in the incidence matrix.
        info_source : Optional[str]
            Identifier for the source of the incidence matrix.

        """
        # Compute the total number of ticks in the graph's time range
        graph_total_ticks = (self.end_time - self.start_time).days + 1

        # Compute the start and end indices for the incidence matrix in the graph's time range
        inci_start_index = (start_time - self.start_time).days
        inci_end_index = inci_start_index + inci_matrix.shape[1]

        # Adjust indices if the graph's time range starts later than the incidence matrix
        if inci_start_index < 0:
            inci_matrix = inci_matrix[:, -inci_start_index:, :]
            inci_start_index = 0

        # Trim the incidence matrix if it extends beyond the graph's time range
        if inci_end_index > graph_total_ticks:
            inci_matrix = inci_matrix[
                :, : graph_total_ticks - inci_start_index, :
            ]
            inci_end_index = graph_total_ticks

        # Initialize an aligned incidence matrix
        aligned_inci_matrix = np.zeros(
            (inci_matrix.shape[0], graph_total_ticks, inci_matrix.shape[2]),
            dtype=inci_matrix.dtype,
        )

        # Align and add the incidence matrix to the graph
        aligned_inci_matrix[:, inci_start_index:inci_end_index, :] = inci_matrix
        self.hyperedge_incimatrix_dict[info_source] = aligned_inci_matrix

    def _prepare_graph(self):
        """Prepare the graph for querying by re-indexing.

        This step is essentially an indexing step.

        """
        num_nodes = len(self.nodes)
        num_edges = len(self.all_edges)
        num_ticks = len(self.ticks)
        num_hyperedges = len(self.all_hyperedges)
        self.node_index_dict = {
            node: index for index, node in enumerate(self.nodes)
        }

        # Make index on the node side
        self.node_edge_src = np.zeros((num_nodes, num_edges), dtype=bool)
        self.node_edge_dst = np.zeros((num_nodes, num_edges), dtype=bool)
        self.hyperedge_incidence = np.zeros(
            (num_nodes, num_hyperedges), dtype=bool
        )
        for index, node in enumerate(self.nodes):
            # For simple edges, check if the node is the source or destination
            edge_src_index = self.node_src_edges[node]
            edge_dst_index = self.node_dst_edges[node]
            hyperedge_contain_index = self.node_hyperedges[node]
            self.node_edge_src[index][edge_src_index] = True
            self.node_edge_dst[index][edge_dst_index] = True
            self.hyperedge_incidence[index][hyperedge_contain_index] = True

        # Make temporal slices
        self.edge_temporal_validity = np.zeros(
            (num_ticks, num_edges), dtype=bool
        )
        self.hyperedge_temporal_validity = np.zeros(
            (num_ticks, num_hyperedges), dtype=bool
        )
        self.edge_src_list = []
        self.edge_dst_list = []
        for edge_id, edge in enumerate(self.all_edges):
            if (
                edge.end_time <= self.start_time
                or edge.start_time >= self.end_time
            ):
                self.edge_temporal_validity[..., edge_id] = False
                continue
            start_tick = self.ticks.index(max(edge.start_time, self.start_time))
            end_tick = self.ticks.index(min(edge.end_time, self.end_time))
            self.edge_src_list.append(edge.source)
            self.edge_dst_list.append(edge.destination)
            self.edge_temporal_validity[
                start_tick : end_tick + 1, edge_id
            ] = True
        self.edge_src_list = np.array(self.edge_src_list)
        self.edge_dst_list = np.array(self.edge_dst_list)
        # for edge_id, hyperedge in enumerate(self.hyperedges):
        #     start_tick = self.ticks.index(hyperedge.start_time)
        #     end_tick = self.ticks.index(hyperedge.end_time)
        #     self.hyperedge_temporal_validity[
        #         start_tick : end_tick + 1, edge_id
        #     ] = True

        self.is_dirty = False

    def get_snapshot(
        self,
        timestamp: datetime,
        query_nodes: tp.List[str],
        info_source: tp.List[str] = None,
    ) -> tp.Tuple[tp.Dict, tp.Dict]:
        """Retrieve a snapshot of the graph at a given timestamp for specified
        nodes."""
        if self.is_dirty:
            self._prepare_graph()
        node_pool_index = np.array(
            [self.node_index_dict[node] for node in query_nodes]
        )
        # Filter edges
        # Filter node req
        valid_start_edges = self.node_edge_src[node_pool_index, :]
        valid_end_edges = self.node_edge_dst[node_pool_index, :]
        valid_node_edges = np.logical_and(
            np.any(valid_start_edges, axis=0), np.any(valid_end_edges, axis=0)
        )
        # Filter tick req
        valid_tick_edges = self.edge_temporal_validity[
            self.ticks.index(timestamp)
        ]
        # Filter info source req
        if info_source is None:
            valid_info_edges = np.ones(len(self.all_edges), dtype=bool)
        else:
            valid_info_edges = np.zeros(len(self.all_edges), dtype=bool)
            for source in info_source:
                valid_info_edges[self.edge_dict[source]] = True

        # Filter out valid edges
        valid_edge_index = np.nonzero(
            valid_node_edges & valid_tick_edges & valid_info_edges
        )[0]
        valid_edges = [self.all_edges[i] for i in valid_edge_index]
        valid_src_dst = {
            "src": self.edge_src_list[valid_edge_index],
            "dst": self.edge_dst_list[valid_edge_index],
        }
        edge_dict = {
            "edges": valid_edges,
            "src_dst": valid_src_dst,
        }

        # Filter hyperedges
        # Previous method
        # valid_tick_hyperedges = self.hyperedge_temporal_validity[
        #     self.ticks.index(timestamp)
        # ]
        # valid_node_hyperedges = np.any(
        #     self.hyperedge_incidence[node_pool_index, :], axis=0
        # )
        # if info_source is None:
        #     valid_info_hyperedges = np.ones(
        #         len(self.all_hyperedges), dtype=bool
        #     )
        # else:
        #     valid_info_hyperedges = np.zeros(
        #         len(self.all_hyperedges), dtype=bool
        #     )
        #     for source in info_source:
        #         valid_info_hyperedges[self.hyperedge_dict[source]] = True
        # valid_hyperedge_index = np.nonzero(
        #     valid_tick_hyperedges
        #     & valid_node_hyperedges
        #     & valid_info_hyperedges
        # )[0]
        # selected_hyperedges = [
        #     self.all_hyperedges[i] for i in valid_hyperedge_index
        # ]
        # hyperedge_names = [
        #     hyperedge.edge_type for hyperedge in selected_hyperedges
        # ]
        # hyp_inci_matrix = self.hyperedge_incidence[query_nodes][
        #     :, valid_hyperedge_index
        # ]
        if len(self.hyperedge_incimatrix_dict) == 0:
            hyp_info = None
        else:
            # New method
            hyperedge_names = list(self.hyperedge_incimatrix_dict.keys())
            timestamp_index = self.ticks.index(timestamp)
            # Extract all info sources contributing to hyperedge relationships
            full_inci_matrix = [
                m[:, timestamp_index]
                for m in self.hyperedge_incimatrix_dict.values()
            ]
            full_inci_matrix = np.stack(full_inci_matrix, axis=0)
            pool_index = [self.nodes.index(x) for x in query_nodes]
            hyp_inci_matrix = full_inci_matrix[:, pool_index]
            hyp_info = {
                "hyperedge_names": hyperedge_names,
                "hyperedge_inci_matrix": hyp_inci_matrix,
            }

        return edge_dict, hyp_info

    @staticmethod
    def construct_dgl_graph(
        nodes: tp.List[str], edges: tp.Dict, hyperedges: tp.Dict
    ) -> DGLGraph:
        """Constructs a DGLGraph from nodes, edges, and hyperedges.

        Parameters
        ----------
        nodes : List[str]
            List of node names.
        edges : List[Edge]
            List of Edge objects, where each Edge contains attributes like source,
            destination, edge_type, and info_source.
        hyperedges : Dict
            A dictionary containing hyperedge information, including 'hyperedge_names'
            and 'hyperedge_inci_matrix'.

        Returns
        -------
        DGLGraph
            A DGLGraph object representing the graph structure defined by the nodes, edges, and hyperedges.

        Examples
        --------
        >>> nodes = ["node1", "node2", "node3"]
        >>> edges = [Edge(source="node1", destination="node2", edge_type="type1", info_source="source1"),
                     Edge(source="node2", destination="node3", edge_type="type2", info_source="source2")]
        >>> hyperedges = {
                "hyperedge_names": ["hyperedge1", "hyperedge2"],
                "hyperedge_inci_matrix": torch.tensor([[1, 0], [0, 1]])
            }
        >>> graph = GraphConstructor.construct_dgl_graph(nodes, edges, hyperedges)

        """

        # Create data_dict
        data_dict = {}
        for edge in edges["edges"]:
            edge_type = edge.edge_type
            info_type = edge.info_source
            etype = f"{info_type}_{edge_type}"
            src = nodes.index(edge.source)
            dst = nodes.index(edge.destination)
            canonical_etype = ("stock", etype, "stock")
            if canonical_etype not in data_dict:
                data_dict[canonical_etype] = ([], [])
            data_dict[canonical_etype][0].append(src)
            data_dict[canonical_etype][1].append(dst)

        # Convert node indices list to tensor
        for k, v in data_dict.items():
            data_dict[k] = (torch.tensor(v[0]), torch.tensor(v[1]))

        # Create graph
        if len(data_dict) == 0:
            data_dict[("stock", "dummy", "stock")] = ([], [])
        g = dgl.heterograph(
            data_dict=data_dict,
            num_nodes_dict={"stock": len(nodes)},
        )
        if hyperedges is not None:
            g.hyperedge_names = hyperedges["hyperedge_names"]
            g.hyperedge_inci_matrix = hyperedges["hyperedge_inci_matrix"]

        return g

    @staticmethod
    def add_hyperedge_simpleedge(
        g: DGLGraph,
        inci_matrix: np.ndarray,
        info: str,
        names: Optional[List[str]] = None,
    ) -> DGLGraph:
        """Given a DGLGraph and an incidence matrix, convert the hyperedges into
        simple edges using clique-expansion, and add them into the DGLGraph in
        batches.

        Parameters
        ----------
        g : DGLGraph
            The DGLGraph to which the simple edges will be added.
        inci_matrix : np.ndarray
            Incidence matrix of shape (N, D), where N is the number of nodes (tickers) in the graph,
            and D is the number of hyperedges.
        info : str
            A string to be used as a prefix for naming the edge types.

        Notes
        -----
        Clique expansion converts each hyperedge into a clique of simple edges. Each node
        in the hyperedge is connected to every other node in the same hyperedge.

        """
        N, D = inci_matrix.shape
        data_dict = {}
        device = g.device

        # Copy existing edges from the original graph
        for etype in g.canonical_etypes:
            src, dst = g.edges(etype=etype)
            data_dict[etype] = (src, dst)

        # Process each hyperedge to add new edges
        for i in range(D):
            nodes_in_hyperedge = np.where(inci_matrix[:, i] == 1)[0]

            # Create all possible pairs (edges) within these nodes
            src_nodes, dst_nodes = np.meshgrid(
                nodes_in_hyperedge, nodes_in_hyperedge
            )
            src_nodes, dst_nodes = src_nodes.reshape(-1), dst_nodes.reshape(-1)
            mask = src_nodes != dst_nodes
            src_nodes, dst_nodes = src_nodes[mask], dst_nodes[mask]

            edge_type = (
                g.ntypes[0],
                f"{info}_{i}" if names is None else f"{info}_{names[i]}",
                g.ntypes[0],
            )

            # Add edges to data_dict
            if edge_type in data_dict:
                data_dict[edge_type] = (
                    torch.from_numpy(
                        np.concatenate([data_dict[edge_type][0], src_nodes]),
                        device=device,
                    ),
                    torch.from_numpy(
                        np.concatenate([data_dict[edge_type][1], dst_nodes]),
                        device=device,
                    ),
                )
            else:
                data_dict[edge_type] = (
                    torch.from_numpy(src_nodes).to(device),
                    torch.from_numpy(dst_nodes).to(device),
                )

        # Create a new graph with the updated data
        new_g = dgl.heterograph(data_dict)

        return new_g
