"""
"""

from typing import Union, Optional, Tuple, Sequence, NoReturn, Any

import numpy as np
import pandas as pd
import networkx as nx
from networkx import algorithms as NXA
import nxmetis  # noqa: F401
from scipy.sparse import spmatrix, load_npz


__all__ = [
    "BipartiteGraph",
]


class BipartiteGraph(nx.Graph):
    """ """

    def __init__(
        self,
        incoming_graph_data: Optional[Any] = None,
        row_nodes: Optional[Sequence[str]] = None,
        col_nodes: Optional[Sequence[str]] = None,
        edges: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> NoReturn:
        """finished, checked,

        Parameters:
        -----------
        incoming_graph_data: input graph, optional,
        row_nodes: sequence of str, optional,
            (names) of the row nodes
        col_nodes: sequence of str, optional,
            (names) of the column nodes
        edges: sequence of tuples of str, optional,
            edges of the graph
        """
        super().__init__(incoming_graph_data)
        if row_nodes is not None:
            self.add_nodes_from(row_nodes, bipartite="row")
        if col_nodes is not None:
            self.add_nodes_from(col_nodes, bipartite="col")
        if edges is not None:
            self.add_edges_from(edges)
        assert NXA.bipartite.is_bipartite(self)
        if not self.empty:
            assert all(
                [d["bipartite"] in ["row", "col"] for _, d in self.nodes(data=True)]
            )

    @property
    def row_nodes(self) -> list:
        """ """
        rn = [n for n, d in self.nodes(data=True) if d["bipartite"] == "row"]
        return rn

    @property
    def col_nodes(self) -> list:
        """ """
        cn = [n for n, d in self.nodes(data=True) if d["bipartite"] == "col"]
        return cn

    @property
    def n_edges(self) -> int:
        """ """
        ne = len(self.edges)
        return ne

    @property
    def n_nonzeros(self) -> int:
        """ """
        return self.n_edges

    @property
    def n_connected_components(self) -> int:
        """ """
        ncc = len(list(NXA.components.connected_components(self)))
        return ncc

    @property
    def connected_components(self) -> list:
        """ """
        cc = [self.subgraph(item) for item in NXA.components.connected_components(self)]
        return cc

    @property
    def sorted_connected_components(self) -> list:
        """ """
        cc = self.connected_components
        cc = sorted(cc, key=lambda item: item.size, reverse=True)
        return cc

    @property
    def shape(self) -> Tuple[int, int]:
        """ """
        sp = (len(self.row_nodes), len(self.col_nodes))
        return sp

    @property
    def size(self) -> int:
        """ """
        shape = self.shape
        sz = shape[0] * shape[1]
        return sz

    @property
    def empty(self) -> bool:
        """ """
        return len(self.nodes) == 0

    @property
    def density(self) -> float:
        """ """
        d = self.n_nonzeros / self.size
        return d

    @property
    def density_strict(self) -> float:
        """ """
        cc = self.connected_components
        cc_areas = [item.size for item in cc]
        cc_n_edges = [item.n_nonzeros for item in cc]
        d = sum(cc_n_edges) / sum(cc_areas)
        return d

    @staticmethod
    def from_array(
        arr: Union[Sequence, np.ndarray],
        row_names: Optional[Sequence[str]] = None,
        col_names: Optional[Sequence[str]] = None,
    ) -> "BipartiteGraph":
        """finished, checked,

        Parameters:
        -----------
        arr: array_like,
        row_names: sequence of str, optional,
        col_names: sequence of str, optional,

        Returns:
        --------
        bg: BipartiteGraph,
        """
        _arr = np.array(arr)
        assert _arr.ndim == 2
        nrows, ncols = np.array(_arr).shape
        rn = [f"row_{idx}" for idx in range(nrows)] if row_names is None else row_names
        cn = [f"col_{idx}" for idx in range(ncols)] if col_names is None else col_names
        assert nrows == len(rn) and ncols == len(cn)
        nz_row, nz_col = np.where(_arr != 0)
        nz_row = [rn[idx] for idx in nz_row]
        nz_col = [cn[idx] for idx in nz_col]

        bg = BipartiteGraph(
            row_nodes=sorted(list(set(nz_row))),
            col_nodes=sorted(list(set(nz_col))),
            edges=[(r, c) for r, c in zip(nz_row, nz_col)],
        )
        return bg

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "BipartiteGraph":
        """finished, checked,

        Parameters:
        -----------
        df: DataFrame,
            from `df` to construct a `BipartiteGraph`

        Returns:
        --------
        bg: BipartiteGraph,
        """
        row_names = [f"row_{item}" for item in df.index]
        col_names = [f"col_{item}" for item in df.columns]
        bg = BipartiteGraph.from_array(
            arr=df.values,
            row_names=row_names,
            col_names=col_names,
        )
        return bg

    @staticmethod
    def from_sparse(ssm: Union[spmatrix, str]) -> "BipartiteGraph":
        """finished, checked,

        Parameters:
        -----------
        ssm: spmatrix or str,
            a scipy sparse matrix (can be of format bsr, coo, csc, csr, dia, dok, lil);
            or the path to a .npz file that stores a scipy sparse matrix

        Returns:
        --------
        bg: BipartiteGraph,
        """
        if isinstance(ssm, str):
            _ssm = load_npz(ssm)
        else:
            _ssm = ssm
        nz_row, nz_col = _ssm.nonzero()
        nz_row = [f"row_{idx}" for idx in nz_row]
        nz_col = [f"col_{idx}" for idx in nz_col]
        bg = BipartiteGraph(
            row_nodes=sorted(list(set(nz_row))),
            col_nodes=sorted(list(set(nz_col))),
            edges=[(r, c) for r, c in zip(nz_row, nz_col)],
        )
        return bg

    def to_dataframe(
        self, rows: Optional[Sequence[str]] = None, cols: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """finished, checked,

        Parameters:
        -----------
        rows: sequence of str, optional,
            names of the rows (in this specific ordering) to form the output `DataFrame`,
            if is None, `self.row_nodes` will be used
        cols: sequence of str, optional,
            names of the columns (in this specific ordering) to form the output `DataFrame`,
            if is None, `self.col_nodes` will be used

        Returns:
        --------
        df: DataFrame,
        """
        df = pd.DataFrame(np.zeros(shape=self.shape)).astype(int)
        df.index = self.row_nodes
        df.columns = self.col_nodes
        for r in df.index:
            for c in df.columns:
                if (r, c) in self.edges:
                    df.at[r, c] = 1
        df = df.loc[(rows or df.index), (cols or df.columns)]
        df.columns = df.columns.map(lambda s: s.replace("col_", ""))
        df.index = df.index.map(lambda s: s.replace("row_", ""))
        return df
