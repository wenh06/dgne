"""
"""

from typing import NoReturn, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse

from utils import ReprMixin


class Graph(ReprMixin):
    """
    """
    __name__ = "Graph"

    def __init__(self,
                 num_agents:Optional[int]=None,
                 edge_set:Optional[Union[np.ndarray, list]]=None,
                 adj_mat:Optional[Union[np.ndarray, sparse.spmatrix, list]]=None) -> NoReturn:
        """
        """
        if adj_mat is None:
            assert num_agents is not None and edge_set is None, \
                "Either `adj_mat` is set, or `num_agents` and `edge_set` are set."
            self._num_agents = num_agents
            self._edge_set = np.array(edge_set, dtype=int)
            self._adj_mat = sparse.lil_matrix((self._num_agents, self._num_agents))
            self._adj_mat[self._edge_set[:, 0], self._edge_set[:, 1]] = 1
        self.__init_via_adj_mat(adj_mat)

    def __post_init(self) -> NoReturn:
        """
        """
        assert self._adj_mat.shape == (self._num_agents, self._num_agents), \
            f"adj_mat must be of shape ({num_agents}, {num_agents}), but got {self._adj_mat.shape}."
        assert sparse.find(self._adj_mat.T != self._adj_mat)[0].size == 0, \
            "The adjacency matrix must be symmetric."
        assert sparse.find(self._adj_mat < 0)[0].size == 0, \
            "The adjacency matrix must be non-negative."
        assert self._adj_mat.diagonal().sum() == 0, \
            "The graph must not have self-loops."
        assert self._edge_set.shape[1] == 2, "edge_set must be a 2-column matrix."
        assert self._edge_set.max() < self._num_agents, \
            f"edge_set must be a matrix with entries in [0, {num_agents}), but got {self._edge_set.max()}."
        assert self._edge_set.min() >= 0, \
            f"edge_set must be a matrix with entries in [0, {num_agents}), but got {self._edge_set.min()}."
        assert set(self._edge_set.flatten()) == set(range(self._num_agents)), \
            "The graph should be connected."

    def __init_via_adj_mat(self, adj_mat:Union[np.ndarray, sparse.spmatrix, list]) -> NoReturn:
        """
        """
        if isinstance(adj_mat, sparse.spmatrix):
            self._adj_mat = adj_mat
        elif isinstance(adj_mat, [np.ndarray, list]):
            self._adj_mat = sparse.lil_matrix(adj_mat)
        else:
            raise TypeError(
                "edge_set must be a sparse matrix or a numpy array or list, "
                f"but got {type(edge_set)}"
            )
        self._num_agents = self._adj_mat.shape[0]
        self._edge_set = \
            pd.DataFrame(np.column_stack(sparse.find(self._adj_mat)[:2])).sort_values(axis=0, by=[0,1]).values
    
    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def edge_set(self) -> np.ndarray:
        return self._edge_set

    @property
    def adj_mat(self) -> sparse.spmatrix:
        return self._adj_mat

    @property
    def deg_mat(self) -> sparse.spmatrix:
        return sparse.diags(
            np.array(self._adj_mat.sum(axis=1).sum(axis=1)).flatten()
        )
