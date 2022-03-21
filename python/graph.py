"""
"""

import random
from pathlib import Path
from typing import NoReturn, Optional, Union, List, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
try:
    from tqdm.auto import tqdm, trange
except:
    from tqdm import tqdm, trange

from utils import ReprMixin


__all__ = [
    "Graph",
]


class Graph(ReprMixin):
    """
    """
    __name__ = "Graph"

    def __init__(self,
                 num_vertices:Optional[int]=None,
                 edge_set:Optional[Union[np.ndarray, list]]=None,
                 adj_mat:Optional[Union[np.ndarray, sparse.spmatrix, list]]=None) -> NoReturn:
        """
        """
        if adj_mat is None:
            assert num_vertices is not None and edge_set is not None, \
                "Either `adj_mat` is set, or `num_vertices` and `edge_set` are set."
            self._num_agents = num_vertices
            self._edge_set = np.array(edge_set, dtype=int)
            self._adj_mat = sparse.lil_matrix((self._num_agents, self._num_agents))
            self._adj_mat[:,:] = 0
            self._adj_mat[self._edge_set[:, 0], self._edge_set[:, 1]] = 1
            self._adj_mat[self._edge_set[:, 1], self._edge_set[:, 0]] = 1
        else:
            self.__init_via_adj_mat(adj_mat)
        self.__post_init()
        self.__simplify_edge_set()

    def __init_via_adj_mat(self, adj_mat:Union[np.ndarray, sparse.spmatrix, list]) -> NoReturn:
        """
        """
        if isinstance(adj_mat, sparse.spmatrix):
            self._adj_mat = adj_mat.tolil(copy=True)
        elif isinstance(adj_mat, (np.ndarray, list)):
            self._adj_mat = sparse.lil_matrix(adj_mat)
        else:
            raise TypeError(
                "adj_mat must be a sparse matrix or a numpy array or list, "
                f"but got {type(adj_mat)}"
            )
        self._num_agents = self._adj_mat.shape[0]
        self._edge_set = \
            pd.DataFrame(np.column_stack(sparse.find(self._adj_mat)[:2])).sort_values(axis=0, by=[0,1]).values

    def __post_init(self) -> NoReturn:
        """
        """
        assert self._adj_mat.shape == (self._num_agents, self._num_agents), \
            f"adj_mat must be of shape ({num_vertices}, {num_vertices}), but got {self._adj_mat.shape}."
        assert sparse.find(self._adj_mat.T != self._adj_mat)[0].size == 0, \
            "The adjacency matrix must be symmetric."
        assert sparse.find(self._adj_mat < 0)[0].size == 0, \
            "The adjacency matrix must be non-negative."
        assert self._adj_mat.diagonal().sum() == 0, \
            "The graph must not have self-loops."
        assert self._edge_set.shape[1] == 2, "edge_set must be a 2-column matrix."
        assert self._edge_set.max() < self._num_agents, \
            f"edge_set must be a matrix with entries in [0, {num_vertices}), but got {self._edge_set.max()}."
        assert self._edge_set.min() >= 0, \
            f"edge_set must be a matrix with entries in [0, {num_vertices}), but got {self._edge_set.min()}."
        assert set(self._edge_set.flatten()) == set(range(self._num_agents)), \
            "The graph should be connected."

    def __simplify_edge_set(self) -> NoReturn:
        """
        """
        self._edge_set = \
            pd.DataFrame(np.column_stack(sparse.find(self._adj_mat)[:2])).sort_values(axis=0, by=[0,1]).values
        keep_inds = np.where(self._edge_set[:, 0] < self._edge_set[:, 1])[0]
        self._edge_set = self._edge_set[keep_inds]

    def get_neighbors(self, vertex_id:int) -> List[int]:
        """
        """
        return (self._adj_mat[vertex_id, :].nonzero()[1]).tolist()
    
    @property
    def num_vertices(self) -> int:
        return self._num_agents

    @property
    def edge_set(self) -> np.ndarray:
        return self._edge_set

    @property
    def num_edges(self) -> int:
        return self._edge_set.shape[0]

    @property
    def adj_mat(self) -> sparse.spmatrix:
        return self._adj_mat

    @property
    def deg_mat(self) -> sparse.spmatrix:
        return sparse.diags(
            np.array(self._adj_mat.sum(axis=1).sum(axis=1)).flatten()
        )

    @property
    def Deg(self) -> sparse.spmatrix:
        return self.deg_mat

    @property
    def Laplacian(self) -> sparse.spmatrix:
        return self.deg_mat - self._adj_mat

    @property
    def is_weighted(self) -> bool:
        return sum([len(sparse.find(self.adj_mat == value)[0]) for value in [1]]) \
            != len(sparse.find(self.adj_mat != 0)[0])

    def extra_repr_keys(self) -> List[str]:
        return ["num_vertices", "num_edges", "is_weighted",]

    @classmethod
    def random(cls,
               num_vertices:int=10000,
               num_neighbors:Sequence[int]=(3,20),
               weight:Optional[Sequence[float]]=None) -> "Graph":
        """
        """
        # edge_set = np.array([
        #     [i,j] for i in range(num_vertices-1) \
        #         for j in random.sample(
        #             range(i+1, num_vertices),
        #             min(num_vertices-i-1, random.randint(num_neighbors[0], num_neighbors[1]))
        #         )
        # ], dtype=int)
        edge_set = np.array([], dtype=int).reshape(0,2)
        for i in trange(num_vertices-1, desc="Generating edge set", unit="vertex"):
            n_i = len(np.where(edge_set[:,1] == i)[0])
            if n_i >= num_neighbors[1]:
                continue
            low = max(0, num_neighbors[0] - n_i)
            high = max(0, num_neighbors[1] - n_i)
            edge_set = np.vstack(
                (edge_set, np.array([
                    [i,j] for j in random.sample(
                        range(i+1, num_vertices),
                        min(num_vertices-i-1, random.randint(low, high))
                    )
                ], dtype=int).reshape(-1,2))
            )
        print("Creating graph...")
        g = cls(num_vertices=num_vertices, edge_set=edge_set)
        if weight is not None:
            print("Assigning weights...")
            g._adj_mat[edge_set[:, 0], edge_set[:, 1]] = random.uniform(weight[0], weight[1])
            g._adj_mat[edge_set[:, 1], edge_set[:, 0]] = g._adj_mat[edge_set[:, 0], edge_set[:, 1]]
        return g

    def save(self, filepath:Union[str,Path]) -> str:
        """
        """
        _filepath = Path(filepath).with_suffix(".npz")
        sparse.save_npz(_filepath, self._adj_mat.tocsr())
        return str(_filepath)

    @classmethod
    def load(cls, filepath:Union[str,Path]) -> "Graph":
        """
        """
        g = cls(adj_mat=sparse.load_npz(Path(filepath).with_suffix(".npz")))
        return g

    @classmethod
    def from_file(cls, filepath:Union[str,Path]) -> "Graph":
        """
        """
        return cls.load(filepath)
