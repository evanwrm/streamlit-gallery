from abc import ABC, abstractmethod
from typing import Type, Union
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN


PARTITION_CLUSTERING = "Partition"
DENSITY_CLUSTERING = "Density"
CLUSTERING_METHODS = [PARTITION_CLUSTERING, DENSITY_CLUSTERING]


@dataclass
class ClusteringAlgorithm(ABC):
    name: str
    algorithm: Union[Type[KMeans], Type[HDBSCAN]]
    method: Union[PARTITION_CLUSTERING, DENSITY_CLUSTERING]

    @abstractmethod
    def cluster_trees(self, tree_data, **kwargs):
        pass


@dataclass
class PartitionClustering(ClusteringAlgorithm):
    method: str = PARTITION_CLUSTERING

    def cluster_trees(self, tree_data, n_clusters=3):
        km = self.algorithm(
            n_clusters=int(n_clusters), init="k-means++", random_state=42
        ).fit(tree_data)
        return km.labels_


@dataclass
class DensityClustering(ClusteringAlgorithm):
    method: str = DENSITY_CLUSTERING

    def cluster_trees(self, tree_data, eps=0.5, metric="euclidean"):
        if metric == "haversine":
            tree_data = np.radians(tree_data)
        ds = self.algorithm(metric=metric).fit(tree_data)
        return ds.labels_


PARTITION_METHOD = PartitionClustering(name="K-means", algorithm=KMeans)
DENSITY_METHOD = DensityClustering(name="DBScan", algorithm=HDBSCAN)
