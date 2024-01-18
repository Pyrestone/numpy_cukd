from . import numpy_cukd as kd
import numpy as np
from typing import Optional
from numpy.typing import NDArray

__all__ = ["create_kdtree", "create_tree", "make_tree", "build_tree", "create_query", "create_result", "query_tree"]

# These classes are only here for type hints


class KDTree3D:
    pass


class KDTree3DQuery:
    pass


class KDTree3DQueryResult:
    pass


def create_kdtree(size: int = 0) -> KDTree3D:
    return kd.create_kdtree(size)


create_tree = create_kdtree


def make_tree(points: NDArray[np.float32]) -> KDTree3D:
    return kd.make_tree(points)


def build_tree(tree: KDTree3D, points: NDArray[np.float32]) -> None:
    kd.build_tree(tree, points)


def create_query(tree: KDTree3D, size: int = 0) -> KDTree3DQuery:
    return kd.create_query(tree, size)


def create_result(tree: KDTree3D, size: int = 0) -> KDTree3DQueryResult:
    return kd.create_result(tree, size)


def query_tree(tree: KDTree3D, points: NDArray[np.float32], radius: float = np.inf, query: Optional[KDTree3DQuery] = None, result: Optional[KDTree3DQueryResult] = None) -> NDArray[np.int32]:
    return kd.query_tree(tree, points, radius, query, result)
