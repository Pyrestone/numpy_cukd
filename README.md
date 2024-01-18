# numpy_cukd

A blazingly fast library for nearest-neighbor queries for 3D Pointclouds.
Interface using numpy arrays, accelerated using a KD-Tree on GPU.

Based on the C++/CUDA code from https://github.com/ingowald/cudaKDTree

## Benchmarks

See test/benchmark_X.py for details

|              Method             | total time (mean ± std) |     build time    |     query time     |
|:------------------------------- |------------------------:|------------------:|-------------------:|
| scipy cKDTree                   |     132.81 ± 4.00 ms |   29.15 ± 0.74 ms |   103.65 ± 3.51 ms |
| numpy_cukd                      |       9.49 ± 0.56 ms |    7.19 ± 0.49 ms |     2.30 ± 0.17 ms |
| numpy_cukd (with buffer re-use) |   **7.59** ± 0.60 ms |**6.18** ± 0.54 ms | **1.40** ± 0.14 ms |

## Examples

### Simple Example
```python
import numpy as np
import numpy_cukd as kd

num_keys = 1024 * 128
num_queries = 1024 * 128 + 1

# define our pointclouds, shape (num_points, 3), dtype float32
points = np.random.normal(size=(num_keys, 3)).astype("float32")
query_points = np.random.normal(size=(num_queries, 3)).astype("float32")

# build a kd-tree directly from the points
tree = kd.make_tree(points)

# query the tree using our query points
neighbor_indices = kd.query_tree(tree, query_points)

# `neighbors` is an int32 array of shape (num_queries,) with the indices of the nearest point in `points` for each point in query_points

# we can index our `points` array with the returned neighbor indices to get the actual neighbor points, as an array of shape (num_queries, 3):
neighbor_points = points[neighbor_indices,:]
```
### Radius query
```python
...
# we can provide a `radius` argument to the query_tree function to limit the query radius.
neighbor_indices = kd.query_tree(tree, query_points, radius=0.5)
# if there is no point in the tree within 0.5 units around a point (e.g. query_points[i]), the resulting neighbor index (neighbor_indices[i]) value will be -1
...

```

### Re-using buffers for even faster performance
You can pre-allocate and re-use trees, query and result buffers on the GPU so that they are not re-allocated every time when running nearest neighbor searches in a loop:

```python
...
# Pre-allocate buffers
tree = kd.create_tree()
query = kd.create_query(tree)
result = kd.create_result(tree)

for i in range(200):
    points = ...
    
    # We can re-use the same tree object to hold different sets of points (even with different shape.)
    # this should re-allocate GPU memory pretty rarely, even if the shape differs every time.
    kd.build_tree(tree, points)

    for j in range(50):
        query_points = ...

        # We can query the same (built) tree multiple times (again, even with different query_points shapes).
        # `query` and `result` are just used as GPU memory buffers here. Their previous contents are overwritten every time by query_points. 
        neighbor_indices = kd.query_tree(tree, query_points, query=query, result=result)

# Clean up our GPU memory once we are done:
# Delete queries and results before their tree, as the tree object owns the cuda stream.
# Otherwise we might leak GPU memory.
del query, result
del tree

```

## Installation

### Pre-Requisites

You need CMake version 3.12 or newer
```bash
sudo apt install cmake
```
You also need a reasonably modern CUDA installation,
```bash
sudo apt install cuda-toolkit-11-8
```
as well as the python3 development files
```bash
sudo apt install python3-dev python3-pip
```

Also make sure that all three are in PATH:
```bash
which python3
which nvcc
which cmake
```

### How to Install

```bash

pip install pybind11 numpy setuptools

git clone https://github.com/Pyrestone/numpy_cukd.git && cd numpy_cukd

pip install .

```

## Credit
This package is based on the following codebases:
* [cukd](https://github.com/ingowald/cudaKDTree) by Ingo Wald and Brice Rebsamen
* [pybind11-cuda](https://github.com/pkestene/pybind11-cuda) CMake example by torstem, PWhiddy, and pkstene

If this project is helpful to your research endeavours, please cite the following works:
```bibtex
@article{waldGPUfriendlyParallelAlmost2022a,
    title        = {A {{GPU-friendly}}, Parallel, and (Almost-) in-Place Algorithm for Building Left-Balanced Kd-{{Trees}}},
    author       = {Wald, Ingo},
    year         = 2022,
    journal      = {arXiv preprint arXiv:2211.00120},
    eprint       = {2211.00120},
    archiveprefix = {arxiv}
}

@article{ueckerCanYouSeeMeNow2024,
    title        = {{{Can You See Me Now?}}, Blind Spot Estimation for Autonomous Vehicles using Scenario-Based Simulation with Random Reference Sensors},
    author       = {Uecker, Marc},
    year         = 2024,
    eprint       = {TODO}
}
```
