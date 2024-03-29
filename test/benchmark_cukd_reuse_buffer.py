import numpy as np
import numpy_cukd as kd
import time


num_keys = 1024 * 128
num_queries = 1024 * 128 + 1

tree = kd.create_tree()
query = kd.create_query(tree)
result = kd.create_result(tree)

times = []
build_times = []
query_times = []
for i in range(20):
    points = np.random.normal(size=(num_keys, 3)).astype("float32")
    query_points = np.random.normal(size=(num_queries, 3)).astype("float32")
    t0 = time.time()
    # Build KD-Tree for the 'points' array
    kd.build_tree(tree, points)
    t1 = time.time()
    # Query nearest neighbors for each point in 'query_points'
    indices = kd.query_tree(tree, query_points, query=query, result=result)
    t2 = time.time()
    if (i > 0):
        times.append(t2 - t0)
        build_times.append(t1 - t0)
        query_times.append(t2 - t1)
print(f"total time (mean ± std): {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")
print(f"build time (mean ± std): {np.mean(build_times)*1000:.2f} ± {np.std(build_times)*1000:.2f} ms")
print(f"query time (mean ± std): {np.mean(query_times)*1000:.2f} ± {np.std(query_times)*1000:.2f} ms")

del result, query
del tree
