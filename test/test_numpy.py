import numpy as np
import numpy_cukd as cukd
import time


# cukd.test_numpy(np.ones(4,dtype="float32"))

num_keys = 128 * 1024 + 3
num_queries = 128 * 1024 + 1

# key_points=np.arange(num_keys)
# key_points=np.stack([key_points]*3,axis=-1)

key_points = np.random.normal(0, 50, size=(num_keys, 3))
query_points = np.random.normal(0, 50, size=(num_queries, 3))
# query_points=np.arange(num_queries)
# query_points=np.stack([query_points]*3,axis=-1)

print("key:")
print(key_points)
print("query:")
print(query_points)

tree = cukd.create_kdtree(num_keys)
t0 = time.time()
cukd.build_tree(tree, key_points)
t1 = time.time()
print(f"tree build took {(t1-t0)*1000:.2f} ms. (warmup)")

t0 = time.time()
cukd.build_tree(tree, key_points)
t1 = time.time()
print(f"tree build took {(t1-t0)*1000:.2f} ms.")

t0 = time.time()
tree2 = cukd.make_tree(key_points)
t1 = time.time()
print(f"tree build took {(t1-t0)*1000:.2f} ms. (no pre-alloc)")
del tree2


query = cukd.create_query(tree, num_queries)
result = cukd.create_result(tree, num_queries)

t0 = time.time()
out1 = cukd.query_tree(tree, query_points[::-1], 100.0, query, result)
t1 = time.time()
print(f"tree query took {(t1-t0)*1000:.2f} ms. (warmup)")

t0 = time.time()
out2 = cukd.query_tree(tree, query_points[::-1], np.inf, query, result)
t1 = time.time()
print(f"tree query took {(t1-t0)*1000:.2f} ms.")

t0 = time.time()
out = cukd.query_tree(tree, query_points[::-1], np.inf)
t1 = time.time()
print(f"tree query took {(t1-t0)*1000:.2f} ms. (no pre-alloc)")

assert (np.all(np.equal(out1, out2)))
assert (np.all(np.equal(out1, out)))
print(out.shape, out.dtype)
print(out)
del query
del result
del tree
