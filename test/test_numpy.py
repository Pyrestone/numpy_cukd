import numpy as np
import numpy_cukd as cukd
import time


from scipy.spatial import cKDTree

# cukd.test_numpy(np.ones(4,dtype="float32"))

num_keys = 128 * 1024 + 3
num_queries = 128 * 1024 + 1

# key_points=np.arange(num_keys)
# key_points=np.stack([key_points]*3,axis=-1)

key_points = np.random.normal(0, 50, size=(num_keys, 3))
query_points = np.random.normal(0, 50, size=(num_queries, 3))
# query_points=np.arange(num_queries)
# query_points=np.stack([query_points]*3,axis=-1)

print("running reference impl by scipy...")
ref_kdtree = cKDTree(key_points)
_, ref_indices = ref_kdtree.query(query_points[:2:-1], k=1, workers=16)
print("done")

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
out1 = cukd.query_tree(tree, query_points[:2:-1], 100.0, query, result)
t1 = time.time()
print(f"tree query took {(t1-t0)*1000:.2f} ms. (warmup)")

t0 = time.time()
out2 = cukd.query_tree(tree, query_points[:2:-1], np.inf, query, result)
t1 = time.time()
print(f"tree query took {(t1-t0)*1000:.2f} ms.")

t0 = time.time()
out = cukd.query_tree(tree, query_points[:2:-1], np.inf)
t1 = time.time()
print(f"tree query took {(t1-t0)*1000:.2f} ms. (no pre-alloc)")

del query
del result

assert (np.all(np.equal(out1, out2)))
assert (np.all(np.equal(out1, out)))
try:
    assert (np.all(np.equal(out.astype(ref_indices.dtype),ref_indices))), f"out != ref_indices:\nout:{out.shape} {out.dtype} vs ref: {ref_indices.shape} {ref_indices.dtype}"
except AssertionError as e:
    a=out.astype(ref_indices.dtype)
    b=ref_indices
    mask=a!=b
    print("Mismatched elements:")
    print(a[mask])
    print(b[mask])

    q=query_points[:2:-1]

    print("query:")
    print(q[mask])
    print("our found:")
    print(key_points[a[mask]])
    print("our distance:")
    print(np.linalg.norm(key_points[a[mask]]-q[mask],axis=-1))
    print("ref found:")
    print(key_points[b[mask]])
    print("ref distance:")
    print(np.linalg.norm(key_points[b[mask]]-q[mask],axis=-1))

    raise e

print(out.shape, out.dtype)
print(out)

del tree
