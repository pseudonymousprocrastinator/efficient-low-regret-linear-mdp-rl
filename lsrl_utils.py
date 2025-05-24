import numpy as np
from math import ceil

def repeat_avg(f, num_reps):
    def rep_fn(*args, **kwargs):
        return np.mean(np.array([f(*args, **kwargs) for _ in range(num_reps)]), axis=0)
    return rep_fn
# End fn repeat_avg

def repeat_cat(f, num_reps):
    def rep_fn(*args, **kwargs):
        return np.array([list(f(*args, **kwargs)) for ctr in range(num_reps)])
    return rep_fn
# End fn repeat

def generate_intervals(i_min, i_max, i_step, chunk_size):
    ran = range(i_min, i_max, i_step)
    N = len(ran)
    int_list = []
    for i in range(0, ceil(1.*N/chunk_size)):
        int_list.append((ran[i*chunk_size], ran[min(N, (i+1)*chunk_size)-1]+1))
    return int_list
# End fn generate_intervals

# Compute (A + uv^T)^{-1} from A^{-1}, u, and v, using the Sherman-Morrison formula
# (https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula)
def sherman_morrison_update(a_inv, u, v):
    u, v = u.reshape(-1, 1), v.reshape(-1, 1)  # Reshape to column vectors
    norm = 1. + np.linalg.multi_dot([v.T, a_inv, u])
    res = a_inv - np.linalg.multi_dot([a_inv, u, v.T, a_inv]) / norm

    return res
# End fn sherman_morrison_update

def repeat_and_aggregate(f, n, aggregator=lambda x: x):
    def rep_fn(*args, **kwargs):
        results = np.array([f(*args, **kwargs) for _ in range(n)])
        return aggregator(results)
    return rep_fn
# End fn repeat_and_aggregate
