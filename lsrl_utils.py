import numpy as np
#import matplotlib.pyplot as plt

from math import ceil

def repeat_avg(f, num_reps):
    def rep_fn(*args, **kwargs):
        return np.mean(np.array([f(*args, **kwargs) for _ in range(num_reps)]), axis=0)
    return rep_fn
# End fn repeat_avg

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

#def plot_weights_convergence(w):
#    w_adj_dists = []
#    horizon = w.shape[0]
#    n_episodes = w.shape[1]
#
#    for h in range(horizon):
#        print('h = %d' % h)
#        w_adj_dists.append([])
#        for k in range(n_episodes-1):
#            w_adj_dists[h].append(max((np.linalg.norm(w[h][k] - w[h][k1], ord=np.inf) for k1 in range(k, n_episodes))))
#        # End for k
#    # End for h
#
#    w_adj_dists = np.array(w_adj_dists)
#    print('Done\n')
#    w_adj_dists_max = np.linalg.norm(w_adj_dists.T, ord=np.inf, axis=-1)
#    w_adj_dists_bound = np.array([1.0 if w_adj_dists_max[t] > 10.0/(np.sqrt(n_episodes)+1) else 0.0
#                                  for t in range(w_adj_dists_max.shape[0])])
#    print('\n')
#    for h in range(horizon):
#        plt.plot(np.arange(1, n_episodes), w_adj_dists[h], label=('h = %d' % h))
#    plt.plot(np.arange(1, n_episodes), w_adj_dists_bound, label='bound indicator')
#    plt.title('max subsequent w l_infty distances')
#    plt.xlabel('k')
#    plt.ylabel('||w_k - w_{k-1}||_infty')
#    plt.legend()
#    plt.show()
## End fn plot_weights_convergence
