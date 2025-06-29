import numpy as np
import pandas as pd
import time
from pathlib import Path

from lsrl_utils import sherman_morrison_update, repeat_cat

def lsvi_ucb_learning(chunk_min, chunk_max, chunk_step, num_reps, output_folder, base_file_name, mdp, lambbda_fn, beta_fn, V_opt_zero, random_state):
    rng_new = np.random.default_rng(random_state)
    mdp.set_rng(rng_new)
    ofldr = Path(output_folder)
    output_file = ofldr / ('%s_%d_%d_%d.csv' % (base_file_name, chunk_min, chunk_max-1, int(time.time())))
    
    res = []
    K_range = np.arange(chunk_min, chunk_max, chunk_step)
    for K in K_range:
        res.append(repeat_cat(lsvi_ucb_learning_si, num_reps)(mdp, K, lambbda_fn(K), beta_fn(K), V_opt_zero))
    res = np.array(res)
    res = np.reshape(res, newshape=(-1, 3))
    index = pd.MultiIndex.from_product([K_range, range(1, num_reps+1)], names=["K", "I"])
    out_data = pd.DataFrame(res, columns=['Regret', 'ProcessTime', 'SpaceUsage'], index=index)
    out_data.to_csv(output_file.absolute().as_posix(), encoding='utf-8', index=True)
    print('Completed output for chunk K between {%d --- %d} (file %s)' % (chunk_min, chunk_max-1, output_file.as_posix()))
    return 0
# End fn lsvi_ucb_learning

# Vectorized LSVI-UCB implementation (for ease of incorporating sketching)
def lsvi_ucb_learning_si(mdp, K, lambbda, beta, V_opt_zero):
    A = mdp.A
    d = mdp.d
    H = mdp.H
    
    if K % 10 == 0:
        print('lsvi_ucb_learning :: K = %d' % K)
    t1 = time.process_time_ns()
    
    Lambda_inv = np.zeros(shape=(H, d, d))
    phi_alg_prev = np.zeros(shape=(H, d))
    
    Phi_acts = np.zeros(shape=(H, A, K, d))
    Phi_alg  = np.zeros(shape=(H, K, d))
    rewards  = np.zeros(shape=(H, K))
    # q_vec[h] = \vec{q}_h^k for the current k (see Section 3.2, subsec "Vectorizing the algorithm",
    # page 10, eqn no 22 in notes)
    q_vec = np.zeros(shape=(H, K))
    
    w = np.zeros(shape=(H, d))
    TR = 0.
    
    for k in range(K):
        # Policy formulation
        for h in range(H - 1, -1, -1):
            if k > 0:
                Lambda_inv[h] = sherman_morrison_update(Lambda_inv[h], phi_alg_prev[h], phi_alg_prev[h])
            else:
                Lambda_inv[h] = (1. / lambbda) * np.eye(d, dtype=np.float64)
            # End if
            if h < H - 1:
                q_vec[h] = np.maximum.reduce(np.clip(np.array([np.dot(Phi_acts[h + 1, a], w[h + 1]) + beta * np.sqrt(
                    np.diagonal(Phi_acts[h + 1, a] @ Lambda_inv[h] @ Phi_acts[h + 1, a].T))
                                        for a in range(A)]), a_min=None, a_max=H))
            w[h] = np.linalg.multi_dot([Lambda_inv[h], Phi_alg[h].T, rewards[h] + q_vec[h]])
        # End for (h)
        
        # Learning
        episode_total_reward = 0.
        for h in range(H):
            # Query and store phis for all actions with current state before updating state with mdp.take_action()
            for a in range(A):
                Phi_acts[h, a, k, :] = mdp.query_phi(a)
            # End for
            
            opt_a = np.argmax(np.clip(np.array([np.dot(Phi_acts[h, a, k, :], w[h]) + beta * np.sqrt(
                np.dot(Phi_acts[h, a, k, :], Lambda_inv[h] @ Phi_acts[h, a, k, :])) for a in range(A)]), a_min=None,
                                      a_max=H))
            reward, phi = mdp.take_action(opt_a)
            episode_total_reward += reward
            rewards[h, k] = reward
            Phi_alg[h, k, :] = phi
            phi_alg_prev[h] = phi
        # End for (h)
        TR += (V_opt_zero[mdp.s_0] - episode_total_reward)
    # End for (k)
    
    apx_space_usage = Lambda_inv.nbytes + phi_alg_prev.nbytes + Phi_acts.nbytes + Phi_alg.nbytes + rewards.nbytes + q_vec.nbytes + w.nbytes
    return TR, time.process_time_ns() - t1, apx_space_usage
# End fn lsvi_ucb_learning_si
