import numpy as np
import pandas as pd
import time
from pathlib import Path

from lsrl_utils import sherman_morrison_update, repeat_avg

from scipy.spatial.distance import pdist

from sys import getsizeof

class LSVIHistory:
    def __init__(self, int_size, threshold, H, d):
        self.int_size = int_size
        self.H = H
        self.d = d
        if threshold is None:
            self.threshold = 0.1
        else:
            self.threshold = threshold

        self.buffer = np.zeros(shape=(H, int_size, d*d))
        self.is_close = np.full(shape=(H, int_size, int_size), fill_value=False, dtype=bool)
        self.idxs = np.zeros(shape=(H,), dtype=np.int32)
        self.sizes = np.zeros(shape=(H,), dtype=np.int32)
    # End fn __init__
    
    def clear(self):
        self.buffer.fill(0.)
        self.is_close.fill(False)
        self.idxs.fill(0)
        self.sizes.fill(0)
    # End fn clear
    
    def add(self, h, M):
        self.buffer[h, self.idxs[h]] = M.flatten()
        for i in range(self.int_size):
            isc = bool(np.linalg.norm(self.buffer[h, i] - self.buffer[h, self.idxs[h]]) <= self.threshold*self.d*self.d)
            self.is_close[h, self.idxs[h], i] = self.is_close[h, i, self.idxs[h]] = isc
        self.idxs[h] += 1
        self.sizes[h] = min(self.int_size, self.sizes[h] + 1)
        if self.idxs[h] >= self.int_size:
            self.idxs[h] = 0
    # End fn add
    
    def size(self, h):
        return self.sizes[h]
    # End fn size
     
    def learning_cond(self, h):
        return not(np.all(self.is_close))
    # End fn learning_cond
    
    def __sizeof__(self):
        return 16 + self.buffer.nbytes + self.is_close.nbytes + 2*self.idxs.nbytes
    # End fn __sizeof__
# End class LSVIHistory


def lsvi_ucb_alt_learning_fixed(chunk_min, chunk_max, chunk_step, num_reps, output_folder, base_file_name, mdp, lambbda_fn, beta_fn, V_opt_zero, learn_iters_base_fn):
    ofldr = Path(output_folder)
    output_file = ofldr / ('%s_%d_%d_%d.csv' % (base_file_name, chunk_min, chunk_max-1, int(time.time())))
    
    res = []
    K_range = np.arange(chunk_min, chunk_max, chunk_step)
    for K in K_range:
        res.append(repeat_avg(lsvi_ucb_alt_learning_fixed_si, num_reps)(mdp, K, lambbda_fn, beta_fn, V_opt_zero, learn_iters_base_fn(K)))
    res = np.array(res)
    
    out_data = pd.DataFrame({'K':K_range, 'Regret':res[:,0], 'ProcessTime':res[:,1], 'SpaceUsage':res[:,2]})
    out_data.to_csv(output_file.absolute().as_posix(), encoding='utf-8', index=False)
    print('Completed output for chunk K between {%d --- %d} (file %s)' % (chunk_min, chunk_max-1, output_file.as_posix()))
    return 0
# End fn lsvi_ucb_alt_learning_fixed

# Vectorized LSVI-UCB implementation with fixed learning alternation
def lsvi_ucb_alt_learning_fixed_si(mdp, K, lambbda_fn, beta_fn, V_opt_zero, learn_iters_base):
    A = mdp.A
    d = mdp.d
    H = mdp.H

    r = learn_iters_base # take to be = K^rho for some fixed rho
    lambbda = lambbda_fn(r)
    beta = beta_fn(r)
    
    if K % 10 == 0:
        print('lsvi_ucb_alt_learning_fixed :: K = %d, r = %d' % (K, r))
    t1 = time.process_time_ns()
    TR = 0.

    # Initialize working space
    arr_index = 0

    Lambda_inv = np.zeros(shape=(H, d, d))
    phi_alg_prev = np.zeros(shape=(H, d))

    Phi_acts = np.zeros(shape=(H, A, r, d))
    cur_Phi_acts = np.zeros(shape=(H, A, d))

    Phi_alg = np.zeros(shape=(H, r, d))
    rewards = np.zeros(shape=(H, r))
    # q_vec[h] = \vec{q}_h^k for the current k (see Section 3.2, subsec "Vectorizing the algorithm",
    # page 10, eqn no 22 in notes)
    q_vec = np.zeros(shape=(H, r))

    w = np.zeros(shape=(H, d))
    
    for k in range(K):
        # Inverse Covariance update
        for h in range(0, H - 1, 1):
            if arr_index > 0:
                Lambda_inv[h] = sherman_morrison_update(Lambda_inv[h], phi_alg_prev[h], phi_alg_prev[h])
            else:
                Lambda_inv[h] = (1. / lambbda) * np.eye(d, dtype=np.float64)
            # End if
        # End for (h)

        for h in range(H - 1, -1, -1):
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
                cur_Phi_acts[h, a] = mdp.query_phi(a)
            # End for
            # Choose action from UCB-regularized greedy policy
            opt_a = np.argmax(np.clip(np.array([np.dot(cur_Phi_acts[h, a], w[h]) + beta * np.sqrt(
                np.dot(cur_Phi_acts[h, a], Lambda_inv[h] @ cur_Phi_acts[h, a])) for a in range(A)]), a_min=None,
                                      a_max=H))
            # Take action and aggregate reward
            reward, phi = mdp.take_action(opt_a)
            episode_total_reward += reward
            phi_alg_prev[h] = phi

            for a in range(A):
                Phi_acts[h, a, arr_index, :] = cur_Phi_acts[h, a]
            rewards[h, arr_index] = reward
            Phi_alg[h, arr_index, :] = phi
        # End for (h)
        TR += (V_opt_zero[mdp.s_0] - episode_total_reward)
        
        arr_index += 1
        if arr_index == r:
            # Reset learning (LSVI-UCB)
            arr_index = 0

            Lambda_inv.fill(0.)
            phi_alg_prev.fill(0.)

            Phi_acts.fill(0.)
            cur_Phi_acts.fill(0.)

            Phi_alg.fill(0.)
            rewards.fill(0.)
            # q_vec[h] = \vec{q}_h^k for the current k (see Section 3.2, subsec "Vectorizing the algorithm",
            # page 10, eqn no 22 in notes)
            q_vec.fill(0.)

            w.fill(0.)
        # End if
    # End for (k)

    apx_space_usage = Lambda_inv.nbytes + phi_alg_prev.nbytes + Phi_acts.nbytes + cur_Phi_acts.nbytes + Phi_alg.nbytes + rewards.nbytes + q_vec.nbytes + w.nbytes
    return TR, time.process_time_ns() - t1, apx_space_usage
# End fn lsvi_ucb_alt_learning_fixed_si

def lsvi_ucb_alt_learning_adaptive(chunk_min, chunk_max, chunk_step, num_reps, output_folder, base_file_name, mdp, lambbda_fn, beta_fn, V_opt_zero, space_budget_fn, max_phase_len_fn, min_phase_len, lookback_period, alt_threshold):
    ofldr = Path(output_folder)
    output_file = ofldr / ('%s_%d_%d_%d.csv' % (base_file_name, chunk_min, chunk_max-1, int(time.time())))
    
    res = []
    K_range = np.arange(chunk_min, chunk_max, chunk_step)
    for K in K_range:
        res.append(repeat_avg(lsvi_ucb_alt_learning_adaptive_si, num_reps)(mdp, K, lambbda_fn, beta_fn, V_opt_zero, space_budget_fn(K), max_phase_len_fn(K), min_phase_len, lookback_period, alt_threshold))
    res = np.array(res)
    
    out_data = pd.DataFrame({'K':K_range, 'Regret':res[:,0], 'ProcessTime':res[:,1], 'SpaceUsage':res[:,2]})
    out_data.to_csv(output_file.absolute().as_posix(), encoding='utf-8', index=False)
    print('Completed output for chunk K between {%d --- %d} (file %s)' % (chunk_min, chunk_max-1, output_file.as_posix()))
    return 0
# End fn lsvi_ucb_alt_learning_adaptive

# Vectorized LSVI-UCB implementation with adaptive learning alternation
def lsvi_ucb_alt_learning_adaptive_si(mdp, K, lambbda_fn, beta_fn, V_opt_zero, space_budget, max_phase_len, min_phase_len, lookback_period, alt_threshold):
    A = mdp.A
    d = mdp.d
    H = mdp.H
    
    r = space_budget
    lambbda = lambbda_fn(max_phase_len)
    beta = beta_fn(max_phase_len)
    
    
    if K % 10 == 0:
        print('lsvi_ucb_alt_learning_adaptive :: K = %d, r = %d' % (K, r))
    
    t1 = time.process_time_ns()
    TR = 0.
    history = LSVIHistory(int_size=lookback_period, threshold=alt_threshold, H=H, d=d)
    
    arr_index = np.zeros(shape=(H,), dtype=np.int32)
    is_learning = np.full(shape=(H,), fill_value=True, dtype=bool)
    learning_finished = np.full(shape=(H,), fill_value=False, dtype=bool)
    cur_learn_iter_ctr = np.zeros(shape=(H,), dtype=np.int32)
    total_iter_ctr = np.zeros(shape=(H,), dtype=np.int32)
    
    Lambda_inv = np.zeros(shape=(H, d, d)) # Only including the learning iterations
    phi_alg_prev = np.zeros(shape=(H, d))
    temp_lambda_inv = np.zeros(shape=(H, d, d)) # Including all iterations

    Phi_acts = np.zeros(shape=(H, A, r, d))
    cur_Phi_acts = np.zeros(shape=(H, A, d))

    Phi_alg = np.zeros(shape=(H, r, d))
    rewards = np.zeros(shape=(H, r))
    # q_vec[h] = \vec{q}_h^k for the current k (see Section 3.2, subsec "Vectorizing the algorithm",
    # page 10, eqn no 22 in notes)
    q_vec = np.zeros(shape=(H, r))

    w = np.zeros(shape=(H, d))
    
    for k in range(K):
        # Inverse Covariance update
        for h in range(0, H - 1, 1):
            total_iter_ctr[h] += 1
            if arr_index[h] > 0:
                temp_lambda_inv[h] = sherman_morrison_update(temp_lambda_inv[h], phi_alg_prev[h], phi_alg_prev[h])
                if is_learning[h]:
                    Lambda_inv[h] = sherman_morrison_update(Lambda_inv[h], phi_alg_prev[h], phi_alg_prev[h]) 
            else:
                temp_lambda_inv[h] = (1. / lambbda) * np.eye(d, dtype=np.float64)
                Lambda_inv[h] = (1. / lambbda) * np.eye(d, dtype=np.float64)
            # End if
            history.add(h, temp_lambda_inv[h])
        # End for (h)
        
        # Update is_learning and learning_finished
        if is_learning[h]:
            cur_learn_iter_ctr[h] += 1
            if cur_learn_iter_ctr[h] > min_phase_len:
                if not history.learning_cond(h):
                    is_learning[h] = False
        else:
            if history.learning_cond(h) and not learning_finished[h]:
                cur_learn_iter_ctr[h] = 0
                is_learning[h] = True
        # End if
        
        # Policy formulation
        for h in range(H - 1, -1, -1):
            if is_learning[h]:
                if h < H - 1:
                    q_vec[h] = np.maximum.reduce(np.clip(np.array([np.dot(Phi_acts[h + 1, a], w[h + 1]) + beta * np.sqrt(
                        np.diagonal(Phi_acts[h + 1, a] @ Lambda_inv[h] @ Phi_acts[h + 1, a].T))
                                              for a in range(A)]), a_min=None, a_max=H))
                w[h] = np.linalg.multi_dot([Lambda_inv[h], Phi_alg[h].T, rewards[h] + q_vec[h]])
            # End if is_learning[h]
        # End for (h)

        # Learning
        episode_total_reward = 0.
        for h in range(H):
            # Query and store phis for all actions with current state before updating state with mdp.take_action()
            for a in range(A):
                cur_Phi_acts[h, a] = mdp.query_phi(a)
            # End for
            # Choose action from UCB-regularized greedy policy
            opt_a = np.argmax(np.clip(np.array([np.dot(cur_Phi_acts[h, a], w[h]) + beta * np.sqrt(
                np.dot(cur_Phi_acts[h, a], Lambda_inv[h] @ cur_Phi_acts[h, a])) for a in range(A)]), a_min=None,
                                      a_max=H))
            # Take action and aggregate reward
            reward, phi = mdp.take_action(opt_a)
            episode_total_reward += reward
            phi_alg_prev[h] = phi

            if is_learning[h]:
                for a in range(A):
                    Phi_acts[h, a, arr_index[h], :] = cur_Phi_acts[h, a]
                rewards[h, arr_index[h]] = reward
                Phi_alg[h, arr_index[h], :] = phi
                if arr_index[h] < r-1:
                    arr_index[h] += 1
                else: # Spent the learning space budget
                    is_learning[h] = False
                    learning_finished[h] = True
                # End if
            # End if is_learning[h]
            
            if (learning_finished[h] or total_iter_ctr[h] == max_phase_len):
                # Reset working space
                arr_index.fill(0)
                is_learning.fill(True)
                learning_finished.fill(False)
                cur_learn_iter_ctr.fill(0)
                total_iter_ctr.fill(0)
                
                Lambda_inv.fill(0.)
                phi_alg_prev.fill(0.)
                temp_lambda_inv.fill(0.)

                Phi_acts.fill(0.)
                cur_Phi_acts.fill(0.)

                Phi_alg.fill(0.)
                rewards.fill(0.)
                q_vec.fill(0.)

                w.fill(0.)
                history.clear()
            # End if
        # End for (h)
        TR += (V_opt_zero[mdp.s_0] - episode_total_reward)
    # End for (k)

    apx_space_usage = history.__sizeof__() + 18*H + Lambda_inv.nbytes + temp_lambda_inv.nbytes + phi_alg_prev.nbytes + Phi_acts.nbytes + cur_Phi_acts.nbytes + Phi_alg.nbytes + rewards.nbytes + q_vec.nbytes + w.nbytes
    return TR, time.process_time_ns() - t1, apx_space_usage
# End fn lsvi_ucb_alt_learning_adaptive
