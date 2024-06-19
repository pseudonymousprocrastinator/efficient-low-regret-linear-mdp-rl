import numpy as np
from numpy.random import default_rng
import math

from lsvi_ucb_base import lsvi_ucb_learning
from lsvi_sketching import lsvi_ucb_sketched_learning
from lsvi_learning_alt import lsvi_ucb_alt_learning_fixed, lsvi_ucb_alt_learning_adaptive

from lsrl_utils import generate_intervals, repeat_avg

from argparse import ArgumentParser
import pickle
from pathlib import Path

from joblib import Parallel, delayed
import pandas as pd

from linear_mdp import TabLinearMDP
from linearize_interface import NeuralLinearMDP

random_seed = 1542973613
rng = default_rng(random_seed)


def main():
    K_step = 20
    
    parser = ArgumentParser(prog="experiment_base")
    
    parser.add_argument("alg", choices=["basic", "sketch", "alt_fixed", "alt_adaptive"],)
    parser.add_argument("K_max", type=int, help=("The maximum number of episodes (will try #episodes from K_min to K_max in steps of %d)" % (K_step)))
    parser.add_argument("mdp_file", help="The name of the file to read the MDP data from")
    # Optional arguments - general
    parser.add_argument("-o", "--output-folder", default="./output/", help="Folder to store the experiment results (CSV format).")
    parser.add_argument("-n", "--num-jobs", type=int, default=16, help="The number of parallel jobs (processes) to use")
    parser.add_argument("--k-min", type=int, default=100, help="The K value to start with.")
    parser.add_argument("--num-reps", type=int, default=5, help="The number of repetitions for each K")
    parser.add_argument("--chunk-size", type=int, default=10, help="The number of K values in each chunk")
    # Alg-specific arguments
    parser.add_argument("--sketch-dim", type=int, default=100, help="The projection-dimension of the sketch transform (if alg=sketch)")
    parser.add_argument("--learn-iters-base-exp", type=float, default=0.5, help="The exponent e of int(2*K^e)+1 --- the base number of iterations used with alg=alt_fixed.")
    parser.add_argument("--lookback-period", type=int, default=10, help="The number of steps to look-back for the alternation condition (if alg=alt_adaptive)")
    parser.add_argument("--alt-threshold", type=float, default=0.1, help="The alternation condition threshold (if alg=alt_adaptive) --- actual threshold will be this param * d^2")
    parser.add_argument("--learn-iters-budget-exp", type=float, default=0.5, help="The exponent e of int(4*K^e)+1 --- the total learning iterations budget if alg=alt_adaptive")
    
    args = parser.parse_args()
    K_min = args.k_min
    num_reps = args.num_reps
    chunk_size = args.chunk_size
    print('Arguments:', args)
    print('Params:', 'K_min = %d, K_step = %d, chunk_size = %d, num_reps = %d' % (K_min, K_step, chunk_size, num_reps))
    
    assert(args.K_max >= 200)
    assert(args.num_jobs >= 1)
    assert(args.sketch_dim >= 1)
    assert(args.lookback_period >= 1)
    assert(args.alt_threshold > 0.)
    
    chunk_intervals = generate_intervals(K_min, args.K_max + 1, K_step, chunk_size)
    print('Output chunk intervals:', ' '.join(['[%d, %d, %d]' % (c[0], c[1], K_step) for c in chunk_intervals]))
    
    # Load MDP data from file
    mdp_file_path = Path(args.mdp_file)
    with mdp_file_path.open(mode='rb') as mdp_file_obj:
        data = pickle.load(mdp_file_obj) # format {'mdp': mdp, 'pi_opt': pi_opt, 'V_opt':V_opt}
    mdp = data['mdp']
    V_opt = data['V_opt']
    #print('MDP: S = %d, A = %d, d = %d, H = %d' % (mdp.S, mdp.A, mdp.d, mdp.H))
    
    # Create output folder if necessary (mkdir -p equivalent)
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    
    lambbda_fn = lambda K : 1.
    beta_fn = lambda K : 0.05 * mdp.d * mdp.H * math.sqrt(math.log(3.*mdp.d*mdp.H*K))
    
    if args.alg == 'basic':
        print('Basic LSVI-UCB\n-------------------')
        res = Parallel(n_jobs=args.num_jobs)(delayed(lsvi_ucb_learning)(
                                                               chunk[0], chunk[1], K_step, num_reps,
                                                               args.output_folder, 'output_chunk',
                                                               mdp,
                                                               lambbda_fn,
                                                               beta_fn,
                                                               V_opt[0])
                                    for chunk in chunk_intervals)
    elif args.alg == 'sketch':
        print('Sketched LSVI-UCB\n-------------------')
        res = Parallel(n_jobs=args.num_jobs)(delayed(lsvi_ucb_sketched_learning)(
                                                               chunk[0], chunk[1], K_step, num_reps,
                                                               args.output_folder, 'output_chunk',
                                                               mdp,
                                                               args.sketch_dim,
                                                               lambbda_fn,
                                                               beta_fn,
                                                               V_opt[0])
                                    for chunk in chunk_intervals)
    elif args.alg == 'alt_fixed':
        print('Space-saving LSVI-UCB (fixed)\n------------------------------')
        learn_iters_base_fn = lambda K: int(2.*(K**args.learn_iters_base_exp))+1
        total_learn_iters_fn = lambda K : 2*learn_iters_base_fn(K) + mdp.H # Works with scale_factor = 0.5
        res = Parallel(n_jobs=args.num_jobs)(delayed(lsvi_ucb_alt_learning_fixed)(
                                                               chunk[0], chunk[1], K_step, num_reps,
                                                               args.output_folder, 'output_chunk',
                                                               mdp,
                                                               lambbda_fn,
                                                               beta_fn,
                                                               V_opt[0],
                                                               learn_iters_base_fn=learn_iters_base_fn,
                                                               scale_factor=0.5,
                                                               total_learn_iters_fn=total_learn_iters_fn
                                                            )
                                    for chunk in chunk_intervals)
    elif args.alg == 'alt_adaptive':
        print('Space-saving LSVI-UCB (adaptive)\n----------------------------------')
        learn_iters_budget_fn = lambda K: int(4.*(K**args.learn_iters_budget_exp))+1
        res = Parallel(n_jobs=args.num_jobs)(delayed(lsvi_ucb_alt_learning_adaptive)(
                                                               chunk[0], chunk[1], K_step, num_reps,
                                                               args.output_folder, 'output_chunk',
                                                               mdp,
                                                               lambbda_fn,
                                                               beta_fn,
                                                               V_opt[0],
                                                               total_learn_iters_fn=learn_iters_budget_fn,
                                                               min_phase_len=20,
                                                               lookback_period=args.lookback_period,
                                                               alt_threshold=args.alt_threshold
                                                            )
                                    for chunk in chunk_intervals)
    # End if
    #res = np.array(res)
    #K_range = np.arange(K_min, args.K_max + 1, K_step)
    #out_data = pd.DataFrame({'K':K_range, 'Regret':res[:,0], 'ProcessTime':res[:,1], 'SpaceUsage':res[:,2]})
    #out_data.to_csv(args.output_file, encoding='utf-8', index=False)
# End fn main

if __name__ == '__main__':
    main()
