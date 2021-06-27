import os
import pickle
import sys

import numpy as np

sys.path.append("../evaluation")
from config import parse_arguments, save_config
from privacy_analysis import compute_iter_given_epsilon
import main


def run_noise_experiment(args):
    noise_vals = np.arange(1.0, 2.3, 0.25)
    epsilons = [0.1, 1.0, 10.0, 100.0]
    save_dir = args.save_dir
    for noise in noise_vals:
        save_iters = [compute_iter_given_epsilon(noise, x, prob=1/100) for x in epsilons]

        setattr(args, 'noise_multiplier', noise)
        setattr(args, 'save_iterations', save_iters)
        setattr(args, 'iterations', save_iters[-1])

        for run in range(3):
            print(f"run {run} with noise: {noise}, save_iters {save_iters}")
            setattr(args, 'save_dir', os.path.join(save_dir, f"noise{noise}_run{run}"))
            setattr(args, 'args.random_seed', run)

            os.makedirs(args.save_dir, exist_ok=True)
            config = vars(args)
            pickle.dump(config, open(os.path.join(args.save_dir, 'params.pkl'), 'wb'), protocol=2)
            with open(os.path.join(args.save_dir, 'params.txt'), 'w') as f:
                for k, v in config.items():
                    kv_str = k + ':' + str(v) + '\n'
                    print(kv_str)
                    f.writelines(kv_str)

            main.main(args)


if __name__ == '__main__':
    args = parse_arguments()
    save_config(args)
    run_noise_experiment(args)
