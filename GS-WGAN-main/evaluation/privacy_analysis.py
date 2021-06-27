import sys

from autodp import rdp_acct, rdp_bank

sys.path.insert(0, '../source')

def main(config):
    delta = 1e-5
    batch_size = config['batchsize']
    prob = 1. / config['num_discriminators']  # subsampling rate
    n_steps = config['iterations']  # training iterations
    sigma = config['noise_multiplier']  # noise scale
    func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

    acct = rdp_acct.anaRDPacct()
    acct.compose_subsampled_mechanism(func, prob, coeff=n_steps * batch_size)
    epsilon = acct.get_eps(delta)
    print("Privacy cost is: epsilon={}, delta={}".format(epsilon, delta))




def get_epsilon(sigma, iter, delta=1 / 10000, batch_size=32, prob=1 / 100):
    if iter < 1:
        return 0
    func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
    acct = rdp_acct.anaRDPacct()
    acct.compose_subsampled_mechanism(func, prob, coeff=iter * batch_size)
    epsilon = acct.get_eps(delta)
    return epsilon


def compute_iter_given_epsilon(sigma, epsilon_target,
                               delta=1 / 10000, batch_size=32,
                               prob=1. / 100, max_iters=100000):
    func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

    l, r = 0, max_iters
    curr_iter = 5000
    while True:
        if curr_iter < 1:
            print("epsilon too small for given noise")
            return -1
        acct = rdp_acct.anaRDPacct()
        acct.compose_subsampled_mechanism(func, prob, coeff=curr_iter * batch_size)
        epsilon = acct.get_eps(delta)
        print(f"searching for iter, epsilon, target eps {epsilon_target}. Current combo: {curr_iter}, {epsilon}")

        if abs(epsilon - epsilon_target) < prob * 10:
            return curr_iter
        elif epsilon > epsilon_target:
            r = curr_iter
        else:
            l = curr_iter
        curr_iter = l + (r - l) // 2

