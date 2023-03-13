import numpy as np
import argparse as ap
from sketch import frequencies

class SketchWrapper:

    def __init__(self, x, m, options):
        # signal dim
        self.d = x.shape[1]
        self.n = 0
        # sketch dim
        self.m = m
        self.options = options
        self.sk_emp = np.zeros(2*m)
        # draw frequencies
        idx_list = np.random.permutation(x.shape[0])
        idx = idx_list[:min(x.shape[0], self.options.smalln)]
        x_0 = x[idx, :]
        # self.mean_var = 2e10;
        self.mean_var = frequencies.estimate_mean_var(self, x_0)
        # self.freq_gmm = mixture_gmm.MixtureGmm(np.zeros([1, self.d]), self.mean_var * np.eye(self.d), 1)
        #
        self.W = frequencies.draw_freq(self.d, self.m, self.mean_var)

    def set_sketch(self, x):
        self.sk_emp = frequencies.emp_sketch(x, self.W)
        self.n = x.shape[0]

    def update_sketch(self, x):
        n1 = self.n
        n2 = x.shape[0]
        self.n = n1 + n2
        self.sk_emp = (n2 * frequencies.emp_sketch(x, self.W) + n1 * self.sk_emp) / self.n

    # def subsample(self, newm):


def load_sk_options(sk_options=None):
    p = ap.ArgumentParser()
    p.add_argument('-sm', '--smallm', default=500, type=int)
    p.add_argument('-sn', '--smalln', default=5000, type=int)
    p.add_argument('-snit', '--estim-Sigma-nb-iter', default=3, type=int)
    p.add_argument('-sdiv', '--estim-Sigma-sketch-division', default=30, type=int)

    args = p.parse_args(sk_options)
    return args
