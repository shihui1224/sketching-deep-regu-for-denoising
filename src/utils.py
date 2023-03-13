import numpy as np
import random
import torch
import math
import pycle.sketching as sk
from sketch.sketch_wrapper import SketchWrapper, load_sk_options

def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return "same image"
    PSNR = 10 * math.log10(1 / math.sqrt(mse))
    return PSNR

def snr(ref, output):
    # noise = output - ref
    a = np.sum(ref ** 2)  ## P_s
    b = np.sum((output - ref) ** 2) ## P_noise
    return 10 * math.log10(a / b)


def spiral_data(n):
    theta = np.sqrt(np.random.rand(n)) * 2 * math.pi # np.linspace(0,2*pi,100)
    r_a = theta + .1 * math.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a])
    return data_a.transpose()


def spiral_data_3d(opt, noisy=False):
    if noisy:
        n = opt.N_noisy
    else:
        n = opt.N
    theta = np.sqrt(np.random.rand(n)) * 2 * math.pi  # np.linspace(0,2*pi,100)
    r_a = opt.a * theta + opt.b * math.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a, theta])
    if noisy:
        return data_a.transpose() + opt.epsilon * np.random.randn(n, opt.INP_DIM)
    else:
        return data_a.transpose()

def circle_data(opt, r=1, noisy=False):
    if noisy:
        npoints = opt.N_noisy
    else:
        npoints = opt.N
    phi = np.linspace(0, 2.*np.pi, npoints, endpoint=False)
    x = r * np.sin(phi)
    y = r * np.cos(phi)
    if noisy:
        return np.array([x, y]).transpose() + opt.epsilon * np.random.randn(npoints, 2)
    else:
        return np.array([x, y]).transpose()


def gmm_data(opt, noisy=False):
    # weights = np.random.rand(k)
    # weights = weights/sum(weights)
    d = opt.DIM
    k = 2
    Sigma = np.zeros([k, d, d])
    mu = np.zeros([k, d])
    # for i in range(k):
    #     y = np.random.randn(1,d)
    #     y = y/np.linalg.norm(y)
    #     sig = np.transpose(y)@y + 0.02*np.eye(d)
    #     Sigma[i, :, :] = sig
    Sigma[0, :, :] = [(3.7322,    0.8027), (0.8027,    0.5306)]
    Sigma[1, :, :] = [(0.1347 ,   0.3581), (0.3581,    6.3217)]
    if noisy:
        n = opt.N_noisy
    else:
        n = opt.N
    X = np.zeros([n, d])
    label = random.choices(range(k), weights=None, k=n)
    for i in range(n):
        j = label[i]
        X[i, :] = np.random.multivariate_normal(mu[j, :], Sigma[j, :, :])
    return X


def make_grid_3d(opt):
    x = torch.linspace(opt.min, opt.max, opt.STEPs)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='xy')
    nx = opt.STEPs ** opt.DIM
    return torch.stack((X, -Y, Z), 3).reshape(nx, opt.DIM)

def make_grid(opt):
    x = torch.linspace(opt.min, opt.max, opt.STEPs)
    X, Y = torch.meshgrid(x, x, indexing='xy')
    nx = X.shape[0] * X.shape[1]
    return torch.stack((X, -Y), 2).reshape(nx, opt.DIM)


# def get_sketch(opt, x):
#     print('Computing the sketch')
#     sk_options = load_sk_options()
#     """set sketch"""
#     skw = SketchWrapper(x, opt.SK_SIZE, sk_options)
#     n = x.shape[0]
#     if n > 5000:
#         b_size = 5
#         div = int(n/b_size) - 1
#         skw.set_sketch(x[:b_size])
#         for nb_bloc in range(div):
#             skw.update_sketch(x[(nb_bloc + 1) * b_size: (nb_bloc +2) * b_size])
#     else:
#         skw.set_sketch(x)
#     y = skw.sk_emp
#     w = skw.W
#     return y, w


def compute_sketch(X, opt):
    # Draw the frequencies
    # W = sk.drawFrequencies('adaptedRadius', opt.INP_DIM, opt.SK_SIZE)
    Omega = sk.drawFrequencies('folded_gaussian', opt.DIM, opt.SK_SIZE, opt.sketch_sigma2 * np.eye(opt.DIM))
    # Create the sketching operator
    Phi = sk.SimpleFeatureMap('complexExponential', Omega, c_norm='unit')
    # Actual sketching
    z = sk.computeSketch(X, Phi)
    z_stack = np.concatenate([z.real, z.imag], axis=0)
    return Omega, z_stack