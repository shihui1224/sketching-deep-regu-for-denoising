import matplotlib.pyplot as plt
import numpy as np
import torch
from argparse import Namespace
from sklearn.metrics import mean_squared_error as mse
from src.denoising import denoiser
from src.models import make_model, get_output
from src.utils import gmm_data, make_grid
import pycle.utils


if __name__ == '__main__':
    # plt.close('all')
    # spiral_model_test
    model_path = ['../saved_models/sk100_spiral_v2_d2','../saved_models/data_spiral_it1e5.pt']
    # model_path = ['saved_models/test.pt','saved_models/data_spiral_it4e4.pt']
    # model_path = [ 'saved_models/d3_gmm_sk.pt', 'saved_models/data_gmm_it4e4.pt']
    fname = 'results4.png'
    # model_path = [ '../saved_models/sk100_gmm_it1e5_v2.pt', '../saved_models/data_gmm_it1e5_v2.pt']
    opt = Namespace()
    opt.model1 = False
    opt.N_noisy = 500
    opt.epsilon = 0.15
    opt.DIM = 2
    """grid parameters"""
    opt.r = 1
    opt.STEPs = int(20)
    opt.STEP = 2 / opt.STEPs
    """Model definition"""
    opt.DEPTH = 3
    opt.DIM = 2
    opt.WIDTH = 64
    opt.LR_denoise = 1e-3
    opt.IT_MAX = 1000
    opt.gamma = .99 #for learning rate
    opt.lambdas = .5 # for denoising
    opt.N_noisy = 500
    opt.N_test = opt.STEPs * opt.STEPs
    """generate and display data"""
    data = pycle.utils.generateSpiralDataset(opt.N_noisy, normalize='l_2-unit-ball')
    # data = gmm_data(opt, True)
    data /= data.max()
    opt.min = data.min()
    opt.max = data.max()
    noisy_data = data + opt.epsilon * np.random.randn(opt.N_noisy, opt.DIM)
    noisy_data_t = torch.from_numpy(noisy_data).float()
    # noisy_data = generate_gmm_data(2, 3, opt.N)
    fig, axs = plt.subplots(1, len(model_path))
    """set grid"""
    grid = make_grid(opt)
    for i, path in enumerate(model_path):
        """load model"""
        R = make_model(opt)
        checkpoint = torch.load(path)
        R.load_state_dict(checkpoint['model_state_dict'])
        R.eval()

        """denoising data"""
        denoising_loss, x_k = denoiser(noisy_data_t, R, opt)
        axs[i].scatter(noisy_data[:, 0], noisy_data[:, 1], 1, c='red', alpha=0.5, label="noisy data")
        if i == 0:
            color = 'blue'
        else:
            color = 'olive'
        axs[i].scatter(x_k[:, 0], x_k[:, 1], 1, c=color, alpha=0.5, label="denoised data")
        axs[i].legend()
        axs[i].set_xlim(-opt.r, opt.r)
        axs[i].set_ylim(-opt.r, opt.r)
        axs[i].set_aspect('equal')

    plt.savefig(fname, bbox_inches ="tight")
    plt.show()
