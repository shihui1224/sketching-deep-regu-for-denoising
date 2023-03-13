import matplotlib.pyplot as plt
import numpy as np
import torch
from argparse import Namespace
from src.denoising import denoiser
from src.models import make_model, get_output
from src.utils import gmm_data, make_grid, spiral_data, snr
import  random
import pycle.utils

if __name__ == '__main__':
    # plt.close('all')

    # model_path = 'saved_models/d3_gmm_sk.pt'

    # model_path = '../saved_models/data_gmm_it4e4.pt'
    # model_path = '../saved_models/sk100_spiral_d3.pt'
    # model_path = '../saved_models/sk_spiral_noisy_v1.pt'
    # model_path = '../saved_models/data_gmm_it1e5.pt'
    # model_path = 'saved_models/d5_g1_sk200_spiral_it5e4_lr8.pt'
    # model_path = 'saved_models/sk300_spiral.pt'
    # model_path2 = 'saved_models/data_train_regu.pt'
    # model_path = '../saved_models/data_gmm_it1e5_v2.pt'
    # model_path = '../new_saved_models/v2_n1e6_sk500.pt'
    model_path = '../new_saved_models/v2_n1e6_gmm_sk50.pt'
    np.random.seed(10)
    # random.seed(10)
    opt = Namespace()
    """grid parameters"""
    opt.r = 1
    opt.STEPs = int(20)
    opt.STEP = 2 / opt.STEPs
    """model definition"""
    opt.model1 = False
    opt.DEPTH = 3
    opt.DIM = 2
    opt.WIDTH = 64
    """denoising parameters"""
    opt.IT_MAX = 10000
    opt.LR_denoise = 1e-3
    opt.gamma = .999     #for learning rate
    opt.lambdas = 0.04  # for denoising
    """data parameters"""
    opt.epsilon = .15  #noise level
    opt.N_noisy = 500

    """(option 1) gmm data"""
    data = gmm_data(opt, True)

    # """(option 2) spiral data"""
    # data = pycle.utils.generateSpiralDataset(opt.N_noisy, normalize='l_2-unit-ball')
    # data = spiral_data(opt.N_noisy)
    if data.max() > 1:
        data /= data.max()
    opt.min = data.min()
    opt.max = data.max()
    noisy_data = data + opt.epsilon * np.random.randn(opt.N_noisy, opt.DIM)

    # opt.N_test = opt.STEPs * opt.STEPs

    """load model"""
    R = make_model(opt)
    checkpoint = torch.load(model_path)
    R.load_state_dict(checkpoint['model_state_dict'])
    time = checkpoint['Elapsed time']
    print("Training time", time)
    R.eval()

    """denoising data"""
    noisy_data_t = torch.from_numpy(noisy_data).float()
    denoising_loss, x_k = denoiser(noisy_data_t, R, opt)

    """display"""
    plt.figure(0)
    ax = plt.subplot(221)
    ax.scatter(noisy_data[:, 0], noisy_data[:, 1], c="red", s=1, alpha=.5, label="Noisy data")
    ax.scatter(x_k[:, 0], x_k[:, 1], c='blue', s=1, alpha=0.5, label="Denoised data")
    ax.set_xlim([-opt.r, opt.r])
    ax.set_ylim([-opt.r, opt.r])
    ax.set_aspect('equal')
    ax.legend()

    ax = plt.subplot(222)
    ax.plot(np.log(denoising_loss))
    ax.title.set_text("Denoising loss (log)")

    ax = plt.subplot(223)
    grid = make_grid(opt)
    regu, mu = get_output(R, grid)
    mu_np = np.reshape(mu.detach().numpy(), [opt.STEPs, opt.STEPs])
    img = ax.imshow(mu_np, extent=[-opt.r, opt.r, -opt.r, opt.r])
    plt.colorbar(img, ax=ax)
    ax.set_aspect('equal')
    ax.title.set_text('The learned density of data')

    ax = plt.subplot(224)

    mu_np = np.reshape(regu.detach().numpy(), [opt.STEPs, opt.STEPs])
    ax.imshow(mu_np, extent=[-opt.r, opt.r, -opt.r, opt.r])
    ax.set_xlim([-opt.r, opt.r])
    ax.set_ylim([-opt.r, opt.r])
    ax.set_aspect('equal')
    ax.title.set_text('The learned R')

    snr_value1 = snr(data, noisy_data)
    print("noisy snr", snr_value1)

    x_k = np.array(x_k)
    snr_value = snr(data, x_k)
    print("new snr", snr_value)
    print(snr_value - snr_value1)
    plt.figure(1)
    ax = plt.axes()
    ax.scatter(noisy_data[:, 0], noisy_data[:, 1], c="red", s=1, label="Noisy data")
    ax.scatter(x_k[:, 0], x_k[:, 1], c='blue', s=1, label="Denoised data")
    ax.set_xlim([-opt.r, opt.r])
    ax.set_ylim([-opt.r, opt.r])
    ax.set_aspect('equal')
    ax.legend()
    plt.show()
