import matplotlib.pyplot as plt
from argparse import Namespace
import numpy as np
import torch
from src.denoising import denoiser
from src.models import make_model, get_output
from src.utils import make_grid_3d, snr
import pycle.utils

if __name__ == '__main__':
    np.random.seed(10)
    model_path = '../new_saved_models/3Dspiral_d4v1_sk1000.pt'
    opt = Namespace()
    opt.model1 = True
    opt.DEPTH = 4
    # number of hidden layers in the network
    opt.WIDTH = 64 # number of neurons in the first hidden layer

    opt.LR_denoise = 1e-3
    opt.gamma = .999 # decay parameter for learning rate
    opt.IT_MAX = 1000
    opt.lambdas = .03  # >0 the weight of the regularization for denoising
    """data parameters"""
    opt.N_noisy = 500
    opt.DIM = 3
    opt.epsilon = .2 # noise level
    """grid parameters"""
    opt.r = 1
    opt.STEPs = int(20) # number of points in each dimenssion
    opt.STEP = 2 / opt.STEPs

    """generate and display data"""
    data = pycle.utils.generateSpiralDataset(opt.N_noisy, dim=3)
    opt.min = data.min()
    opt.max = data.max()
    noisy_data = data + opt.epsilon * np.random.randn(opt.N_noisy, opt.DIM)

    """load model"""
    R = make_model(opt)
    print(R)
    checkpoint = torch.load(model_path)
    R.load_state_dict(checkpoint['model_state_dict'])
    R.eval()

    """set grid"""
    grid = make_grid_3d(opt)

    """denoising data"""
    noisy_data_t = torch.from_numpy(noisy_data).float()
    denoising_loss, x_k = denoiser(noisy_data_t, R, opt)

    """display"""
    plt.figure(0)
    ax = plt.subplot(221, projection="3d")
    ax.scatter(noisy_data[:,0], noisy_data[:,1], noisy_data[:,2], c="red", s=1, alpha=.5, label="Noisy data")
    ax.scatter(x_k[:, 0], x_k[:, 1], x_k[:, 2], c='blue', s=1, alpha=0.5, label="Denoised data")
    ax.legend()

    ax = plt.subplot(222)
    ax.plot(np.log(denoising_loss))
    ax.title.set_text("Denoising loss (log)")

    ax = plt.subplot(223)
    _, mu = get_output(R, grid)
    mu_np = np.reshape(mu.detach().numpy(), [opt.STEPs, opt.STEPs, opt.STEPs])
    mu_np_2d = mu_np.sum(axis=2)
    img = ax.imshow(mu_np_2d, extent=[-opt.r, opt.r,-opt.r, opt.r])
    plt.colorbar(img, ax=ax)
    ax.set_aspect('equal')
    ax.title.set_text('The learned density of data (projected on 2D)')

    ax = plt.subplot(224, projection="3d")
    x, y, z = np.where(mu_np >.01)
    x = x - x.mean()
    y = y - y.mean()
    z = z - z.mean()
    ax.scatter3D(x/x.max(), y/y.max(), z/z.max())
    ax.title.set_text('The learned density of data')

    # plt.figure(1)
    # ax = plt.axes(projection='3d')
    # ax.scatter(noisy_data[:, 0], noisy_data[:, 1], noisy_data[:, 2], c='red', s=1, alpha=1, label="noisy data")
    # ax.scatter(x_k[:, 0], x_k[:, 1], x_k[:, 2], c='blue', s=1, alpha=1, label="denoised data")
    # ax.legend()

    # plt.figure(2)
    # ax = plt.axes(projection='3d')
    # x, y, z = np.where(mu_np > .01)
    # x = x - x.mean()
    # y = y - y.mean()
    # z = z - z.mean()
    # ax.scatter3D(x / x.max(), y / y.max(), z / z.max())
    snr_value = snr(data, noisy_data)
    print("noisy snr", snr_value)

    x_k = np.array(x_k)
    snr_value = snr(data, x_k)
    print("new snr", snr_value)
    time = checkpoint['time']
    print(time)
    plt.figure(1)
    ax = plt.axes(projection="3d")
    ax.scatter(noisy_data[:,0], noisy_data[:,1], noisy_data[:,2], c="red", s=1, label="Noisy data")
    ax.scatter(x_k[:, 0], x_k[:, 1], x_k[:, 2], c='blue', s=1, alpha=0.5, label="Denoised data")
    ax.legend()
    plt.show()