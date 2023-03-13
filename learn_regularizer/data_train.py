import time
from argparse import Namespace
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models import make_model, data_train_model, get_output
from src.utils import make_grid
import pycle.utils


if __name__ == '__main__':
    # np.random.seed(10)
    opt = Namespace()
    """data parameters"""
    opt.N = int(1e6)  # number of data
    """Model definition"""
    opt.DEPTH = 3
    opt.DIM = 2
    opt.WIDTH = 64
    model_path = '../new_saved_models/v2_data_spiral.pt'
    opt.model1 = False
    """grid parameters"""
    # opt.r = 1
    opt.STEPs = int(20)
    opt.STEP = 2 / opt.STEPs
    """Training parameters"""
    opt.NUM_ITER = int(5e4)#100000
    opt.LR = 1e-6
    opt.gamma1 = 1
    print("Generating", opt.N, "data")
    data = pycle.utils.generateSpiralDataset(opt.N, normalize='l_2-unit-ball')
    # data = gmm_data(opt)
    if data.max() > 1:
        data /= data.max()
    opt.min = data.min()
    opt.max = data.max()
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(data[:, 0], data[:, 1], 1, "blue")
    axs[0, 0].set_xlim([opt.min, opt.max])
    axs[0, 0].set_ylim([opt.min, opt.max])
    axs[0, 0].set_aspect('equal')
    axs[0, 0].title.set_text("data")

    grid = make_grid(opt)
    """train network"""
    net = make_model(opt)
    print(net)
    tic = time.time()
    loss, optimizer = data_train_model(opt, net, data, grid)
    elapsed = time.time() - tic
    print('Elapsed:', elapsed)
    axs[0, 1].plot(loss)
    axs[0, 1].title.set_text('loss (log)')
    """Save model"""
    torch.save({'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'Elapsed time': elapsed}, model_path)

    """show regularization"""
    regularization, mu = get_output(net, grid)
    # output = net(grid)
    # regularization = torch.sum(output ** 2, 1)
    # mu = torch.exp(-regularization)
    regularization = np.reshape(regularization.detach().numpy(), (opt.STEPs, opt.STEPs))
    im1 = axs[1, 0].imshow(regularization, extent=[opt.min, opt.max, opt.min, opt.max])
    fig.colorbar(im1, ax=axs[1, 0])
    axs[1, 0].title.set_text("R(z)")
    """show estimated distribution"""
    mu = np.reshape(mu.detach().numpy(), (opt.STEPs, opt.STEPs))
    im2 = axs[1, 1].imshow(mu, extent=[opt.min, opt.max, opt.min, opt.max])
    plt.colorbar(im2, ax=axs[1, 1])
    axs[1, 1].title.set_text("mu")
    plt.show()