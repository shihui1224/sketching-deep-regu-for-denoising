import time
from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__),
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import src.utils
import src.models



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # np.random.seed(10)
    opt = Namespace()
    """Model definition"""
    opt.model1 = False
    opt.DEPTH = 3
    opt.DIM = 2
    opt.WIDTH = 64
    opt.SK_SIZE = 500
    opt.sketch_sigma2 = .001
    model_path = '../new_saved_models/v2_n1e6_gmm_sk500.pt'
    """grid parameters"""
    opt.STEPs = int(20)

    """Training parameters"""
    opt.NUM_ITER = int(2e5)#100000
    opt.LR = 1e-6
    opt.gamma1 = 1
    """Set training data (N, d)"""
    opt.N = int(1e6)  # number of data
    # """(option 1) gmm data"""
    data = src.utils.gmm_data(opt)

    #
    # """(option 2) spiral data"""
    # data = pycle.utils.generateSpiralDataset(opt.N, normalize='l_2-unit-ball')

    # """(option 3) learn from noisy data"""
    # data = pycle.utils.generateSpiralDataset(opt.N, normalize='l_2-unit-ball')
    # data = spiral_data(opt.N)
    # opt.epsilon = .15
    # data = data + opt.epsilon * np.random.randn(opt.N, opt.DIM)

    if data.max() > 1:
        data /= data.max()
    # opt.min = data.min()
    # opt.max = data.max()
    opt.min = -1
    opt.max = 1
    opt.STEP = (opt.max -opt.min) / opt.STEPs
    plt.figure(0)
    ax = plt.axes()
    ax.scatter(data[:, 0], data[:, 1], c='blue', s=1, alpha=0.5)
    ax.set_aspect('equal')

    """compute the sketch"""
    Omega, z_stack = src.utils.compute_sketch(data, opt)
    """set grid"""
    grid = src.utils.make_grid(opt)
    W = torch.from_numpy(Omega).float().transpose(1, 0)
    y = torch.from_numpy(z_stack).float()
    angle = W @ grid.transpose(1, 0)
    cos_angle_step = torch.cos(angle) * (opt.STEP ** opt.DIM)
    sin_angle_step = torch.sin(angle) * (opt.STEP ** opt.DIM)
    """train network"""
    net = src.models.make_model(opt)
    print(net)
    print("Learning model")
    tic = time.time()
    loss, optimizer = src.models.train_model(opt, net, grid, y, cos_angle_step, sin_angle_step)
    elapsed = time.time() - tic
    print('Elapsed:', elapsed)
    """Save model"""
    torch.save({'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'Elapsed time': elapsed}, model_path)
    """display"""
    plt.figure(0)
    ax = plt.subplot(221)
    ax.scatter(data[:, 0], data[:, 1], c='blue', s=1, alpha=0.5)
    ax.set_xlim(opt.min, opt.max)
    ax.set_ylim(opt.min, opt.max)
    ax.set_aspect('equal')
    ax.title.set_text("The sampled data")
    ax = plt.subplot(222)
    ax.plot(np.log(loss))
    ax.title.set_text('NMSE loss (log)')
    ax = plt.subplot(223)
    net.eval()
    regularization, mu = src.utils.get_output(net, grid)
    regu_np = np.reshape(regularization.detach().numpy(), (opt.STEPs, opt.STEPs))
    img = ax.imshow(regu_np, extent=[opt.min, opt.max, opt.min, opt.max])
    plt.colorbar(img, ax=ax)
    ax.set_aspect('equal')
    ax.title.set_text("R(x)")
    ax = plt.subplot(224)
    mu_np = np.reshape(mu.detach().numpy(), (opt.STEPs, opt.STEPs))
    img = ax.imshow(mu_np, extent=[opt.min, opt.max, opt.min, opt.max])
    plt.colorbar(img, ax=ax)
    ax.set_aspect('equal')
    ax.title.set_text("The learned density")
    plt.show()





