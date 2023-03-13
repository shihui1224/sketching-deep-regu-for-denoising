import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from argparse import Namespace
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__),
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
import src.utils
import src.models
import pycle.utils

if __name__ == '__main__':
    # plt.close('all')
    opt = Namespace()
    opt.model1 = True
    opt.DEPTH = 4
    opt.WIDTH = 64
    opt.SK_SIZE = 1000
    opt.sketch_sigma2 = .001
    """data parameters"""
    opt.N = int(1e6)
    opt.DIM = 3
    """grid parameters"""
    opt.STEPs = int(20)
    opt.STEP = 2 / opt.STEPs
    """Training parameters"""
    opt.NUM_ITER = int(1e5)#100000
    opt.LR = 1e-6
    opt.gamma1 = 1
    model_path = '../new_saved_models/3Dspiral_d4v1_sk1000.pt'
    """generate and display data"""
    data = pycle.utils.generateSpiralDataset(opt.N, dim=3)
    opt.min = data.min()
    opt.max = data.max()
    """set sketch"""
    Omega, z_stack = src.utils.compute_sketch(data, opt)
    W = torch.from_numpy(Omega).float().transpose(1, 0)
    y = torch.from_numpy(z_stack).float()
    """set grid"""
    grid = src.utils.make_grid_3d(opt)
    angle = W @ grid.transpose(1, 0)
    cos_angle_step = torch.cos(angle) * (opt.STEP ** opt.DIM)
    sin_angle_step = torch.sin(angle) * (opt.STEP ** opt.DIM)
    """train network"""
    net = src.utils.make_model(opt)
    print(net)
    tic = time.time()
    loss, optimizer = src.models.train_model(opt, net, grid, y, cos_angle_step, sin_angle_step)
    elapsed = time.time() - tic
    print('Elapsed:', elapsed)
    """Save model"""
    torch.save({'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'time': elapsed}, model_path)
    net.eval()

    plt.figure(0)
    ax = plt.subplot(221, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', s=1, alpha=0.5)
    plt.subplot(2,2,2)
    plt.plot(np.log(loss))
    plt.title('NMSE loss (log)')
    """show regularization"""
    regularization, mu = src.utils.get_output(net, grid)
    plt.subplot(2,2,3)
    regu_np = np.reshape(regularization.detach().numpy(), (opt.STEPs, opt.STEPs, opt.STEPs))
    regu_np_2d = np.sum(regu_np, 2)
    img = plt.imshow(regu_np_2d, extent=[opt.min, opt.max, opt.min, opt.max])
    plt.colorbar(img)
    plt.title('regularization')
    """show estimated distribution"""
    plt.subplot(2,2,4)
    mu_np = np.reshape(mu.detach().numpy(), (opt.STEPs, opt.STEPs, opt.STEPs))
    mu_np_2d = np.sum(mu_np, 2)
    img = plt.imshow(mu_np_2d, extent=[opt.min, opt.max, opt.min, opt.max])
    plt.colorbar(img)
    plt.title('mu')
    plt.show()
