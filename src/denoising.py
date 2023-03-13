import torch
import torch.nn as nn


class DenoisingNetwork(nn.Module):
    def __init__(self, opt):
        super(DenoisingNetwork, self).__init__()
        self.epsilon = opt.epsilon

    def forward(self, x, y):
        return torch.sum((x - y) ** 2) / (2 * self.epsilon)


def denoiser(y, R, opt):
    n = y.shape[0]
    loss_values = []
    x_k = y.clone().requires_grad_()
    optimizer = torch.optim.Adam(params=[x_k], lr=opt.LR_denoise)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)
    net = DenoisingNetwork(opt)
    for iter in range(opt.IT_MAX):
        loss = net(x_k, y) + opt.lambdas * torch.sum(R(x_k) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iter % (opt.IT_MAX // 10) == 0:
            print(">Iter>", iter, sep='', end='\n')
            loss_v = loss.clone() / n
            print(loss_v.item(), end='\n')
            loss_values.append(loss_v.item())
    return loss_values, torch.squeeze(x_k, 0).detach()

