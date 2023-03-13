import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR


class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        layer = nn.Linear(opt.DIM, opt.WIDTH)
        layers = [layer, nn.ReLU()]
        width = opt.WIDTH
        if opt.model1:
            for i in range(opt.DEPTH - 1):
                if i < 3:
                    width_n = opt.WIDTH * (i+1)
                layer = nn.Linear(width, width_n)
                width = width_n
                layers.append(layer)
                layers.append(nn.ReLU())
        else:
            for i in range(opt.DEPTH - 1):
                width_n = width * 2
                layer = nn.Linear(width, width_n)
                width = width_n
                layers.append(layer)
                layers.append(nn.ReLU())
        layer = nn.Linear(width_n, opt.DIM)
        layers.append(layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SkLoss(nn.Module):
    def __int__(self):
        super(SkLoss, self).__int__()

    def forward(self, output, target, cos_angle_step, sin_angle_step):
        R = torch.sum(output ** 2, 1)
        mu = torch.exp(-R)
        real_part = torch.sum(cos_angle_step * mu, 1)
        img_part = torch.sum(sin_angle_step * mu, 1)
        sk = torch.cat((real_part, img_part))
        res = sk - target
        return torch.dot(res, res)


def get_output(net, grid):
    output = net(grid)
    R = torch.sum(output ** 2, 1)
    mu = torch.exp(-R)
    return R, mu


def make_model(opt):
    model = Network(opt)
    return model


def train_model(opt, net, grid, target, cos_angle, sin_angle):
    net.train()  # set train model
    loss_values = []
    optimizer = optim.Adam(params=net.parameters(), lr=opt.LR)
    scheduler = ExponentialLR(optimizer, gamma=opt.gamma1)
    my_loss = SkLoss()
    # if opt.CUDA:
    #     if torch.cuda.is_available():
    #         x = x.cuda()
    #         target = target.cuda()
    for iter in range(opt.NUM_ITER):
        optimizer.zero_grad()
        output = net(grid)
        loss = my_loss(output, target, cos_angle, sin_angle)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iter % 100 == 99:
            loss_v = loss.clone()
            print(">Iter>", iter, '', loss_v.item(), end='\n')
            loss_values.append(loss_v.item())
    return loss_values, optimizer


class DATALoss(nn.Module):
    def __int__(self):
        super(DATALoss, self).__int__()

    def forward(self, net, data, d, grid):
        f_z = net(grid)
        R_z = torch.sum(f_z ** 2, 1)
        dist = (R_z - d)
        return torch.sum(dist ** 2)


def data_train_model(opt, net, data, grid):
    net.train()
    loss_values = []
    data = torch.from_numpy(data).float()
    optimizer = optim.Adam(params=net.parameters(), lr=opt.LR)
    scheduler = ExponentialLR(optimizer, gamma=opt.gamma1)
    my_loss = DATALoss()
    d, _ = torch.min(torch.cdist(grid, data), 1)
    for iter in range(opt.NUM_ITER):
        optimizer.zero_grad()
        loss = my_loss(net, data, d, grid)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iter % 100 == 99:
            loss_v = loss.clone()
            print(">Iter>", iter, '', loss_v.item(), end='\n')
            loss_values.append(loss_v.item())
    return loss_values, optimizer