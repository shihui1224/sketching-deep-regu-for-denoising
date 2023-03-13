
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from argparse import Namespace
from scipy import signal
from scipy.io import wavfile
from src.models import make_model, train_model, get_output
from src.utils import make_grid_3d, compute_sketch
import pycle.sketching as sk
import pycle.utils
from scipy.io import wavfile

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # np.random.seed(10)
    opt = Namespace()
    """Model definition"""
    opt.model1 = True
    opt.DEPTH = 5
    opt.model1 = True
    opt.WIDTH = 64
    opt.SK_SIZE = 200
    opt.sketch_sigma2 = .001
    model_path = '../new_saved_models/guitar_audio_d5_g2_it2sk200_lr-test.pt'
    """grid parameters"""
    # opt.r = 1
    opt.STEPs = int(20)
    opt.STEP = 2 / opt.STEPs
    """Training parameters"""
    opt.NUM_ITER = int(1e5)#100000
    opt.LR = 1e-5
    opt.gamma1 = 1
    print("Loading data...")

    """Set training data (N, d)"""
    audio_file = '../audio_files/guitar_acoustic_021-068-050.wav'
    sample_rate, samples = wavfile.read(audio_file)
    samples = samples[4000:6000]
    wavfile.write('../audio_files/test.wav', sample_rate, samples)
    # samples = samples[1000:7000]
    print(samples.shape)
    samples = samples - np.mean(samples)
    samples /= samples.max()

    print(f"sample rate is {sample_rate / 1000} kHz")  # 22050 samples/sec
    length = samples.shape[0] / sample_rate
    print(f"length = {length}s")
    t = np.linspace(0., length, samples.shape[0])
    order = 4
    fs = sample_rate / 1000  # sampling frequency
    # cutoff1 = 10*2 /fs
    cutoff1 = 20*2/fs
    cutoff2 = 30*2/fs
    print(cutoff1)
    print(cutoff2)
    sos1 = signal.butter(order, cutoff1, fs=fs, btype='low', analog=False, output='sos')
    s1 = signal.sosfilt(sos1, samples)
    sos2 = signal.butter(order, cutoff2, fs=fs, btype='low', analog=False, output='sos')
    s2 = signal.sosfilt(sos2, samples - s1)
    s3 = samples - s1 - s2
    # frequencies, ts, spectrogram = signal.spectrogram(x=samples, fs=sample_rate)
    # plt.figure(2)
    # ax = plt.subplot(311)
    # ax.plot(20* np.log10(np.abs(samples)))
    # ax = plt.subplot(312)
    # ax.plot(20* np.log10(np.abs(s1)))
    # ax = plt.subplot(313)
    # ax.plot(20* np.log10(np.abs(s2)))
    #
    #
    plt.figure(0)
    plt.subplot(211)

    plt.plot(t, samples, 'blue', label='s (original signal)')
    plt.subplot(212)
    plt.plot(t, s1, 'green', linewidth=2, label='s1')
    plt.plot(t, s2, 'red', linewidth=2, label='s2')
    plt.plot(t, s3, 'yellow', linewidth=2, label='s3')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()




    #
    # plt.subplot(3, 1, 2)
    # frequencies, ts, spectrogram = signal.spectrogram(x=s1, fs=sample_rate)
    # plt.pcolormesh(ts, frequencies, spectrogram, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.title('spectrogram of s1')
    # plt.subplot(3, 1, 3)
    # frequencies, ts, spectrogram = signal.spectrogram(s2, sample_rate)
    # plt.pcolormesh(ts, frequencies, spectrogram, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.title('spectrogram of s2')


    data = np.vstack((s1, s2, s3)).transpose()
    data /= data.max()
    print(data.shape)
    opt.DIM = data.shape[1]
    opt.min = data.min()
    opt.max = data.max()


    plt.figure(1)
    ax = plt.subplot(221, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1, c='blue')

    ax.title.set_text("data")
    """sketch learning"""
    Omega, z_stack = compute_sketch(data, opt)
    grid = make_grid_3d(opt)
    W = torch.from_numpy(Omega).float().transpose(1, 0)
    y = torch.from_numpy(z_stack).float()
    angle = W @ grid.transpose(1, 0)
    cos_angle_step = torch.cos(angle) * (opt.STEP ** opt.DIM)
    sin_angle_step = torch.sin(angle) * (opt.STEP ** opt.DIM)
    """train network"""
    net = make_model(opt)
    print(net)
    print("Learning model")
    tic = time.time()
    loss, optimizer = train_model(opt, net, grid, y, cos_angle_step, sin_angle_step)
    elapsed = time.time() - tic
    print('Elapsed:', elapsed)
    ax = plt.subplot(222)
    ax.plot(np.log(loss))
    ax.title.set_text('NMSE loss (log)')


    """Save model"""
    torch.save({'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'time': elapsed}, model_path)


    """show regularization"""
    regularization, mu = get_output(net, grid)
    # print(output.shape)
    ax = plt.subplot(223)
    # regularization = np.reshape(regularization.detach().numpy(), (opt.STEPs, opt.STEPs, opt.STEPs))
    # regularization /= regularization.max()
    # im1 = ax.imshow(regularization.sum(axis=2), extent=[-opt.r, opt.r, -opt.r, opt.r])
    mu_np = mu.detach().numpy()
    mu_np = np.reshape(mu_np, (opt.STEPs, opt.STEPs, opt.STEPs))
    mu_np /= mu_np.max()
    # im1 = ax.imshow(mu_np.sum(axis=2), extent=[-opt.r, opt.r, -opt.r, opt.r])
    im1 = ax.imshow(mu_np.sum(axis=2), extent=[opt.min, opt.max, opt.min, opt.max])
    plt.colorbar(im1, ax=ax)
    ax.title.set_text("mu 2d")

    ax = plt.subplot(224, projection='3d')
    x, y, z = np.where(mu_np >.01)
    x = x - x.mean()
    y = y - y.mean()
    z = z - z.mean()
    ax.scatter3D(x/x.max(), y/y.max(), z/z.max())
    ax.title.set_text("mu")


    # plt.figure(2)
    # ax = plt.axes(projection="3d")
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1, c='blue')
    plt.show()





