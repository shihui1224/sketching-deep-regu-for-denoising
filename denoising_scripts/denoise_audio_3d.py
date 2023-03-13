import urllib.request as urllib2
import scipy.io
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from argparse import Namespace
from scipy import signal
from scipy.io import wavfile
from src.models import make_model, get_output
from src.utils import make_grid_3d, snr
from src.denoising import denoiser
from scipy.io import wavfile
from axes_zoom_effect import zoom_effect01

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.random.seed(10)
    opt = Namespace()
    """Model definition"""
    opt.model1 = True
    opt.DEPTH = 5
    opt.model1 = True
    opt.WIDTH = 64
    opt.LR_denoise = 1e-3
    opt.gamma = .999
    opt.lambdas = 3 #0.003
    opt.IT_MAX = 20000
    opt.epsilon = .2
    # model_path = '../saved_models/guitar_audio_sk200_d5_g2.pt'
    # model_path = '../new_saved_models/guitar_audio_d5_g2_it2sk200.pt'
    # model_path = '../new_saved_models/guitar_audio_d5_g2_it2sk200_lr-test.pt'
    model_path = '../new_saved_models/v1_data_guitar_audio_d5_g2_lr5.pt'
    """grid parameters"""
    # opt.r = 1
    opt.STEPs = int(20)
    opt.STEP = 2 / opt.STEPs
    """Set training data (N, d)"""
    audio_file = '../audio_files/guitar_acoustic_021-068-100.wav'
    sample_rate, samples = wavfile.read(audio_file)
    # samples = samples[200:50000]
    samples = samples[3000:5000]

    wavfile.write('../audio_files/test2.wav', sample_rate, samples)
    samples = samples - np.mean(samples)
    samples /= samples.max()

    # print(f"sample rate is {sample_rate / 1000} kHz")
    length = samples.shape[0] / sample_rate
    # print(f"length = {length}s")
    t = np.linspace(0., length, samples.shape[0])
    order = 4
    fs = sample_rate / 1000  # sampling frequency
    cutoff1 = 20*2/fs
    cutoff2 = 30*2/fs
    # print(cutoff1)
    # print(cutoff2)

    """clean data"""
    sos1 = signal.butter(order, cutoff1, fs=fs, btype='low', analog=False, output='sos')
    s1 = signal.sosfilt(sos1, samples)
    sos2 = signal.butter(order, cutoff2, fs=fs, btype='low', analog=False, output='sos')
    s2 = signal.sosfilt(sos2, samples - s1)
    s3 = samples - s1 - s2

    data = np.vstack((s1, s2, s3)).transpose()
    """noisy data"""

    noise = np.random.normal(0, opt.epsilon, samples.shape)
    signal_noise = samples + noise
    noisy_s1 = signal.sosfilt(sos1, signal_noise)
    noisy_s2 = signal.sosfilt(sos2, signal_noise - noisy_s1)
    noisy_s3 = signal_noise - noisy_s1 - noisy_s2
    noisy_data = np.vstack((noisy_s1, noisy_s2, noisy_s3)).transpose()

    #
    # plt.figure(0)
    # plt.subplot(221)
    # plt.plot(t, signal_noise, 'silver', label='noisy s')
    # plt.plot(t, samples, 'blue', label='s')
    # plt.legend()
    # plt.subplot(222)
    # plt.plot(t, noisy_s1, 'silver', label='noisy s1')
    # plt.plot(t, s1, 'g', label='s1')
    # plt.legend()
    # plt.subplot(223)
    # plt.plot(t, noisy_s2, 'silver', label='noisy s2')
    # plt.plot(t, s2, 'r', label='s2')
    # plt.legend()
    # plt.subplot(224)
    # plt.plot(t, noisy_s3, 'silver', label='noisy s3')
    # plt.plot(t, s3, 'y', label='s3')
    # plt.legend()
    # plt.xlabel('Time [sec]')
    # plt.legend()


    # data = np.vstack((s1, s2, s3)).transpose()
    # noisy_data = np.vstack((noisy_s1, noisy_s2, noisy_s3)).transpose()
    # noisy_data /= noisy_data.max()
    opt.DIM = noisy_data.shape[1]
    opt.min = noisy_data.min()
    opt.max = noisy_data.max()

    """load model"""
    R = make_model(opt)
    # print(R)
    checkpoint = torch.load(model_path)
    R.load_state_dict(checkpoint['model_state_dict'])
    time = checkpoint['time']
    print(time)
    R.eval()

    grid = make_grid_3d(opt)
    """denoising data"""
    noisy_data_t = torch.from_numpy(noisy_data).float()
    denoising_loss, x_k = denoiser(noisy_data_t, R, opt)
    x_k = x_k.detach().numpy()
    new_s = x_k[:, 0] + x_k[:, 1] + x_k[:, 2]

    plt.figure(1)
    ax = plt.subplot(221, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1, c='green', alpha=.5, label='clean data')
    # ax.scatter(x_k[:, 0], x_k[:, 1], x_k[:, 2], s=1, c='blue', alpha=.5, label="denoised data")
    # ax.scatter(noisy_data[:, 0], noisy_data[:, 1], noisy_data[:, 2], s=1, c='silver', alpha=.5, label='noisy data')
    ax.title.set_text("data")
    ax = plt.subplot(222)
    ax.plot(np.log(denoising_loss))
    ax.title.set_text('Denoising loss (log)')


    """show regularization"""
    regularization, mu = get_output(R, grid)
    # print(output.shape)
    ax = plt.subplot(223)
    # regularization = np.reshape(regularization.detach().numpy(), (opt.STEPs, opt.STEPs, opt.STEPs))
    # regularization /= regularization.max()
    # im1 = ax.imshow(regularization.sum(axis=2), extent=[-opt.r, opt.r, -opt.r, opt.r])
    mu_np = mu.detach().numpy()
    mu_np = np.reshape(mu_np, (opt.STEPs, opt.STEPs, opt.STEPs))
    # mu_np /= mu_np.max()
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


    # plt.figure(1)
    # plt.subplot(221)
    # plt.plot(signal_noise, 'silver', label='noisy signal')
    # plt.plot(new_s, 'green', label='denoised signal')
    # plt.plot(samples, 'blue', label='original signal')
    # plt.legend()
    # plt.subplot(2,2,2)
    # plt.plot(noisy_s1, 'silver', label='noisy s1')
    # plt.plot(x_k[:, 0], 'green', label='denoised s1')
    # plt.legend()
    #
    # plt.subplot(2,2,3)
    # plt.plot(noisy_s2, 'silver', label='noisy s2')
    # plt.plot(x_k[:, 1], 'red', label='denoised s2')
    # plt.legend()
    # plt.subplot(2,2,4)
    # plt.plot(noisy_s3, 'silver', label='noisy s3')
    # plt.plot(x_k[:, 2], 'red', label='denoised s3')
    # plt.legend()


    plt.figure(3)
    ax = plt.axes(projection='3d')
    ax.scatter(noisy_data[:, 0], noisy_data[:, 1], noisy_data[:, 2], s=1, alpha=.5, c='red', label='noisy data')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1, alpha=.5, c='green', label='original data')
    ax.scatter(x_k[:, 0], x_k[:, 1], x_k[:, 2], s=1, c='blue', alpha=.5, label="denoised data")
    plt.legend()
    snr_noisy = snr(data, noisy_data)
    print('snr of noisy signal', snr_noisy)

    snr_new = snr(data, x_k)
    print('snr of new_s', snr_new)
    print(snr_new - snr_noisy)
    wavfile.write('../audio_files/output_audio.wav', sample_rate, new_s)
    axs = plt.figure().subplot_mosaic([
        ["zoom1", "zoom2"],
        ["main", "main"],
    ])
    b1 = 320
    e1 = 640
    axs["main"].plot(t, signal_noise, 'red', label='noisy signal')
    axs["main"].plot(t, new_s, 'blue', label='denoised signal')
    axs["main"].plot(t, samples, 'green', label='original signal')
    axs["zoom1"].set(xlim=(0.02, 0.04))
    axs["zoom1"].plot(t[b1: e1], signal_noise[b1: e1], 'red')
    axs["zoom1"].plot(t[b1: e1], new_s[b1: e1], 'blue')
    axs["zoom1"].plot(t[b1: e1], samples[b1: e1], 'green')
    zoom_effect01(axs["zoom1"], axs["main"], 0.02, 0.04)
    axs["zoom2"].set(xlim=(0.08, 0.09))
    b1 = 1280
    e1 = 1440

    axs["zoom2"].plot(t[b1: e1], signal_noise[b1: e1], 'red')
    axs["zoom2"].plot(t[b1: e1], new_s[b1: e1], 'blue')
    axs["zoom2"].plot(t[b1: e1], samples[b1: e1], 'green')
    zoom_effect01(axs["zoom2"], axs["main"], 0.08, 0.09)
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.show()





