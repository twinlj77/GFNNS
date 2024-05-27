# import relevant packages for gasp

#{"id": "celebahq64_experiment", "dataset": "celebahq", "path_to_data": ".\\data\\celeba_hq\\train\\64_28000\\", "resolution": 64, "training": {"epochs": 300, "batch_size": 64, "max_num_points": null, "lr": 0.0001, "lr_disc": 0.0004, "r1_weight": 10.0, "print_freq": 50, "model_save_freq": 50}, "generator": {"layer_sizes": [128, 128, 128], "latent_dim": 64, "hypernet_layer_sizes": [256, 512], "encoding": {"num_frequencies": 128, "std_dev": 2.0}}, "discriminator": {"norm_order": 2.0, "add_batchnorm": true, "add_weightnet_batchnorm": true, "deterministic": true, "same_coordinates": true, "linear_layer_sizes": [], "layer_configs": [{"out_channels": 64, "num_output_points": 4096, "num_neighbors": 9, "mid_channels": [16, 16, 16, 16]}, {"num_output_points": 1024, "num_neighbors": 9}, {"out_channels": 128, "num_output_points": 1024, "num_neighbors": 9, "mid_channels": [16, 16, 16, 16]}, {"num_output_points": 256, "num_neighbors": 9}, {"out_channels": 256, "num_output_points": 256, "num_neighbors": 9, "mid_channels": [16, 16, 16, 16]}, {"num_output_points": 64, "num_neighbors": 9}, {"out_channels": 512, "num_output_points": 64, "num_neighbors": 9, "mid_channels": [16, 16, 16, 16]}, {"num_output_points": 16, "num_neighbors": 9}, {"out_channels": 1, "num_output_points": 1, "num_neighbors": 16, "mid_channels": [16, 16, 16, 16]}]}}

import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import numpy as np

# packages for gasp
import imageio
import json
import torch
from viz.plots import plot_point_cloud_batch, plot_voxels_batch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from data.conversion import GridDataConverter, PointCloudDataConverter
# Note that this import is necessary for load_function_distribution
# properly instantiate the FourierFeatures
from models.function_representation import FourierFeatures
from models.function_distribution import load_function_distribution
from torchvision.utils import save_image

# GASP Generater ########################################################
gasp_device = torch.device('cuda')

exp_dir = 'trained-models/celebahq64'  # 修改此处

with open(exp_dir + '/config.json') as f:  # 修改此处
    config = json.load(f)

# Create appropriate data converter based on config
"""data_shape (tuple of ints): Tuple of the form (feature_dim, coordinate_dim_1, coordinate_dim_2, ...). 
   While point clouds do not have a data_shape this will be used when sampling points on grid to generate samples.
   normalize_features (bool): If True normalizes features (e.g. RGB or occupancy values to lie in [-1, 1]."""

if config["dataset"] == 'mnist':
    data_shape = (1, config["resolution"], config["resolution"])
    data_converter = GridDataConverter(gasp_device, data_shape,
                                       normalize_features=True)
elif config["dataset"] == 'celebahq':
    data_shape = (3, config["resolution"], config["resolution"])
    data_converter = GridDataConverter(gasp_device, data_shape,
                                       normalize_features=True)
elif config["dataset"] == 'shapenet_voxels':
    data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
    data_converter = GridDataConverter(gasp_device, data_shape,
                                       normalize_features=True)
elif config["dataset"] == 'shapenet_point_clouds':
    data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
    data_converter = PointCloudDataConverter(gasp_device, data_shape,
                                             normalize_features=True)

t_o=time.time()

# Load function distribution weights
func_dist = load_function_distribution(gasp_device, exp_dir + '/model_250.pt')  # 修改此处
func_dist.to('cuda')

# Sample one image from model
num_samples = [1]
latent_z = func_dist.latent_distribution.sample((num_samples))

samples = func_dist.sample_data_by_latent(data_converter, latent_z, num_samples=1)

# Convert list of samples to batch of samples  样品列表转换为样品批次
samples = torch.cat([sample.unsqueeze(0) for sample in samples], dim=0).detach()

t_1=time.time()
t=t_1-t_o

# Generator end #########################################################################################

# fixed neural network #########################################################################################
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from torch import nn
import random
import argparse
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from steganogan import SteganoGAN
from steganogan.encoders import BasicEncoder
from steganogan.decoders import BasicDecoder
from steganogan.critics import BasicCritic

import torch
from torch.optim import LBFGS
import torch.nn.functional as F

t_2=time.time()


# set seed
seed = 11111
np.random.seed(seed)
torch.manual_seed(seed)

# L1 = np.random.randn(3, 3)
# print(L1)
#
# np.random.seed(seed)
# L2 = np.random.randn(3, 3)
# print(L2)

from math import log10
import cv2


def calc_psnr(img1, img2):
    ### args:
    # img1: [h, w, c], range [0, 255]
    # img2: [h, w, c], range [0, 255]
    diff = (img1 - img2) / 255.0
    diff[:, :, 0] = diff[:, :, 0] * 65.738 / 256.0
    diff[:, :, 1] = diff[:, :, 1] * 129.057 / 256.0
    diff[:, :, 2] = diff[:, :, 2] * 25.064 / 256.0

    diff = np.sum(diff, axis=2)
    mse = np.mean(np.power(diff, 2))
    return -10 * log10(mse)


def calc_ssim(img1, img2):
    def ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    ### args:
    # img1: [h, w, c], range [0, 255]
    # img2: [h, w, c], range [0, 255]
    # the same outputs as MATLAB's
    border = 0
    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h - border, border:w - border]
    img2_y = img2_y[border:h - border, border:w - border]
    img2_y = img2_y[border:h - border, border:w - border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def shuffle_params(m):
    if type(m) == nn.Conv2d or type(m) == nn.BatchNorm2d:
        param = m.weight
        m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, param.shape)).float())

        param = m.bias
        m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))


idx = 801

num_bits = 1
steps = 2000
max_iter = 20
alpha = 0.1
eps = 0.305

criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

from datetime import datetime


def img_plot(img):
    if img.shape[0] == 1:
        plt.imshow(img[0], cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(img.numpy().transpose(1, 2, 0), vmin=0, vmax=1)


for seed in [11111, 22222, 33333, 44444, 55555, 66666, 777777, 88888, 99999, 0, 22222, 33333, 44444, 55555, 66666,
             777777, 88888, 99999, 0]:
    np.random.seed(seed)
    model = BasicDecoder(num_bits, hidden_size=128)
    model.apply(shuffle_params)
    model.to('cuda')

    timestamp = datetime.now().strftime('%d_%H%M%S')

    image = samples

    # save_image(image, 'sample_image_timestamp.png')

    # image = f"/home/vk352/FaceDetection/datasets/div2k/val/512/{idx:04d}.jpg"
    # image = f"./xxx_64.jpg"
    # image = imread(image, pilmode='RGB') / 255.0
    # image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
    image = image.to('cuda')
    out = model(image)
    torch.manual_seed(idx)
    target = torch.bernoulli(torch.empty(out.shape).uniform_(0, 1)).to(out.device)
    print(target.shape)
    eps = eps - 0.005
    print("eps:", eps)

    adv_image = samples.clone().detach().contiguous()
    for i in range(steps // max_iter):
        adv_image.requires_grad = True
        optimizer = LBFGS([adv_image], lr=alpha, max_iter=max_iter)


        def closure():
            outputs = model(adv_image)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            return loss


        optimizer.step(closure)

        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=0, max=1).detach().contiguous()

        acc = len(torch.nonzero((model(adv_image) > 0).float().view(-1) != target.view(-1))) / target.numel()
        print(i, acc)
        if acc == 0: break

    print(seed)
    psnr = calc_psnr((image.squeeze().permute(2, 1, 0) * 255).detach().cpu().numpy(),
                     (adv_image.squeeze().permute(2, 1, 0) * 255).detach().cpu().numpy())
    print("psnr:", psnr)
    print("ssim:", calc_ssim((image.squeeze().permute(2, 1, 0) * 255).detach().cpu().numpy(),
                             (adv_image.squeeze().permute(2, 1, 0) * 255).detach().cpu().numpy()))
    print("error:", acc)
    lbfgsimg = (adv_image.cpu().squeeze().permute(2, 1, 0).numpy() * 255).astype(np.uint8)


    # save_image(adv_image, 'sample_image_stego_{timestamp}.png')
    if psnr > 19:
        filename = f"sample_image_{timestamp}.png"
        save_image(samples, filename)
        filename = f"sample_image_stego_{timestamp}.png"
        save_image(adv_image, filename)
        break

# 建立文件夹
"""timestamp = time.strftime("%Y-%m-%d_%H-%M")
directory = "{}_{}".format(timestamp, config["id"])
if not os.path.exists(directory):
    os.makedirs(directory)
#保存图像
cv2.imwrite(directory+"/"+"sample_image_stego.png",adv_image)"""

t_3=time.time()
T=t_3-t_2
print(t,T)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow((image.squeeze().permute(2, 1, 0) * 255).detach().cpu().numpy().astype(np.uint8))
ax[1].imshow((adv_image.squeeze().permute(2, 1, 0) * 255).detach().cpu().numpy().astype(np.uint8))



