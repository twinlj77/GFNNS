# import relevant packages
import numpy as np
import torch
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from torch import nn
import random
import argparse
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from steganogan import SteganoGAN

import torch
from torch.optim import LBFGS
import torch.nn.functional as F

# set seed
seed = 11111
np.random.seed(seed)
#%%

# set paramaters
# The mode can be random, pretrained-de or pretrained-d. Refer to the paper for details
mode = "pretrained-d"
steps = 2000
max_iter = 10
alpha = 0.1
eps = 0.3
num_bits = 1

# some pre-trained steganoGAN models can be found here: https://drive.google.com/drive/folders/1-U2NDKUfqqI-Xd5IqT1nkymRQszAlubu?usp=sharing
model_path = "./steganogan/pretrained/research/models/celeba_basic_1_1_mse10.steg"


steganogan = SteganoGAN.load(path=model_path, cuda=True, verbose=True)
input_im = "/home/vk352/FaceDetection/datasets/div2k/val/512/0801.jpg"
output_im = "steganographic.png"