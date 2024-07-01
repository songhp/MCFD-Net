import torch
import numpy as np
import random
from torch.autograd import Variable
from math import *
import torch.nn.functional as F
import os
from skimage.metrics import structural_similarity as ssim_count

USE_MULTI_GPU = False
use_half_precision_flag = False

if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    MULTI_GPU = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_ids = list(range(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(i) for i in device_ids)
else:
    device_ids = [0]
    MULTI_GPU = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def rgb_to_ycbcr(input):
    output = Variable(input.data.new(*input.size()))
    output[:, 0, :, :] = input[:, 0, :, :] * 65.481 + input[:, 1, :, :] * 128.553 + input[:, 2, :, :] * 24.966 + 16
    return output


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2):
    return ssim_count(img1.squeeze().cpu().numpy(), img2.squeeze().cpu().numpy(), data_range=1)
