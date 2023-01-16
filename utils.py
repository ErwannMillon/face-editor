import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import lab2rgb, rgb2lab
from torch import nn


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    return (device)
