import time
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch
def freeze_module(module):
    for param in module.parameters():
      param.requires_grad = False
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    return (device)
