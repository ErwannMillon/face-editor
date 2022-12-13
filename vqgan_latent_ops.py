import torch
import torch.nn.functional as F
import torch.nn as nn
from gradient_flow_ops import ReplaceGrad
replace_grad = ReplaceGrad.apply

def vector_quantize(x, codebook):

    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)