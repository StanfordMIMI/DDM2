import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import copy
from .utils import *


class N2N(nn.Module):
    '''
    Noise model as in Noise2Noise
    '''
    def __init__(
        self,
        denoise_fn,
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()


    @torch.no_grad()
    def denoise(self, x_in):
        return self.denoise_fn(x_in['condition'])

    def p_losses(self, x_in, noise=None):
        debug_results = dict()
        x_start = x_in['X'] # our
        [b, c, w, h] = x_start.shape

        x_recon = self.denoise_fn(x_in['condition'])

        loss1 = self.mse_loss(x_recon, x_in['X'])

        return dict(total_loss=loss1)


    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
