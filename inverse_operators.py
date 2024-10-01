import torch
import numpy as np
import sys
import scipy
import torch.nn.functional as F

sys.path.append('/n/badwater/z/jashu/Patch-Diffusion/odlstuff')
from fanbeam import *
from parbeam import *
from functools import partial
import math

class InverseOperator(object):
    def __init__(self, imsize, name, views=10, channels=1, blursize=5, scale_factor=2):
        self.imsize = imsize
        self.name = name
        self.views = views
        self.channels = channels
        if name == 'ct_parbeam':
            self.radon_sv = parbeam([imsize, imsize], views, 512)
        elif name == 'ct_fanbeam':
            self.radon_sv = fanbeam([imsize, imsize], views, 512)
        elif name == 'lact':
            self.radon_sv = parbeam([imsize, imsize], views, 512, lact=True)
        elif name == 'deblur_uniform':
            return NotImplementedError

        elif name == 'super':
            return NotImplementedError

        elif name == 'denoise':
            pass
        else:
            return NotImplementedError

    def A(self, x):
        if self.name == 'ct_parbeam' or self.name == 'ct_fanbeam' or self.name == 'lact':
            x2 = torch.unsqueeze(torch.clone(x), 0)
            out = self.radon_sv.Atimes(x2)
            return torch.squeeze(out, dim=0)
        elif self.name == 'denoise':
            return x
        return NotImplementedError

    def AT(self, y):
        if self.name == 'ct_parbeam' or self.name == 'ct_fanbeam' or self.name == 'lact':
            y2 = torch.unsqueeze(y, 0)
            out = self.radon_sv.ATtimes(y2)
            return torch.squeeze(out, 0)
        elif self.name == 'denoise':
            return y
        return NotImplementedError

    def Adagger(self, y):
        if self.name == 'ct_parbeam' or self.name == 'ct_fanbeam' or self.name == 'lact':
            y2 = torch.unsqueeze(y, 0)
            out = self.radon_sv.Adagger(y2)
            return torch.squeeze(out, 0)
        elif self.name == 'denoise':
            return y
