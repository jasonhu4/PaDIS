import numpy as np
import odl
from operator2 import *
from scipy import interpolate
import torch
import torch.nn.functional as F
import scipy.io

class fanbeam(object):
    def __init__(self, sigSize, numAngles, numDetector,
                 min_pt=[-20, -20], max_pt=[20, 20],
                 y_area=[-40, 40], Hf=None):
        self.sigSize = sigSize
        self.numAngles = numAngles
        self.numDetector = numDetector
        self.Hf = Hf
        self.name = 'fanbeam'

        self.y_area = y_area
        self.reco_space = odl.uniform_discr(min_pt=min_pt, max_pt=max_pt, shape=sigSize, dtype='float32')
        self.angle_partition = odl.uniform_partition(0, np.pi, numAngles)
        self.detector_partition = odl.uniform_partition(self.y_area[0], self.y_area[1], self.numDetector)

        self.geometry = odl.tomo.FanBeamGeometry(self.angle_partition, self.detector_partition, src_radius=40, det_radius=40)
        #self.geometry = odl.tomo.ConeBeamGeometry(self.angle_partition, self.detector_partition,
                                                  #src_radius=133, det_radius=392, axis=axis)

        # A operator
        self.A = odl.tomo.RayTransform(self.reco_space, self.geometry, impl='astra_cuda')

    def grad(self, x, y):
        if self.Hf is None: # AT*(Ax - y)
            Ax = self.fmult(x, self.A)
            grad_d = self.ftran(Ax - y, self.A)
        else: # AT*HT*H*(Ax - y)
            pass
        return grad_d

    def eval(self, x, y):
        if self.Hf is None:
            Ax = self.fmult(x, self.A)
            d = torch.square(Ax - y)
            d = 1 / (2 * d.shape[0]) * torch.sum(d)
        else:
            pass
        return d


    def fmult(self, x, A): # x: B X Y Z
        Ax = OperatorFunction.apply(A, x)
        return Ax

    def ftran(self, z, A):# z: B A H W
        ATx = OperatorFunction.apply(A.adjoint, z)
        return ATx

    def Adagger(self, y, method='fbp', freq=0.3): # y: A H W
        if method == 'fbp':
            fbp_op = odl.tomo.fbp_op(self.A, filter_type='Hann', frequency_scaling=freq)
            x_init = OperatorFunction.apply(fbp_op, y)
        return x_init

    def Atimes(self, x): #return A*x
        return self.fmult(x, self.A)

    def ATtimes(self, y): #return A*x
        return self.ftran(y, self.A)
