'''
Class for quadratic-norm on Conebeam_3D  Sino measurements
'''
import numpy as np
import odl
from operator2 import *
from scipy import interpolate
import torch 
import torch.nn.functional as F

# Working on data with shape: B C H W 
# In our case it is B X Y Z
class CBCTDF(object):
    def __init__(self, sigSize=[448, 448, 448], numAngles=4, numDetector=[2000, 2000],
                 min_pt=[-5.5875, -5.5875, -5.5875], max_pt=[5.5875, 5.5875, 5.5875], 
                 y_area=[-25, 25], rot_axis=2, Hf=None, use_I0=True):
        self.sigSize = sigSize
        self.numAngles = numAngles
        self.numDetector = numDetector
        self.Hf = Hf
        self.use_I0  = use_I0
        self.name = 'cbct'
        
        self.y_area = y_area
        self.reco_space = odl.uniform_discr(min_pt=min_pt, max_pt=max_pt, shape=sigSize, dtype='float32')
        self.angle_partition = odl.uniform_partition(0, np.pi, numAngles)
        self.detector_partition = odl.uniform_partition([self.y_area[0], self.y_area[0]], [self.y_area[1], self.y_area[1]], self.numDetector)
        if rot_axis == 2:
            axis=(0, 0, 1)
        elif rot_axis == 1:
            axis=(0, 1, 0)
        elif rot_axis == 0:
            axis=(1, 0, 0)
        else:
            raise ValueError('Rotation patten not found !') 
            
        self.geometry = odl.tomo.ConeBeamGeometry(self.angle_partition, self.detector_partition, 
                                                  src_radius=133, det_radius=392, axis=axis)

        # A operator
        self.A = odl.tomo.RayTransform(self.reco_space, self.geometry, impl='astra_cuda')
        
        # get I0
        self.y_res = (self.y_area[1] - self.y_area[0]) / self.numDetector[0]
        self.I0 = self.get_I0(self.y_area, self.numDetector)
            
        
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
    
    def get_I0(self, y_area, y_size):
        if not self.use_I0:
            I0 = torch.ones((y_size[0], y_size[1]))
        else:
            col_data = np.loadtxt('/s4/xjxu/data/imgct/img3d/collimator/RMI_Collimator_ArealMass.dat')
            col_size = int(np.sqrt(len(col_data)))
            col_res = 0.0987  # cm

            x = np.linspace(-col_size//2 * col_res, col_size//2 * col_res, col_size)
            y = np.linspace(-col_size//2 * col_res, col_size//2 * col_res, col_size)
            z = col_data.reshape(col_size, col_size)   
            f = interpolate.interp2d(x, y, z, kind='linear')

            xnew = np.linspace(y_area[0], y_area[1], y_size[0])
            ynew = np.linspace(y_area[0], y_area[1], y_size[1])
            znew = f(xnew, ynew)
            
            metal_alpha = 0.0404
            I0 = torch.from_numpy(znew) * metal_alpha
            I0 = torch.exp(-I0)
        return I0
    
    def get_init(self, y, A, method='fbp'): # y: A H W
        if method == 'fbp':
            fbp_op = odl.tomo.fbp_op(A, filter_type='Hann', frequency_scaling=0.3)
            x_init = OperatorFunction.apply(fbp_op, y)
        elif method == 'gd':
            fbp_op = odl.tomo.fbp_op(A, filter_type='Hann', frequency_scaling=0.3)
            x_init = OperatorFunction.apply(fbp_op, y)
            x_init = torch.clamp(x_init, min=0, max=torch.inf) 
            gamma = 1e-3
            niter = 10
            for _ in range(niter):
                grad_d = self.grad(x_init, y)
                x_init = x_init - gamma * grad_d
        else:
            raise ValueError('Init method not found!')   
        x_init = torch.clamp(x_init, min=0, max=torch.inf) 
        return x_init 

    # def tomoCT(self, mea, A, I0, y_res,
    #             gamma_kernel, photon_kernel, detector_kernel, do_bst,  do_gpnoise,
    #             noise_type,  noise_level,
    #             init_method):
    #     # mea: nA nD nD 
    #     mea = torch.exp(-mea)
    #     mea = I0 * mea
    #     mea = data_process.addsca_torch(mea, gamma_kernel, photon_kernel, detector_kernel, res=y_res, do_bst=do_bst, do_gpnoise=do_gpnoise) 
    #     mea = data_process.addwgn_torch(mea, noise_type=noise_type, noise_level=noise_level)
    #     mea = -torch.log(mea/I0)
    #     mea = torch.clamp(mea, min=0, max=torch.inf)
    #     # ipt: nX nY nZ
    #     ipt = CBCTDF.get_init(mea, A, method=init_method)
    #     return mea, ipt
    
if __name__ == '__main__':
    thing = CBCTDF()
    device = torch.device('cuda')
    input = torch.rand(2, 448, 448, 448).to(device)
    output = thing.fmult(input, thing.A)
    print(output.shape)
    image = thing.ftran(output, thing.A)
    print(image.shape)