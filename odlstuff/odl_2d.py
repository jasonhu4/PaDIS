"""
# University of Michigan, Jeffrey Fessler's Group
"""

import numpy as np
import odl
import torch

import os
from os.path import dirname, abspath
from scipy.io import savemat

##################################################
# init the gpu usages
##################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#############################################
# init settings
#############################################
obj_size = [200, 200]
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=obj_size,
    dtype='float32')
angle_partition = odl.uniform_partition(-np.pi/360, -np.pi/360+2 * np.pi, 360)
detector_partition = odl.uniform_partition(-40, 40, 512)
geometry = odl.tomo.FanBeamGeometry(
    angle_partition, detector_partition, src_radius=40, det_radius=40)

ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Shepp-Logan', frequency_scaling=0.8)
phantom_ori = odl.phantom.shepp_logan(reco_space, modified=True)
phantom_ori = reco_space.element(torch.rand(200,200))
#print(phantom_ori.shape)
#phantom_ori = np.stack([phantom_ori]*3, axis=2)
#phantom_ori = torch.from_numpy(phantom_ori)

#############################################
# run projection / BP / FBP
#############################################
proj_ori = ray_trafo(phantom_ori)
bp_recon_ori = ray_trafo.adjoint(proj_ori)
fbp_recon_ori = fbp(proj_ori)
err_fbp_ori = fbp_recon_ori - phantom_ori

print(type(proj_ori.asarray()))
print(type(bp_recon_ori.asarray()))

#############################################
# show images
#############################################
phantom_ori.show(title='Phantom')
proj_ori.show(title='Simulated Data (Sinogram)')
bp_recon_ori.show(title='Back-projection (Sinogram)')
fbp_recon_ori.show(title='Filtered Back-projection')
err_fbp_ori.show(title='Error', force_show=True)

#############################################
# save data
#############################################
save_dir = dirname(abspath(__file__))
savemat(f'{save_dir}/odl_results2.mat',
        {'phantom':phantom_ori.asarray(),
         'proj':proj_ori.asarray(),
         'bp_recon':bp_recon_ori.asarray()})

#############################################
# log
#############################################
print(f'Experiment done, data saved to {save_dir}')
