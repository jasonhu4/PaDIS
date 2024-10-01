import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from training.pos_embedding import Pos_Embedding
import scipy.io
from diffusers import AutoencoderKL
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import sys
sys.path.append('/n/badwater/z/jashu/Patch-Diffusion/odlstuff')
from fanbeam import *
from parbeam import *

torch.manual_seed(2)

def getIndices(spaced, patches, pad, psize, freezeindex = False):
    a = random.randint(0, pad-1)
    b = random.randint(0, pad-1)
    if freezeindex:
        a = 0
        b = 0
    indices = []
    for p in range(patches):
        for q in range(patches):
            indices.append([spaced[p]+a, spaced[p]+a+psize, spaced[q]+b, spaced[q]+b+psize])
    return indices

def random_patch(images, patch_size, resolution):
    device = images.device

    pos_shape = (images.shape[0], 1, patch_size, patch_size)
    x_pos = torch.ones(pos_shape)
    y_pos = torch.ones(pos_shape)
    x_start = np.random.randint(resolution - patch_size)
    y_start = np.random.randint(resolution - patch_size)

    x_pos = x_pos * x_start + torch.arange(patch_size).view(1, -1)
    y_pos = y_pos * y_start + torch.arange(patch_size).view(-1, 1)

    x_pos = (x_pos / resolution - 0.5) * 2.
    y_pos = (y_pos / resolution - 0.5) * 2.

    # Add x and y additional position channels
    images_patch = images[:, :, x_start:x_start + patch_size, y_start:y_start + patch_size]
    images_pos = torch.cat([x_pos.to(device), y_pos.to(device)], dim=1)

    return images_patch, images_pos

def makeFigures(noisy2, denoised2, orig2, i, imsize=256, pad=64):
    channels = len(denoised2[0,:,0,0])
    dir = '/n/badwater/z/jashu/Patch-Diffusion/inf_results/'
    denoised = torch.clone(denoised2)
    noisy = torch.clone(noisy2)
    orig = orig2.copy()
    if channels == 1:
        denoised = torch.cat([denoised]*3, dim=1)
        noisy = torch.cat([noisy]*3, dim=1)
        orig = np.concatenate([orig]*3, axis=2)
    elif channels == 2:
        denoised = torch.cat([denoised]*2, dim=1)
        noisy = torch.cat([noisy]*2, dim=1)
        orig = np.concatenate([orig]*2, axis=2)
        denoised = denoised[:,0:3,:,:]
        noisy = noisy[:,0:3,:,:]
        orig = orig[:,:,0:3]

    denoised = torch.squeeze(denoised)
    orig = np.squeeze(orig)
    noisy = torch.squeeze(noisy).cpu().numpy()

    #print(denoised.shape, noisy.shape, np.shape(orig))
    denoised = denoised[:,pad:imsize+pad,pad:imsize+pad]
    orig = orig[pad:imsize+pad,pad:imsize+pad,:]
    noisy = np.transpose(noisy, (1,2,0))
    denoised = denoised.detach().cpu().numpy()
    denoised = np.transpose(denoised, (1,2,0))

    noisy = np.clip(noisy, -1, 1)
    denoised = np.clip(denoised, -1,1)
    orig = np.clip(orig, -1, 1)
    mean_array = np.mean(denoised, axis=2, keepdims=True)
    denoised = np.repeat(mean_array, 3, axis=2)

    noisypsnr = psnr(noisy, orig, data_range=2)
    denoisedpsnr = psnr(denoised, orig, data_range=2)
    t1 = 'FBP recon'
    t2 = 'Patch recon'

    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1),plt.imshow(noisy/2+0.5, cmap='gray'),plt.axis('off'),plt.title(str(noisypsnr))
    plt.subplot(1,3,2),plt.imshow(denoised/2+0.5, cmap='gray'),plt.axis('off'),plt.title(str(denoisedpsnr))
    plt.subplot(1,3,3),plt.imshow(orig/2+0.5, cmap='gray'),plt.axis('off')

    plt.show()
    plt.savefig(dir + str(i) + '.png')
    plt.close('all')

def denoisedFromImage(net, x2, t_hat, psize=64, pad=64, patches=5, imsize=256, wrong = False, label=None):
    if wrong:
        x = x2
    else:
        x = torch.clone(x2)
    assert len(x[0,0,0,:]) == imsize
    spaced = np.linspace(0, (patches-1)*psize, patches, dtype=int)
    indices = getIndices(spaced, patches, pad, psize)
    z = torch.nn.functional.pad(x, (pad, pad, pad, pad), "constant", -1)
    x = 0 + t_hat*torch.randn_like(z)
    x[:,:,pad:imsize+pad, pad:imsize+pad] = z[:,:,pad:imsize+pad, pad:imsize+pad]

    x_start = 0
    y_start = 0
    image_size = imsize + 2*pad
    x_pos = torch.arange(x_start, x_start+image_size).view(1, -1).repeat(image_size, 1)
    y_pos = torch.arange(y_start, y_start+image_size).view(-1, 1).repeat(1, image_size)
    x_pos = (x_pos / (image_size - 1) - 0.5) * 2.
    y_pos = (y_pos / (image_size - 1) - 0.5) * 2.
    latents_pos = torch.stack([x_pos, y_pos], dim=0).to(torch.device('cuda'))
    latents_pos = latents_pos.unsqueeze(0).repeat(1, 1, 1, 1)
    out = denoisedFromPatches(net, x, t_hat, latents_pos, label, indices, t_goal=0)
    return out[:,:,pad:imsize+pad, pad:imsize+pad]

def denoiseMultiscale(net, x, t_hat, lpos, class_labels):
    x_hat = torch.clone(x)
    latents_pos = torch.clone(lpos)
    N = len(x_hat[0,0,0,:])
    channels = len(x_hat[0,:,0,0])
    psize=64
    pad = psize
    patches = 4
    scale = patches
    x_hat = x_hat[:,:, psize:N-psize, psize:N-psize]
    latents_pos = latents_pos[:,:, psize:N-psize, psize:N-psize]

    output = torch.zeros_like(x_hat)
    x_input = torch.zeros(patches*patches, channels, psize, psize).to(torch.device('cuda'))
    pos_input = torch.zeros(patches*patches, 2, psize, psize).to(torch.device('cuda'))

    k = 0
    for i in range(patches):
        for j in range(patches):
            indices_row = torch.arange(i, i + psize*scale, scale)
            indices_col = torch.arange(j, j + psize*scale, scale)
            xgrid, ygrid = torch.meshgrid(indices_row, indices_col, indexing='ij')
            x_input[k,:,:,:] = x_hat[0,:, xgrid, ygrid] #should have size 3 64 64 on the RHS
            pos_input[k,:,:,:] = latents_pos[0, :, xgrid, ygrid]
            k = k+1
    bigout = net(x_input, t_hat, pos_input, class_labels).to(torch.float64)

    k = 0
    for i in range(patches):
        for j in range(patches):
            indices_row = torch.arange(i, i + psize*scale, scale)
            indices_col = torch.arange(j, j + psize*scale, scale)
            xgrid, ygrid = torch.meshgrid(indices_row, indices_col, indexing='ij')
            output[0,:, xgrid, ygrid] = bigout[k,:,:,:]
            k = k+1
    return torch.nn.functional.pad(output, (pad, pad, pad, pad), "constant", 0)

#from stitching unet paper
def denoisedTile(net, x, t_hat, latents_pos, class_labels, pad=24, psize=56, overlap = 8, t_goal = -1):
    x_hat = torch.clone(x)
    channels = len(x_hat[0,:,0,0])
    N = len(x_hat[0,0,0,:])
    inds = [4] #this number can be adjusted
    skip = psize-overlap
    while inds[-1] < N-pad-psize:
        inds.append(inds[-1] + skip)
    patches = len(inds)
    indices = getIndices(inds, patches, pad, psize, freezeindex = True)
    patches = len(indices) #now is the square of the prev variable
    lastind = inds[-1] + psize
    assert lastind < N

    output = torch.zeros_like(x_hat)
    x_input = torch.zeros(patches, channels, psize, psize).to(torch.device('cuda'))
    pos_input = torch.zeros(patches, 2, psize, psize).to(torch.device('cuda'))

    for i in range(patches):
        z = indices[i]
        #print(z)
        x_input[i,:,:,:] = torch.squeeze(x_hat[0,:,z[0]:z[1], z[2]:z[3]])
        pos_input[i,:,:,:] = torch.squeeze(latents_pos[:,:,z[0]:z[1], z[2]:z[3]])
    bigout = net(x_input, t_hat, pos_input, class_labels).to(torch.float64)

    M = torch.zeros_like(x_hat) #array for counting how many times each patch is counted
    for i in range(patches):
        z = indices[i]
        output[0,:,z[0]:z[1], z[2]:z[3]] = bigout[i,:,:,:]

    temp = t_goal + torch.randn_like(x_hat) * t_goal
    temp[:,:,pad:N-pad, pad:N-pad] = output[:,:,pad:N-pad, pad:N-pad]
    return temp

#from weather denoising paper
def denoisedOverlap(net, x, t_hat, latents_pos, class_labels, pad=24, psize=56, overlap = 8, t_goal = -1):
    x_hat = torch.clone(x)
    channels = len(x_hat[0,:,0,0])
    N = len(x_hat[0,0,0,:])
    inds = [pad]
    skip = psize-overlap
    while inds[-1] < N-pad-psize:
        inds.append(inds[-1] + skip)
    patches = len(inds)
    indices = getIndices(inds, patches, pad, psize, freezeindex = True)
    patches = len(indices) #now is the square of the prev variable
    lastind = inds[-1] + psize
    assert lastind < N

    output = torch.zeros_like(x_hat)
    x_input = torch.zeros(patches, channels, psize, psize).to(torch.device('cuda'))
    pos_input = torch.zeros(patches, 2, psize, psize).to(torch.device('cuda'))

    for i in range(patches):
        z = indices[i]
        x_input[i,:,:,:] = torch.squeeze(x_hat[0,:,z[0]:z[1], z[2]:z[3]])
        pos_input[i,:,:,:] = torch.squeeze(latents_pos[:,:,z[0]:z[1], z[2]:z[3]])
    bigout = net(x_input, t_hat, pos_input, class_labels).to(torch.float64)

    M = torch.zeros_like(x_hat) #array for counting how many times each patch is counted
    for i in range(patches):
        z = indices[i]
        output[0,:,z[0]:z[1], z[2]:z[3]] += bigout[i,:,:,:]
        M[0,:,z[0]:z[1], z[2]:z[3]] += 1
    output[0,:,pad:lastind, pad:lastind] = torch.div(output[0,:,pad:lastind, pad:lastind], M[0,:,pad:lastind, pad:lastind])

    temp = t_goal + torch.randn_like(x_hat) * t_goal
    temp[:,:,pad:N-pad, pad:N-pad] = output[:,:,pad:N-pad, pad:N-pad]
    return temp

def denoisedFromPatches(net, x, t_hat, latents_pos, class_labels, indices, t_goal = -1, avg=1, spaced=[], wrong=False):
    if len(spaced) > 1:
        indices = getIndices(spaced, 5, 24, 56)
    if wrong:
        x_hat = x
    else:
        x_hat = torch.clone(x)
    channels = len(x_hat[0,:,0,0])
    N = len(x_hat[0,0,0,:])
    psize = indices[0][1] - indices[0][0]
    patches = len(indices)
    pad = int((N - np.sqrt(patches)*psize))

    output = torch.zeros_like(x_hat)
    x_input = torch.zeros(patches, channels, psize, psize).to(torch.device('cuda'))
    pos_input = torch.zeros(patches, 2, psize, psize).to(torch.device('cuda'))

    for i in range(patches):
        z = indices[i]
        x_input[i,:,:,:] = torch.squeeze(x_hat[0,:,z[0]:z[1], z[2]:z[3]])
        pos_input[i,:,:,:] = torch.squeeze(latents_pos[:,:,z[0]:z[1], z[2]:z[3]])
    bigout = net(x_input, t_hat, pos_input, class_labels).to(torch.float64)

    for i in range(patches):
        z = indices[i]
        x_patch = x_hat[0,:,z[0]:z[1], z[2]:z[3]]
        output[0,:,z[0]:z[1], z[2]:z[3]] += bigout[i,:,:,:]
        output[0,:,z[0]:z[1], z[2]:z[3]] -= x_patch
    x_hat = x_hat + output

    temp = t_goal + torch.randn_like(x_hat) * t_goal
    temp[:,:,pad:N-pad, pad:N-pad] = x_hat[:,:,pad:N-pad, pad:N-pad]
    return temp
