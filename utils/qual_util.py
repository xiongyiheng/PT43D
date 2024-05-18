import cv2
import numpy as np
import imageio
from PIL import Image
from einops import rearrange
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms

from pytorch3d import structures

from .util_3d import sdf_to_mesh, render_mesh, rotate_mesh_360


def make_batch(data, B=16):
    x = data['sdf']
    x_idx = data['idx']
    z_q = data['z_q']
    bs = x.shape[1]
    if bs > B:
        return data

    data['sdf'] = x.repeat(B//bs, 1, 1, 1, 1)
    data['idx'] = x_idx.repeat(B//bs, 1, 1, 1)
    data['z_q'] = z_q.repeat(B//bs, 1, 1, 1, 1)
    return data

def save_mesh_as_gif(mesh_renderer, mesh, nrow=3, out_name='1.gif'):
    """ save batch of mesh into gif """

    # img_comb = render_mesh(mesh_renderer, mesh, norm=False)    

    # rotate
    rot_comb = rotate_mesh_360(mesh_renderer, mesh) # save the first one
    
    # gather img into batches
    nimgs = len(rot_comb)
    nrots = len(rot_comb[0])
    H, W, C = rot_comb[0][0].shape
    rot_comb_img = []
    for i in range(nrots):
        img_grid_i = torch.zeros(nimgs, H, W, C)
        for j in range(nimgs):
            img_grid_i[j] = torch.from_numpy(rot_comb[j][i])
            
        img_grid_i = img_grid_i.permute(0, 3, 1, 2)
        img_grid_i = vutils.make_grid(img_grid_i, nrow=nrow)
        img_grid_i = img_grid_i.permute(1, 2, 0).numpy().astype(np.uint8)
            
        rot_comb_img.append(img_grid_i)
    
    with imageio.get_writer(out_name, mode='I', duration=.08) as writer:
        
        # combine them according to nrow
        for rot in rot_comb_img:
            writer.append_data(rot)
   
# copy from quant/test_iou.py
def get_img_prob(resnet2vq_model, test_data, opt=None):
    img = test_data['img'].to(resnet2vq_model.opt.device)
    
    img_logits = resnet2vq_model(img) # bs c d h w

    # logsoftmax
    img_logprob = F.log_softmax(img_logits, dim=1) # compute the prob. of next ele
    # img_logprob = torch.sum(img_logprob, dim=1) # multiply the image priors
    img_logprob = rearrange(img_logprob, 'bs c d h w -> (d h w) bs c')

    # ret = img_prob
    return img_logprob