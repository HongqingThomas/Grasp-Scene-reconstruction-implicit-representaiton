import numpy as np
import torch
import torch.nn.functional as F
import cv2

def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, depth, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth is the corresponding image tensor.

    """
    i, j, sample_depth = get_sample_uv(H0, H1, W0, W1, n, depth, device=device)
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device)
    return rays_o, rays_d, sample_depth

def get_sample_uv(H0, H1, W0, W1, n, depth, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    #TODO: Important
    depth = depth[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(
        W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
    
    i = i.t()  # transpose
    j = j.t()
    i, j, depth = select_uv(i, j, n, depth, device=device)
    return i, j, depth

def get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(-1, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def select_uv(i, j, n, depth, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n)
    j = j[indices]  # (n)
    depth = depth.reshape(-1)
    depth = depth[indices]  # (n)
    return i, j, depth