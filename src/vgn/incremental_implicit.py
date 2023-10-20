import time

import numpy as np
import trimesh
from scipy import ndimage
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
torch.autograd.set_detect_anomaly(True)

#from vgn import vis
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
from vgn.networks import load_network
from vgn.utils import visual
from vgn.utils.implicit import as_mesh

# from render_utils import get_samples
# from render import Renderer

LOW_TH = 0.5

class Incremental_VGNImplicit(object):
    def __init__(self, model_path, model_type, best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False, resolution=40, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device, model_type=model_type)
        for name, para in enumerate(self.net.parameters()):
            if not para.requires_grad:
                print("no grad:", name)

        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        
        self.resolution = resolution
        x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution))
        # 1, self.resolution, self.resolution, self.resolution, 3
        pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)
        self.pos = pos.view(1, self.resolution * self.resolution * self.resolution, 3)
        # self.render = Renderer()
        self.N_samples = 8
        self.N_surface = 8
        self.points_batch_size = 8
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.incremental_iter_nums = 3

    def __call__(self, state,  depth_img_list=None, intrinsic_list=None, extrinsic_list=None, aff_kwargs={}):
        if hasattr(state, 'tsdf_process'):
            tsdf_process = state.tsdf_process
        else:
            tsdf_process = state.tsdf

        if isinstance(state.tsdf, np.ndarray):
            tsdf_vol = state.tsdf
            voxel_size = 0.3 / self.resolution
            size = 0.3
        else:
            tsdf_vol = state.tsdf.get_grid()
            voxel_size = tsdf_process.voxel_size
            tsdf_process = tsdf_process.get_grid()
            size = state.tsdf.size

        tic = time.time()
        #TODO:new: tsdf_vol: inputs (tensor): conditioning input, B*N*3
        # pos: sampled points, B*N*C, 更改self.pos

        # rendering part
        depth_img = torch.from_numpy(depth_img_list[0]).to(self.device)

        intrinsic_matrix = intrinsic_list[0]
        extrinsic_matrix = extrinsic_list[0]
        H, W = depth_img.shape[0], depth_img.shape[1]
        # print("H, W:", H, W)
        fx, fy, cx, cy = intrinsic_matrix[0][0], intrinsic_matrix[1][1], intrinsic_matrix[0][2], intrinsic_matrix[1][2]
        sampling_size = 10
        Hedge,Wedge = int(H/2 - 30), int(W/2 - 30)
        c2w = np.linalg.inv(extrinsic_matrix)
        # c2w = extrinsic_matrix
        c2w[2, :] *= -1
        self.bound = torch.from_numpy(np.array([[-0.5,0.5],[-0.5,0.5], [-0.5,0.5]])).to(self.device)
        # self.bound = torch.from_numpy(np.array([[0,0.05 * 6],[0,0.05 * 6], [0, 0.05 * 6]])).to(self.clearsssssssdevice)
        for i in range(self.incremental_iter_nums):
            print("iter:", i)
            # print("c2w:", c2w)
            batch_rays_o, batch_rays_d, batch_gt_depth = get_samples(Hedge, H-Hedge, Wedge, W-Wedge, sampling_size, H, W, fx, fy, cx, cy, c2w, depth_img, self.device)
            # print("batch_rays_o:", batch_rays_o[0])
            # print("batch_rays_d:", batch_rays_d.shape, batch_rays_d)
            # print("batch_gt_depth:", batch_gt_depth.shape, batch_gt_depth.min())
            with torch.no_grad():

                print("*******************************************************************")
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                # has some error
                # t = (self.bound.unsqueeze(0).to(self.device)-det_rays_o)/det_rays_d
                t = -(self.bound.unsqueeze(0).to(self.device)-det_rays_o)/det_rays_d
                # print("t':", t)
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                # print("t:", t)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            # print("batch_rays_o:", batch_rays_o.shape, )
            # print("batch_rays_d:", batch_rays_d.shape, )
            # print("batch_gt_depth:", batch_gt_depth.shape)
            # print("batch_rays_d grad:",batch_rays_d.requires_grad)
            # print("batch_rays_o grad:",batch_rays_o.requires_grad)
            depth, uncertainty = self.render_batch_ray(
                batch_rays_d, batch_rays_o,  self.device, stage='color',  gt_depth=batch_gt_depth, tsdf_vol=tsdf_vol)
            # qual_vol, rot_vol, width_vol, occ = predict(tsdf_vol, self.pos, pos_occ, self.net, self.device)
            # print("depth grad:",depth.requires_grad)
            # print("uncertainty grad:", uncertainty.requires_grad)
            uncertainty = uncertainty.detach()
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)

            loss = (torch.abs(batch_gt_depth-depth) /
                    torch.sqrt(uncertainty+1e-10))[mask].mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


    def render_batch_ray(self, rays_d, rays_o, device, stage, gt_depth, tsdf_vol):
        N_samples = self.N_samples
        N_surface = self.N_surface
        N_rays = rays_o.shape[0]
        # print("rays_d:", rays_d.shape, rays_d)
        # print("rays_o:", rays_o.shape, rays_o)
        # print("gt_depth:", gt_depth.shape)

        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth_samples = gt_depth.repeat(1, N_samples)
        near = gt_depth_samples*0.01

        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(device) -
                 det_rays_o)/det_rays_d  # (N, 3, 2)

            # print("t shape:",t.shape, "t", t)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01
        
        if gt_depth is not None:
            # in case the bound is too large
            far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2))
        else:
            far = far_bb
        if N_surface > 0:
            gt_none_zero_mask = gt_depth > 0
            gt_none_zero = gt_depth[gt_none_zero_mask]
            gt_none_zero = gt_none_zero.unsqueeze(-1)
            gt_depth_surface = gt_none_zero.repeat(1, N_surface)
            t_vals_surface = torch.linspace(
                0., 1., steps=N_surface).double().to(device)
            # emperical range 0.05*depth
            z_vals_surface_depth_none_zero = 0.95*gt_depth_surface * \
                (1.-t_vals_surface) + 1.05 * \
                gt_depth_surface * (t_vals_surface)
            z_vals_surface = torch.zeros(
                gt_depth.shape[0], N_surface).to(device).double()
            gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
            z_vals_surface[gt_none_zero_mask,
                            :] = z_vals_surface_depth_none_zero
            near_surface = 0.001
            far_surface = torch.max(gt_depth)
            z_vals_surface_depth_zero = near_surface * \
                (1.-t_vals_surface) + far_surface * (t_vals_surface)
            z_vals_surface_depth_zero.unsqueeze(
                0).repeat((~gt_none_zero_mask).sum(), 1)
            z_vals_surface[~gt_none_zero_mask,
                            :] = z_vals_surface_depth_zero
        
        t_vals = torch.linspace(0., 1., steps=N_samples, device=device)

        z_vals = near * (1.-t_vals) + far * (t_vals)

        if N_surface > 0:
            z_vals, _ = torch.sort(
                torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]
        pointsf = pts.reshape(-1, 3)
        # print("pointsf:", pointsf)

        p_split = torch.split(pointsf, self.points_batch_size)
        bound = self.bound
        rets = []
        for pi in p_split:
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z
            # print("mask:", mask)

            pi = pi.unsqueeze(0)
            # ret = decoders(pi, c_grid=c, stage=stage)
            _, _, _, occ = predict(tsdf_vol, self.pos, pi, self.net, self.device)
            # print("occ requiregrad:", occ.requires_grad)
            # ret = ret.squeeze(0)
            # if len(ret.shape) == 1 and ret.shape[0] == 4:
            #     ret = ret.unsqueeze(0)

            # ret[~mask, 3] = 100
            # print("occ_predict:", occ.shape)
            occ[0, ~mask] = 100
            rets.append(occ)
        #TODO:可能需要修改
        print("rets:", rets)
        raw = torch.cat(rets, dim=0)
        # print("rets:", rets)
        raw = raw.reshape(N_rays, N_samples+N_surface, -1)
        print("raw:", raw)
        depth, uncertainty= raw2outputs_nerf_color(
                raw, z_vals, rays_d, device=device)
        return depth, uncertainty


def raw2outputs_nerf_color(raw, z_vals, rays_d, device='cuda:0'):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, N_rays*N_samples*1): prediction from model.
        z_vals (tensor, N_rays*N_samples): integration time.
        rays_d (tensor, N_rays*3): direction of each ray.
        occupancy (bool, optional): occupancy or volume density. Defaults to False.
        device (str, optional): device. Defaults to 'cuda:0'.

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty.
        rgb_map (tensor, N_rays*3): estimated RGB color of a ray.
        weights (tensor, N_rays*N_samples): weights assigned to each sampled color.
    """

    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-act_fn(raw)*dists)
    
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
        device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    # different ray angle corresponds to different unit length
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    # color = raw[..., :-1]
    alpha = raw2alpha(raw[..., -1], dists)  # (N_rays, N_samples)

    weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
        device).float(), (1.-alpha + 1e-10).float()], -1).float(), -1)[:, :-1]
    # color_map = torch.sum(weights[..., None] * color, -2)  # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
    tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
    depth_var = torch.sum(weights*(tmp)**2, dim=1)  # (N_rays)
    return depth_map, depth_var




# def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
#     # avoid grasp out of bound [0.02  0.02  0.055]
#     x_lim = int(limit[0] / voxel_size)
#     y_lim = int(limit[1] / voxel_size)
#     z_lim = int(limit[2] / voxel_size)
#     qual_vol[:x_lim] = 0.0
#     qual_vol[-x_lim:] = 0.0
#     qual_vol[:, :y_lim] = 0.0
#     qual_vol[:, -y_lim:] = 0.0
#     qual_vol[:, :, :z_lim] = 0.0
#     return qual_vol


def predict(tsdf_vol, pos, pos_occ, net, device):
    # TODO: 新增了occ_pos, 输出occ
    assert tsdf_vol.shape == (1, 40, 40, 40)

    # move input to the GPU
    tsdf_vol = torch.from_numpy(tsdf_vol).to(device)

    # forward pass
    # TODO: add a incremental training part
    # with torch.no_grad():
        #TODO:new: tsdf_vol: inputs (tensor): conditioning input, B*N*3
        # pos: sampled points, B*N*C
    qual_vol, rot_vol, width_vol, occ_pred= net(tsdf_vol, pos, 
                                        p_tsdf=pos_occ)
    occ_copy = occ_pred.clone()
    # print("occ_pred:", occ_pred.shape, occ_pred)
    # occ = torch.sigmoid(occ_copy)
    # occ = torch.exp(-1 * occ_copy)
    # occ = occ_copy.sigmoid()
    occ_copy[0,:] = torch.sigmoid(occ_pred[0, :])
    # print("occ:", occ.shape, occ)
    return qual_vol, rot_vol, width_vol, occ_copy


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
    # print("c2w:", c2w)
    # print("***************************************")
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