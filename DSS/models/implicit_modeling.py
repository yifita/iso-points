from typing import Optional, List
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch3d.structures import packed_to_list
from pytorch3d.renderer.cameras import CamerasBase
import time
import trimesh
import numpy as np
from skimage import measure
from matplotlib import cm
import matplotlib.colors as mpc
from .. import logger_py, get_debugging_mode, get_debugging_tensor
from . import BaseGenerator, ModelReturns
from .levelset_sampling import (DirectionalSamplingNetwork, SphereTracing, RayTracing,
                                UniformProjection, SampleNetwork,
                                find_zero_crossing_between_point_pairs)
from ..utils.mathHelper import decompose_to_R_and_t
from ..misc.visualize import plot_cuts
from ..core.cloud import PointClouds3D
from ..utils import (intersection_with_unit_cube,
                     intersection_with_unit_sphere,
                     gather_batch_to_packed,
                     valid_value_mask,
                     get_tensor_values,
                     mask_packed_to_list,
                     get_surface_high_res_mesh,
                     make_image_grid)


class Model(nn.Module):
    ''' DVR model class.

    Args:
        decoder (nn.Module): decoder network (
            occupancy and (or) rgb prediction)
        shader (nn.Module): gets per point color using ambient+diffuse+specular
        renderer (nn.Module): additional point renderer (e.g. DSS)
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self,
                 decoder: nn.Module,
                 texture: Optional[nn.Module] = None,
                 encoder: Optional[nn.Module] = None,
                 cameras: Optional[CamerasBase] = None,
                 device: Optional[torch.device] = 'cpu',
                 exact_gradient: bool = True,
                 approx_grad_step: float = 1e-2,
                 n_points_per_ray: int = 100,
                 max_points_per_pass: int = 160000,
                 proj_max_iters: int = 10,
                 proj_tolerance: float = 5e-5,
                 object_bounding_sphere: float = 1.0,
                 **kwargs
                 ):
        super().__init__()
        self.decoder = decoder.to(device=device)
        self.encoder = encoder
        if self.encoder is not None:
            self.encoder = self.encoder.to(device=device)
        self.texture = texture
        self.cameras = cameras
        self.device = device
        self.object_bounding_sphere = object_bounding_sphere
        self.hooks = []

        self.exact_gradient = exact_gradient
        self.approx_grad_step = approx_grad_step
        # sampled iso-surface points in the space
        self.projection = UniformProjection(
            max_points_per_pass=max_points_per_pass, proj_max_iters=proj_max_iters, proj_tolerance=proj_tolerance,
            **kwargs
        )

        self.sphere_tracing = SphereTracing(
            max_points_per_pass=max_points_per_pass, proj_max_iters=proj_max_iters,
            proj_tolerance=proj_tolerance, alpha=1.0)
        self.ray_tracing = RayTracing(object_bounding_sphere=self.object_bounding_sphere,
                                      sdf_threshold=proj_tolerance,
                                      line_search_step=0.5,
                                      line_step_iters=3,
                                      sphere_tracing_iters=10,
                                      n_steps=100,
                                      n_secant_steps=8,)
        self.directional_sampling = DirectionalSamplingNetwork()
        self.sampling = SampleNetwork()

        self.n_points_per_ray = n_points_per_ray
        self.max_points_per_pass = max_points_per_pass

    def decode_color(self, pointclouds: PointClouds3D, c=None, **kwargs) -> PointClouds3D:
        """
        Get the colors of the sampled points in the world
        Let's not use neural attributes yet
        """
        c_dim = 0 if c is None else c.shape[-1]
        if pointclouds.isempty():
            return pointclouds
        if hasattr(self.texture, 'decoder') and self.texture.decoder.c_dim > c_dim:
            output = self.decode(pointclouds.points_packed(), c=c, **kwargs)
            z = output.latent
            if c is not None:
                c = torch.cat([c, z], dim=-1)
            else:
                c = z

        colored_pointclouds = self.texture(pointclouds, c=c, **kwargs)
        return colored_pointclouds

    def decode(self, p, c=None, **kwargs):
        '''
        Returns sdf values for the sampled points.

        Args:
            p (tensor): points (N, *, 3)
            c (tensor): latent conditioned code c (N, c_dim)
        Returns:
            net_output (namedtuple): sdf (N, *, 1)
        '''
        if c is not None and c.nelement() > 0:
            assert(c.ndim == 2)
            assert(c.shape[0] == p.shape[0])
            c = c.view(c.shape[0] + [1] * max(p.ndim() - 2, 0) + [c.shape[1]])

        sdf = self.decoder(p, c=c, **kwargs)
        return sdf

    def get_point_clouds(self, points: Optional[torch.Tensor] = None,
                         mask: Optional[torch.Tensor] = None,
                         c: Optional[torch.Tensor] = None,
                         with_colors=False, with_normals=True,
                         require_normals_grad=False,
                         project=False, debug_name_prefix='',
                         **kwargs):
        """
        Returns the point clouds object from given points and points mask
        Args:
            p: (B,P,3) or (P,3) points in 3D source space (default = self.points)
            mask: (B,P) or (P) mask for valid points
            c: (B,C) latent code
            with_colors: use shader or neural shader to get colored points
            with_normals: use decoder gradient to get the normal vectors
        """
        assert(points.ndim == 3 and points.shape[2] == 3)
        points = points.to(device=self.device)

        if project:
            if mask is not None:
                points_list = torch.split(points[mask], mask.sum(-1).tolist())
                point_clouds = PointClouds3D(points_list)
                proj_results = self.projection.project_points(
                    point_clouds, self.decoder, latent=c)
            else:
                proj_results = self.projection.project_points(
                    points, self.decoder, latent=c)

            points = proj_results['levelset_points']
            mask_iso = proj_results['mask']

            if next(self.parameters()).requires_grad:
                points = self.sampling(self.decoder,
                                       proj_results['levelset_points'].detach(),
                                       )

            mask = mask_iso

        normals = None
        colors = torch.ones_like(points).detach()

        if mask is not None:
            points = points[mask]
            colors = colors[mask]
            points_list = list(torch.split(
                points, mask.sum(dim=1).tolist(), dim=0))
            colors_list = list(torch.split(
                colors, mask.sum(dim=1).tolist(), dim=0))
        else:
            points_list = list(torch.unbind(points, dim=0))
            colors_list = list(torch.unbind(colors, dim=0))

        # use placeholder normals, as they will be updated
        pointclouds = PointClouds3D(
            points_list, features=colors_list, normals=colors_list)

        if c is not None:
            c = gather_batch_to_packed(
                c, pointclouds.packed_to_cloud_idx())

        if with_normals:
            def save_grad(name):
                num_points_per_cloud = pointclouds.num_points_per_cloud().tolist()

                def _save_grad(grad):
                    # NOTE: for iso_normal, this gradient is always zero (WHY?)
                    dbg_tensor = get_debugging_tensor()
                    # a dict of list of tensors
                    grad = packed_to_list(grad, num_points_per_cloud)
                    dbg_tensor.pts_world_grad[name] = [
                        grad[b].detach().cpu() for b in range(len(grad))]
                    if dbg_tensor.pts_world_grad[name][0].shape != dbg_tensor.pts_world[name][0].shape:
                        import pdb
                        pdb.set_trace()

                return _save_grad

            points_for_normals = pointclouds.points_packed()
            if c is not None:
                assert(
                    c.ndim == 2 and points_for_normals.shape[0] == c.shape[0])

            normals = self.get_normals_from_grad(
                points_for_normals, c=c, requires_grad=require_normals_grad,
                **kwargs)
            assert(normals.requires_grad == require_normals_grad)

            assert(not torch.isnan(normals).any()
                   and torch.isfinite(normals).all())
            valid_normals = valid_value_mask(normals).all(
                dim=-1) & (torch.norm(normals, dim=-1) > 0)
            if (torch.norm(normals, dim=-1) == 0).any():
                # norma can be zero
                logger_py.warn('Detected zero length normal')

            # remove this point
            normals_list = mask_packed_to_list(
                normals.view(-1, 3), pointclouds.num_points_per_cloud(), valid_normals)
            pointclouds = PointClouds3D(
                points=pointclouds.points_list(), normals=normals_list)

            if get_debugging_mode():
                normals.requires_grad_(True)
                dbg_tensor = get_debugging_tensor()
                points_list = pointclouds.points_list()
                dbg_tensor.pts_world[debug_name_prefix + 'normal'] = [
                    x.cpu().detach() for x in points_list]
                handle = normals.register_hook(
                    save_grad(debug_name_prefix + 'normal'))
                self.hooks.append(handle)

        if with_colors:
            pointclouds = self.decode_color(pointclouds, c=c, **kwargs)

        return pointclouds

    def get_normals_from_grad(self, p_world, c=None, requires_grad=False, return_sdf=False, **kwargs):
        """
        Returns the not normalized normals at the query points by differentiating
        the implicit function w.r.t the input points

        Args:
            p_world (tensor): [N,*,3] points in world coordinates
            c (tensor): latent conditioned code [N,*, C]
        """
        p_has_grad = p_world.requires_grad

        if c is not None and c.numel() != 0 and c.shape[0] != p_world.shape[0]:
            assert(c.shape[0] != p_world.shape[0])

        with autograd.enable_grad():
            p_world.requires_grad_(True)
            point_dim = p_world.shape[-1]
            sdf = self.decode(p_world, c=c, **kwargs).sdf
            p_normals = autograd.grad(
                sdf, p_world, torch.ones_like(sdf), create_graph=requires_grad)[0]

        p_world.requires_grad_(p_has_grad)
        if not requires_grad:
            p_normals = p_normals.detach()
        assert(p_normals.requires_grad == requires_grad)
        if return_sdf:
            return p_normals, sdf
        return p_normals

    def pixels_to_world(self, pixels, cameras, c=None, it=0, **kwargs):
        """
        Modified DVR implementation to find intersection via sphere-tracing.
        Ray-trace from both front and back like in IDR
        Args:
            pixels (B, P, 2) pixels with normalized coordinates (top-left is (-1, -1))
                bottom right is (-1, -1)
            cameras (BaseCameras)
            c: latent code (B,*,C)
        Returns:
            p_world (B, P, 3) ray-traced points
            mask_pred (B, P): mask for successful ray-tracing
        """
        with autograd.no_grad():
            # First find initial ray0 as the intersection with the unit sphere
            cam_pos = cameras.get_camera_center()
            cam_ray = cameras.unproject_points(torch.cat(
                [-pixels, pixels.new_ones(pixels.shape[:-1] + (1,))], dim=-1), scaled_depth_input=False) - \
                cam_pos.unsqueeze(1)
            cam_ray = F.normalize(cam_ray, dim=-1, p=2)
            # This returns an intersection between the two tangent plane if the ray
            # doesn't intersects with the sphere
            section0, section1, has_intersection = intersection_with_unit_cube(
                cam_pos.unsqueeze(1), cam_ray, side_length=self.object_bounding_sphere*2)

            # Sphere tracing from the first unit-sphere intersection
            proj_result = self.sphere_tracing.project_points(
                section0, cam_ray, self.decoder, latent=c)
            mask_pred = proj_result['mask'] & has_intersection
            p_world = proj_result['levelset_points'].detach()
            p_world[~mask_pred] = section0[~mask_pred]

            # For failed sphere-tracing, attempt to find the intersection via
            # secant sign change (like DVR)
            proj_result_back = self.sphere_tracing.project_points(
                section1, -cam_ray, self.decoder, latent=c)
            p_world_back = proj_result_back['levelset_points'].detach()
            mask_pred_back = proj_result_back['mask']

            iso_secant, mask_pred_secant = find_zero_crossing_between_point_pairs(
                p_world, p_world_back, self.decoder, c=c, is_occupancy=False)
            mask_pred_secant = (~mask_pred) & mask_pred_secant
            mask_pred = mask_pred_secant | mask_pred

            # merge two partions of iso intersections
            p_world = torch.where(
                mask_pred_secant.unsqueeze(-1), iso_secant, p_world)

            with autograd.enable_grad():
                p_world = p_world.requires_grad_(True)
                p_world_val = self.decoder(p_world, c=c).sdf
                p_world_Dx = autograd.grad([p_world_val], [p_world], torch.ones_like(
                    p_world_val), retain_graph=True)[0]

                # filter out p_world whose Dx is almost perpendicular
                mask = torch.sum(F.normalize(p_world_Dx, dim=-1) * cam_ray, dim=-1) < -1e-2
                mask_pred = mask_pred & mask

        if self.training:
            p_world, _ = self.directional_sampling(self.decoder, p_world,
                                                cam_ray, cam_pos.unsqueeze(1), return_eval=True)

        return p_world, mask_pred

    def sample_world_points(self, p, cameras, n_points_per_ray, mask_gt, mask_pred, c=None):
        """
        get freepsace and occupancy points in source coordinates (-1.0, 1.0)
        Args:
            p (tensor): (N, n_rays, 2)
            n_points_per_ray: number of 3D points per viewing ray
            mask_gt: (N, n_rays)
            mask_pred: (N, n_rays)
        Returns:
            p_freespace (N1*n_points_per_ray, 3)
            mask_freespace (N, n_rays, n_points_per_ray)
            p_occupancy (N2, 3)
            mask_occupancy (N, n_rays)
        """
        batch_size, P = p.shape[:2]
        max_points = self.max_points_per_pass
        packed_to_cloud_idx = torch.arange(
            batch_size).view(-1, 1).expand(batch_size, P).to(device=p.device)
        with autograd.no_grad():
            # 0. invalid points
            iso_incamera = (p[..., :2].abs() <= 1.0).all(dim=-1)
            # 1. find intersect with unit sphere, returns (N, n_rays, 3) and (N, n_rays) mask
            cam_pos = cameras.get_camera_center()
            cam_ray = cameras.unproject_points(torch.cat(
                [-p, p.new_ones(p.shape[:-1] + (1,))], dim=-1), scaled_depth_input=False) - cam_pos.unsqueeze(1)
            cam_ray = F.normalize(cam_ray, p=2, dim=-1, eps=1e-8)
            section0, section1, has_intersection = intersection_with_unit_cube(
                cam_pos.unsqueeze(1), cam_ray, side_length=2*self.object_bounding_sphere)
            # section0, section1, has_intersection = intersection_with_unit_sphere(
            #     cam_pos.unsqueeze(1), cam_ray, radius=self.object_bounding_sphere
            # )

            # 2. sample n_points_per_ray uniformly between the intersections
            section1 = section1[has_intersection]
            section0 = section0[has_intersection]
            cam_ray = cam_ray[has_intersection]
            lengths = torch.norm(section1 - section0, dim=-1)
            # assert(not (section0 == cam_ray).all(dim=-1).any(dim=0))
            lengths = torch.linspace(
                0, 1.0, n_points_per_ray, device=lengths.device) * lengths.unsqueeze(-1)
            world_points = lengths.unsqueeze(-1) * \
                cam_ray.unsqueeze(-2) + section0.unsqueeze(-2)

            # 3. sample freespace and occupancy points
            # NOTE: focus on rays that intersect the unit sphere to limit the sampling space
            # to the unit sphere.
            p_split_list = torch.split(
                world_points.view(-1, 3), max_points, dim=0)

            sdf_sampled = torch.cat([self.decoder.forward(p_split).sdf
                                     for p_split in p_split_list],
                                    dim=0)
            sdf_sampled = sdf_sampled.view(-1, n_points_per_ray)
            p_idx = torch.argmin(sdf_sampled, dim=-1, keepdim=True)

            world_points = world_points[torch.arange(
                world_points.shape[0]), p_idx.view(-1)]

            # pick the point between the two intersections that has the lowest sdf value
            mask_freespace = (mask_gt[has_intersection]
                              == 0) & iso_incamera[has_intersection]
            # if mask_pred is not None:
            #     mask_freespace = mask_freespace & (~mask_pred)
            n_free_per_batch = torch.stack([x.sum() for x in mask_freespace.split(
                has_intersection.sum(-1).tolist(), dim=0)])

            p_freespace = world_points[mask_freespace].view(-1, 3)

            mask_occupancy = mask_gt[has_intersection] & iso_incamera[has_intersection]
            if mask_pred is not None:
                mask_occupancy = mask_occupancy & (
                    ~mask_pred)[has_intersection]

            p_occupancy = world_points[mask_occupancy]
            n_occ_per_batch = torch.stack([x.sum() for x in mask_occupancy.split(
                has_intersection.sum(-1).tolist(), dim=0)])

        return p_freespace, n_free_per_batch, p_occupancy, n_occ_per_batch

    def sample_from_pixels(self, pixels, cameras, mask_gt, c=None, **kwargs):
        """ IDR implementation """
        batch_size, num_pixels = mask_gt.shape[:2]
        with autograd.no_grad():
            cam_pos = cameras.get_camera_center()
            cam_ray = cameras.unproject_points(torch.cat(
                [-pixels, pixels.new_ones(pixels.shape[:-1] + (1,))], dim=-1), scaled_depth_input=False) - \
                cam_pos.unsqueeze(1)
            cam_ray = F.normalize(cam_ray, dim=-1, p=2)
            points, mask_pred, dists = self.ray_tracing(sdf=lambda x: self.decode(x, **kwargs).sdf.squeeze(-1),
                                                        cam_loc=cam_pos,
                                                        object_mask=mask_gt.view(
                                                            -1),
                                                        ray_directions=cam_ray)
            dists = dists.view(batch_size, num_pixels)
            dists[dists == 0] = getattr(cameras, 'znear', 1.0)
            points = points.view(batch_size, num_pixels, 3)
            mask_pred = mask_pred.view(batch_size, num_pixels)

        iso_points = points.clone()
        if self.training:
            iso_points, _ = self.directional_sampling(self.decoder, iso_points.detach(),
                                                      cam_ray, cam_pos.view(batch_size, 1, 3), return_eval=True)

        free_points = points[~mask_gt]
        occ_points = points[(~mask_pred) & mask_gt]
        num_free = (~mask_gt).view(batch_size, -1).sum(-1)
        num_occ = ((~mask_pred) & mask_gt).view(batch_size, -1).sum(-1)

        iso_points = iso_points.view(batch_size, num_pixels, 3)
        mask_pred = mask_pred.view(batch_size, num_pixels)
        return iso_points, free_points, occ_points, mask_pred, num_free, num_occ

    def forward(self,
                pixels: torch.Tensor,
                img: torch.Tensor,
                mask_img: torch.Tensor,
                inputs: Optional[torch.Tensor] = None,
                it: int = None,
                lights=None,
                cameras: Optional[CamerasBase] = None,
                **kwargs
                ):
        ''' Performs a forward pass through the network.

        This function evaluates the depth and RGB color values for respective
        points as well as the occupancy values for the points of the helper
        losses. By wrapping everything in the forward pass, multi-GPU training
        is enabled.

        Args:
            inputs (tensor): input for encoder (like 2D image)
            pixels (tensor): (N, n_rays, 2)
            cameras (Camera): used to project and unproject points in the current view
            it (int): training iteration (used for ray sampling scheduler)
        Returns:
            point_clouds (PointClouds3D): with rgb, normals the successfully sphere-traced points
                and has mask value == 1
            pred_mask (tensor): bool (N, n_rays,) mask -> whether the sphere-tracing was successful
            p_freespace (tensor): (N1, 3) ray-traced points that has the lowest sdf value on the ray
            p_occupancy (tensor): (N2, 3)
            sdf_freespace (tensor): (N1,)
            sdf_occupancy (tensor): (N2,)
        '''
        batch_size = cameras.R.shape[0]
        self.cameras = cameras
        mask_gt = get_tensor_values(
            mask_img.float(), pixels, squeeze_channel_dim=True).bool()
        # encode inputs
        c = self.encode_inputs(inputs)

        # ############# DVR ################
        # # sphere-tracing find intersection
        # p_world, mask_pred = self.pixels_to_world(
        #     pixels, cameras, c, it=it, **kwargs)

        # # get freespace and occupancy points
        # p_freespace, num_freespace, p_occupancy, num_occupancy = self.sample_world_points(
        #     pixels, cameras, n_points_per_ray=self.n_points_per_ray,
        #     mask_gt=mask_gt, mask_pred=mask_pred, c=c)

        # mask_depth = mask_gt & mask_pred

        ############ IDR #############
        p_world, p_freespace, p_occupancy, mask_pred, num_freespace, num_occupancy = self.sample_from_pixels(
            pixels, cameras, mask_gt, **kwargs)

        mask_depth = mask_gt & mask_pred

        if next(self.parameters()).requires_grad and self.training:
            assert(p_world.requires_grad)

        # Code for debugging and visualizing gradients
        # 1. colored pointclouds will create photo-consistency loss (points -> normals -> color)
        # 2. p_freespace and p_occupancy loss will create sdf loss
        if get_debugging_mode():
            self._debugger_logging(
                p_world, p_freespace, p_occupancy, mask_depth, num_freespace, num_occupancy)

        pointclouds = self.get_point_clouds(p_world, mask_depth,
                                            c=c, with_colors=False,
                                            with_normals=True,
                                            require_normals_grad=True,
                                            cameras=cameras,
                                            lights=lights)

        # compute the color at the ray-traced point
        colored_pointclouds = self.decode_color(
            pointclouds, c=c, cameras=cameras, lights=lights, **kwargs)

        sdf_freespace = self.decode(p_freespace).sdf
        sdf_occupancy = self.decode(p_occupancy).sdf

        pixel_pred = pixels[mask_depth]
        rgb_gt = get_tensor_values(img, pixels)[mask_depth]
        return {'iso_pcl': colored_pointclouds,
                'iso_pixel': pixel_pred,
                'p_freespace': p_freespace,
                'p_occupancy': p_occupancy,
                'sdf_freespace': sdf_freespace,
                'sdf_occupancy': sdf_occupancy,
                'iso_rgb_gt': rgb_gt}

    def _debugger_logging(self, p_world, p_freespace, p_occupancy, mask_depth, num_freespace, num_occupancy):
        """ Operations to register gradient hooks for dbg tensor """
        p_freespace.requires_grad_(True)
        p_occupancy.requires_grad_(True)

        with torch.autograd.no_grad():
            dbg_tensor = get_debugging_tensor()
            dbg_tensor.pts_world['iso'] = [
                p_world[b, mask_depth[b]].detach().cpu() for b in range(p_world.shape[0])]

            dbg_tensor.pts_world['freespace'] = [
                x.cpu().detach() for x in torch.split(p_freespace,
                                                      num_freespace.tolist(), dim=0)]
            dbg_tensor.pts_world['occupancy'] = [
                x.cpu().detach() for x in torch.split(p_occupancy,
                                                      num_occupancy.tolist(), dim=0)]

        # return hook function in closure
        def save_grad_with_mask(mask, name):
            def _save_grad(grad):
                if torch.all(grad == 0):
                    return
                if dbg_tensor is None:
                    logger_py.error("dbg_tensor is None")
                if grad is None:
                    logger_py.error('grad is None')
                # a dict of list of tensors
                dbg_tensor.pts_world_grad[name] = [
                    grad[b, mask[b]].detach().cpu() for b in range(grad.shape[0])]
                if dbg_tensor.pts_world[name][0].shape != dbg_tensor.pts_world[name][0].shape:
                    breakpoint()
            return _save_grad

        # return hook function in closure
        def save_grad_sampled_points(num_points_per_batch, name):
            def _save_grad(grad):
                if torch.all(grad == 0):
                    return
                dbg_tensor = get_debugging_tensor()
                if dbg_tensor is None:
                    logger_py.error("dbg_tensor is None")
                if grad is None:
                    logger_py.error('grad is None')
                # a dict of list of tensors
                dbg_tensor.pts_world_grad[name] = [x.detach().cpu()
                                                   for x in torch.split(grad, num_points_per_batch.tolist(), dim=0)]
            return _save_grad

        handle = p_world.register_hook(
            save_grad_with_mask(mask_depth, 'iso'))
        self.hooks.append(handle)
        handle = p_occupancy.register_hook(
            save_grad_sampled_points(num_occupancy, 'occupancy'))
        self.hooks.append(handle)
        handle = p_freespace.register_hook(
            save_grad_sampled_points(num_freespace, 'freespace'))
        self.hooks.append(handle)

    def debug(self, is_debug, **kwargs):
        # retain gradient for debugging
        self.hooks = kwargs.get('hooks', self.hooks)
        if is_debug:
            pass
        else:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            c = None

        return c

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model


class Generator(BaseGenerator):
    '''  Generator class for DVRs.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained DVR model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        refine_max_faces (int): max number of faces which are used as batch
            size for refinement process (we added this functionality in this
            work)
    '''

    def __init__(self, model, device='cpu', points_batch_size=400000,
                 refinement_step=0,
                 resolution=100,
                 img_size=(512, 512),
                 with_normals=False, padding=0.1,
                 with_colors=False,
                 refine_max_faces=10000,
                 is_occupancy=False,
                 object_bounding_sphere=1.0,
                 **kwargs):
        super().__init__(model, device=device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.resolution = resolution
        self.img_size = img_size
        self.with_normals = with_normals
        self.padding = padding
        self.with_colors = with_colors
        self.refine_max_faces = refine_max_faces
        self.is_occupancy = is_occupancy
        self.object_bounding_sphere = object_bounding_sphere

    def generate_mesh(self, data, **kwargs):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
        '''
        with autograd.no_grad():
            self.model.eval()
            mesh = get_surface_high_res_mesh(lambda x: self.model.decode(x).sdf.squeeze(), resolution=self.resolution,
                                             box_side_length=self.object_bounding_sphere * 2)
        return mesh

    def generate_meshes(self, data, **kwargs) -> List[trimesh.Trimesh]:
        ''' Generates the output meshes with data of batch size >=1

        Args:
            data (tensor): data tensor
        '''
        meshes = super().generate_meshes(data, **kwargs)
        self.model.eval()

        mesh = self.generate_mesh(data, **kwargs)
        meshes.append(mesh)

        return meshes

    def eval_points(self, p, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points (N, 3)
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.decode(pi, c, **kwargs).sdf

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)
        # TODO: replace unit sphere with unit cube
        # occ_hat[(p.norm(dim=-1) > 1.0)] = 1.0 if not self.is_occupancy else -100.0
        occ_hat[(p.abs() >= 1.0).any(dim=-1)
                ] = 1.0 if not self.is_occupancy else -100.0
        return occ_hat

    def extract_mesh(self, occ_hat, c=None, **kwargs):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): latent conditioned code c
            spacing (tuple): tuple of length 3
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = self.object_bounding_sphere * 2 + self.padding
        spacing = (box_size / (n_x - 1), box_size /
                   (n_y - 1), box_size / (n_z - 1))
        threshold = 0.0

        # verts, faces = libmcubes.marching_cubes(occ_hat, threshold)
        verts, faces, normals, values = measure.marching_cubes(
            volume=occ_hat, level=threshold, spacing=spacing)
        if (occ_hat.min() * occ_hat.max()).sign() >= 0:
            return trimesh.Trimesh([])
        # Normalize to bounding box
        verts /= np.array([n_x - 1, n_y - 1, n_z - 1])
        verts = box_size * (verts - 0.5)

        # Estimate normals if needed
        with_normals = kwargs.get('with_normals', self.with_normals)
        normals = None
        if with_normals and not verts.shape[0] == 0:
            normals = self.estimate_normals(verts, c)

        # Create mesh
        mesh = trimesh.Trimesh(
            verts, faces, vertex_normals=normals, process=False)

        # Directly return if mesh is empty
        if verts.shape[0] == 0:
            return mesh

        # Refine mesh
        if self.refinement_step > 0:
            self.refine_mesh(mesh, c)

        # Estimate Vertex Colors
        with_colors = kwargs.pop('with_colors', self.with_colors)

        if with_colors and not verts.shape[0] == 0:
            t0 = time.time()
            vertex_colors = self.estimate_colors(
                np.array(mesh.vertices), c, **kwargs)
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices, faces=mesh.faces,
                vertex_normals=mesh.vertex_normals,
                vertex_colors=vertex_colors, process=False)

        if self.is_occupancy:
            mesh.invert()

        trimesh.repair.fix_winding(mesh)
        return mesh

    def estimate_colors(self, vertices, c=None, **kwargs):
        ''' Estimates vertex colors by evaluating the texture field.
        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): latent conditioned code c
        '''
        device = self.device
        with_normals = kwargs.pop('with_normals', self.with_normals)
        if not isinstance(vertices, torch.Tensor):
            vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        colors = []
        for vi in vertices_split:
            with torch.no_grad():
                ci = self.model.get_point_clouds(vi.unsqueeze(
                    0), with_colors=True, with_normals=with_normals, **kwargs).features_padded().squeeze(0).cpu()
                # ci = self.model.decode_color(
                #     vi.unsqueeze(0), c).squeeze(0).cpu()
            colors.append(ci)

        colors = np.concatenate(colors, axis=0)
        if colors.shape[-1] == 1:
            # using uncertainty
            cmap = cm.get_cmap('jet')
            normalizer = mpc.Normalize(vmin=colors.min(), vmax=colors.max())
            colors = normalizer(colors.squeeze(1))
            colors = cmap(colors)[:, :3]

        colors = np.clip(colors, 0, 1)
        return colors

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        with torch.autograd.enable_grad():
            device = self.device
            vertices = torch.FloatTensor(vertices)
            vertices_split = torch.split(vertices, self.points_batch_size)

            normals = []
            if c is not None:
                c = c.unsqueeze(0)
            for vi in vertices_split:
                vi = vi.unsqueeze(0).to(device)
                vi.requires_grad_()
                sdf_hat = self.model.decode(vi, c).sdf
                out = sdf_hat.sum()
                out.backward()
                ni = vi.grad
                ni = ni / torch.norm(ni, dim=-1, keepdim=True)
                ni = ni.squeeze(0).cpu().numpy()
                normals.append(ni)

            normals = np.concatenate(normals, axis=0)
            return normals

    def refine_mesh(self, mesh, c=None):
        ''' Refines the predicted mesh. An additional optimization
        step to correct the constructed face vertices and normals with the
        network prediction (logit and gradient)

        Args:
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''
        autograd.set_grad_enabled(True)
        self.model.eval()

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces)

        # detach c; otherwise graph needs to be retained
        # caused by new Pytorch version?
        if c is not None:
            c = c.detach()

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-5)

        # Dataset
        ds_faces = TensorDataset(faces)
        dataloader = DataLoader(ds_faces, batch_size=self.refine_max_faces,
                                shuffle=True)

        # We updated the refinement algorithm to subsample faces; this is
        # usefull when using a high extraction resolution / when working on
        # small GPUs
        it_r = 0
        while it_r < self.refinement_step:
            for f_it in dataloader:
                f_it = f_it[0].to(self.device)
                optimizer.zero_grad()

                # Loss
                face_vertex = v[f_it]
                eps = np.random.dirichlet((0.5, 0.5, 0.5), size=f_it.shape[0])
                eps = torch.FloatTensor(eps).to(self.device)
                face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

                face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
                face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
                face_normal = torch.cross(face_v1, face_v2)
                face_normal = face_normal / \
                    (face_normal.norm(dim=1, keepdim=True) + 1e-10)

                face_value = torch.cat([
                    self.model.decode(p_split, c).sdf
                    for p_split in torch.split(
                        face_point.unsqueeze(0), 20000, dim=1)], dim=1)

                normal_target = -autograd.grad(
                    [face_value.sum()], [face_point], create_graph=True)[0]

                normal_target = \
                    normal_target / \
                    (normal_target.norm(dim=1, keepdim=True) + 1e-10)
                loss_target = face_value.pow(2).mean()
                loss_normal = \
                    (face_normal - normal_target).pow(2).sum(dim=1).mean()

                loss = loss_target + 0.01 * loss_normal

                # Update
                loss.backward()
                optimizer.step()

                # Update it_r
                it_r += 1

                if it_r >= self.refinement_step:
                    break

        mesh.vertices = v.data.cpu().numpy()
        return mesh

    def generate_iso_contour(self, **kwargs):
        self.model.eval()
        with autograd.no_grad():

            def decode_pts_func(p):
                return self.eval_points(p, **kwargs)

            box_size = (self.object_bounding_sphere * 2 + self.padding, ) * 3
            imgs = plot_cuts(decode_pts_func,
                             box_size=box_size,
                             max_n_eval_pts=self.points_batch_size,
                             thres=0.0,
                             imgs_per_cut=kwargs.get('imgs_per_cut', 1))
        return imgs

    def raytrace_images(self, img_size, mask_img, **kwargs) -> "np.array":
        """
        Returns: a list of uint8 (H,W,4) rgba images
        """
        # generate colored image full res image
        cameras = kwargs.get(
            'cameras', self.model.cameras).to(self.device)
        lights = kwargs.get('lights', None)
        if lights is not None:
            lights = lights.to(device=self.device)
        H, W = img_size
        im_grid = make_image_grid((H, W), ndc=True).view(-1, 2)
        mask_img = mask_img.to(device=self.device)
        images = []
        for b in range(len(cameras)):
            colors_total = []
            masks_total = []
            for sp in torch.split(im_grid, 80000):
                sp = sp.to(device=self.device)
                cam_pos = cameras.get_camera_center()
                cam_rays = cameras.unproject_points(torch.cat(
                    [sp, sp.new_ones((sp.shape[0], 1,))], dim=-1), scaled_depth_input=False) - cam_pos
                cam_pos = cam_pos.unsqueeze(0)
                cam_rays = F.normalize(cam_rays.unsqueeze(0), dim=-1)
                mask_gt = get_tensor_values(
                    mask_img[b].unsqueeze(0).float(), -sp.unsqueeze(0), squeeze_channel_dim=True).bool()
                curr_start_points, network_object_mask = self.model.pixels_to_world(-sp, cameras)

                # curr_start_points, network_object_mask, acc_start_dis = \
                #     self.model.ray_tracing(lambda x: self.model.decode(x).sdf.squeeze(),
                #                            cam_pos.view(-1, 3), mask_gt.view(-1), cam_rays)
                colors = torch.zeros_like(curr_start_points.view(-1, 3), device='cpu')
                if not network_object_mask.any():
                    colors_total.append(colors)
                    masks_total.append(network_object_mask.cpu().detach().view(-1))
                    continue
                colors[network_object_mask.view(-1)] = self.model.get_point_clouds(curr_start_points.view(1, -1, 3), network_object_mask.view(1, -1),
                                                                          with_normals=True, with_colors=True,
                                                                          cameras=cameras, lights=lights).features_packed().cpu()
                colors_total.append(colors.detach())
                masks_total.append(network_object_mask.detach().cpu().view(-1))
                torch.cuda.empty_cache()

            colors_total = torch.cat(colors_total, dim=0)
            masks_total = torch.cat(masks_total, dim=0)
            colors_total = colors_total.reshape(H, W, 3)
            masks_total = masks_total.reshape(H, W, 1)
            rgba = torch.cat([colors_total, masks_total], dim=-1)
            rgba = rgba.numpy()
            images.append(rgba)
        return images

    def generate_images(self, data, **kwargs) -> List[np.array]:
        """
        create cross section implicit function value (logits of the decoder)
        contour plots
        """
        outputs = super().generate_images(data, **kwargs)
        with torch.autograd.no_grad():
            self.model.eval()
            batch_imgs = []
            imgs = self.generate_iso_contour(**kwargs)
            batch_imgs.append(imgs)
            outputs.extend(batch_imgs)

            cameras = kwargs.get('cameras', self.model.cameras)
            lights = kwargs.get('lights', None)
            img_mask = data.get('mask_img', None)
            if cameras is None or img_mask is None:
                logger_py.warn(
                    'Must provide cameras and data to generate full res view.')
                return outputs

            batch_imgs = self.raytrace_images(
                self.img_size, img_mask.to(self.device), cameras=cameras, lights=lights)
            outputs.extend(batch_imgs)

        return outputs
