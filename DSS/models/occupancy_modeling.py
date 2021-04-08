""" Neural Implicit Function for global shape regularization """
from typing import List, Optional
import torch
import time
import trimesh
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import autograd
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from pytorch3d.renderer.cameras import CamerasBase
from im2mesh.dvr.models import depth_function
from im2mesh.common import (make_3d_grid, check_tensor,
                            image_points_to_world,
                            get_mask,
                            transform_pointcloud,
                            origin_to_world,
                            normalize_tensor,
                            get_occupancy_loss_points,
                            get_freespace_loss_points)
from .. import (get_debugging_mode, set_debugging_mode_,
                get_debugging_tensor, logger_py)
from ..core.cloud import PointClouds3D
from ..utils import gather_batch_to_packed
from .implicit_modeling import Model as SDFModel
from .implicit_modeling import Generator as ImplicitGenerator


class Model(SDFModel):
    ''' DVR model class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
        depth_function_kwargs (dict): keyworded arguments for the
            depth_function
    '''

    def __init__(self, decoder, encoder=None, device='cpu',
                 depth_function_kwargs={},
                 use_cube_intersection=True,
                 occupancy_random_normal=False,
                 depth_range=(1.0, 10),
                 **kwargs):
        super().__init__(decoder, encoder=None, device=device, **kwargs)
        self.decoder = decoder.to(device=device)
        self.encoder = encoder

        if encoder is not None:
            self.encoder = encoder.to(device=device)

        if kwargs.get('shader', None):
            self.shader = kwargs.get('shader', None)

        self.device = device
        self.hooks = []
        self.call_depth_function = depth_function.DepthModule(
            **depth_function_kwargs)

        self.use_cube_intersection = use_cube_intersection
        self.occupancy_random_normal = occupancy_random_normal
        self.depth_range = depth_range

    def get_point_clouds(self, points, mask: torch.Tensor,
                         c: Optional[torch.Tensor] = None,
                         with_colors=False, with_normals=True,
                         require_normals_grad=False,
                         project=False,
                         **kwargs) -> PointClouds3D:
        """
        Returns the point clouds object from given points and points mask
        Args:
            p: (B,P,3) points in 3D source space (default = self.points)
            mask: (B,P) mask for valid points
            c: (B,C) latent code
            with_colors: use shader or neural shader to get colored points
            with_normals: use decoder gradient to get the normal vectors
        """
        assert(not project), "Occupancy Model doesn't support projection"
        return super().get_point_clouds(points, mask=mask, c=c, with_colors=with_colors, with_normals=with_normals,
                                        require_normals_grad=require_normals_grad, project=project, **kwargs)

    def get_normals_from_grad(self, *args, **kwargs):
        """
        Returns the not normalized normals at the query points by differentiating
        the implicit function w.r.t the input points

        Args:
            p_world (tensor): [N,*,3] points in world coordinates
            c (tensor): latent conditioned code [N,*, C]
        """
        outputs = super().get_normals_from_grad(*args, **kwargs)
        if kwargs.get('return_sdf', False):
            outputs[0] = -outputs[0]
        return outputs

    def sample_world_points(self, pixels, cameras):
        """
        Calculate 3D points which need to be evaluated for the occupancy and
        freespace loss
        """
        p_freespace = get_freespace_loss_points(
            pixels, cameras, self.use_cube_intersection,
            [cameras.znear, cameras.zfar])  # (B,N,3)

        # depth_input = depth_img if (
        #     self.lambda_depth != 0 or self.depth_from_visual_hull) else None
        p_occupancy = get_occupancy_loss_points(
            pixels, cameras, None, self.use_cube_intersection, self.occupancy_random_normal,
            [cameras.znear, cameras.zfar])  # (B,N,3)

        return p_freespace, p_occupancy

    def forward(self, pixels, mask_gt: torch.Tensor,
                inputs: Optional[torch.Tensor] = None,
                it: int = None,
                cameras: Optional[CamerasBase] = None,
                **kwargs):
        ''' Performs a forward pass through the network.

        This function evaluates the depth and RGB color values for respective
        points as well as the occupancy values for the points of the helper
        losses. By wrapping everything in the forward pass, multi-GPU training
        is enabled.

        Args:
            pixels (tensor): sampled pixels
            mask_gt (tensor): mask value at the sampled pixels
            cameras (Cameras): camera model for the views
            inputs (tensor): input
            it (int): training iteration (used for ray sampling scheduler)
        Returns:
            point_clouds (PointClouds3D): with rgb, normals
            pred_mask (tensor): bool (N, n_rays,) mask -> whether the sphere-tracing was successful
            p_freespace (tensor): (N1, 3) ray-traced points that has the lowest sdf value on the ray
            p_occupancy (tensor): (N2, 3)
            sdf_freespace (tensor): (N1, 1)
            sdf_occupancy (tensor): (N2, 1)
        '''
        self.cameras = cameras or self.cameras

        batch_size = self.cameras.R.shape[0]

        # encode inputs
        c = self.encode_inputs(inputs)

        # find isosurface points using ray-tracing
        p_world, mask_pred, mask_zero_occupied = \
            self.pixels_to_world(pixels, cameras, c, it)

        if not (~mask_zero_occupied & mask_pred).any():
            logger_py.info("Found no iso-surface intersection")

        if next(self.parameters()).requires_grad:
            assert(p_world.requires_grad)

        # TODO: once occupancy network works, change this to the same signature
        # as the implicit function sample_world_points
        p_freespace, p_occupancy = self.sample_world_points(pixels, cameras)
        p_freespace[mask_pred] = p_world[mask_pred].detach()

        # mask_depth = (mask_pred & mask_gt & (~mask_zero_occupied))
        mask_depth = mask_pred & mask_gt
        mask_occupancy = (mask_pred == 0) & mask_gt
        # mask_freespace = (mask_gt == 0) & mask_pred
        mask_freespace = mask_gt == 0
        p_freespace = p_freespace[mask_freespace].detach()
        p_occupancy = p_occupancy[mask_occupancy].detach()

        # Code for debugging and visualizing gradients
        if get_debugging_mode():
            self._debugger_logging(
                p_world, p_freespace, p_occupancy, mask_depth, mask_freespace.sum(dim=1), mask_occupancy.sum(dim=1))

        colored_pointclouds = self.get_point_clouds(p_world, mask_depth,
                                                    c=c, with_normals=(self.texture is not None),
                                                    with_colors=True, require_normals_grad=True,
                                                    cameras=cameras)

        # compute sdf value of the freespace and occupancy points
        if c is not None:
            first_idx = [torch.full((mask_freespace[b].sum(
            ),), b, device=self.device, dtype=torch.long) for b in range(batch_size)]
            latent_freespace = gather_batch_to_packed(c, first_idx)

            first_idx = [torch.full((mask_occupancy[b].sum(
            ),), b, device=self.device, dtype=torch.long) for b in range(batch_size)]
            latent_occupancy = gather_batch_to_packed(c, first_idx)
        else:
            latent_freespace = None
            latent_occupancy = None

        sdf_freespace = -self.decode(p_freespace, c=latent_freespace).sdf
        sdf_occupancy = -self.decode(p_occupancy, c=latent_occupancy).sdf

        return (colored_pointclouds, mask_pred, p_freespace, p_occupancy, sdf_freespace, sdf_occupancy)


    def get_normals(self, points, mask, c=None, h_sample=1e-3,
                    h_finite_difference=1e-3):
        ''' Returns the unit-length normals for points and one randomly
        sampled neighboring point for each point.

        Args:
            points (tensor): points tensor
            mask (tensor): mask for points
            c (tensor): latent conditioned code c
            h_sample (float): interval length for sampling the neighbors
            h_finite_difference (float): step size finite difference-based
                gradient calculations
        '''
        device = self.device

        if mask.sum() > 0:
            if c is not None:
                c = c.unsqueeze(1).repeat(1, points.shape[1], 1)[mask]
            points = points[mask]
            points_neighbor = points + (torch.rand_like(points) * h_sample -
                                        (h_sample / 2.))

            normals_p = normalize_tensor(
                self._get_central_difference(points, c=c,
                                             h=h_finite_difference))
            normals_neighbor = normalize_tensor(
                self._get_central_difference(points_neighbor, c=c,
                                             h=h_finite_difference))
        else:
            normals_p = torch.empty(0, 3).to(device)
            normals_neighbor = torch.empty(0, 3).to(device)

        return [normals_p, normals_neighbor]

    def _get_central_difference(self, points, c=None, h=1e-3):
        ''' Calculates the central difference for points.

        It approximates the derivative at the given points as follows:
            f'(x) ≈ f(x + h/2) - f(x - h/2) for a small step size h

        Args:
            points (tensor): points
            c (tensor): latent conditioned code c
            h (float): step size for central difference method
        '''
        n_points, _ = points.shape
        device = self.device

        if c is not None and c.shape[-1] != 0:
            c = c.unsqueeze(1).repeat(1, 6, 1).view(-1, c.shape[-1])

        # calculate steps x + h/2 and x - h/2 for all 3 dimensions
        step = torch.cat([
            torch.tensor([1., 0, 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([-1., 0, 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, 1., 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, -1., 0]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, 0, 1.]).view(1, 1, 3).repeat(n_points, 1, 1),
            torch.tensor([0, 0, -1.]).view(1, 1, 3).repeat(n_points, 1, 1)
        ], dim=1).to(device) * h / 2
        points_eval = (points.unsqueeze(1).repeat(1, 6, 1) + step).view(-1, 3)

        # Eval decoder at these points
        f = self.decoder(points_eval, c=c, only_occupancy=True,
                         batchwise=False).view(n_points, 6)

        # Get approximate derivate as f(x + h/2) - f(x - h/2)
        df_dx = torch.stack([
            (f[:, 0] - f[:, 1]),
            (f[:, 2] - f[:, 3]),
            (f[:, 4] - f[:, 5]),
        ], dim=-1)
        return df_dx

    def decode(self, p, c=None, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, **kwargs)
        return logits

    def march_along_ray(self, ray0, ray_direction, c=None, it=None,
                        sampling_accuracy=None):
        ''' Marches along the ray and returns the d_i values in the formula
            r(d_i) = ray0 + ray_direction * d_i
        which returns the surfaces points.

        Here, ray0 and ray_direction are directly used without any
        transformation; Hence the evaluation is done in object-centric
        coordinates.

        Args:
            ray0 (tensor): ray start points (camera centers)
            ray_direction (tensor): direction of rays; these should be the
                vectors pointing towards the pixels
            c (tensor): latent conditioned code c
            it (int): training iteration (used for ray sampling scheduler)
            sampling_accuracy (tuple): if not None, this overwrites the default
                sampling accuracy ([128, 129])
        '''
        device = self.device

        d_i = self.call_depth_function(ray0, ray_direction, self.decoder,
                                       c=c, it=it, n_steps=sampling_accuracy)

        # Get mask for where first evaluation point is occupied
        mask_zero_occupied = d_i == 0

        # Get mask for predicted depth
        mask_pred = get_mask(d_i).detach()

        # For sanity for the gradients
        d_hat = torch.ones_like(d_i).to(device)
        d_hat[mask_pred] = d_i[mask_pred]

        d_hat[mask_zero_occupied] = 0

        return d_hat, mask_pred, mask_zero_occupied

    def pixels_to_world(self, pixels, cameras, c,
                        it=None, sampling_accuracy=None):
        ''' Projects pixels to the world coordinate system.

        Args:
            pixels (tensor): sampled pixels in range [-1, 1]
            c (tensor): latent conditioned code c
            it (int): training iteration (used for ray sampling scheduler)
            sampling_accuracy (tuple): if not None, this overwrites the default
                sampling accuracy ([128, 129])
        '''
        batch_size, n_points, _ = pixels.shape
        pixels_world = image_points_to_world(pixels, cameras)
        camera_world = origin_to_world(n_points, cameras)
        ray_vector = (pixels_world - camera_world)

        d_hat, mask_pred, mask_zero_occupied = self.march_along_ray(
            camera_world, ray_vector, c, it, sampling_accuracy)
        # NOTE: d_i can be zero because call_depth_function set mask_zero_occupied points to 0,
        # but camera transform will outputs nan/infinite for zero depth points
        d_hat[mask_zero_occupied] = cameras.znear
        p_world_hat = camera_world + ray_vector * d_hat.unsqueeze(-1)
        return p_world_hat, mask_pred, mask_zero_occupied

    def decode_color(self, pointclouds: PointClouds3D, c=None, **kwargs) -> PointClouds3D:
        ''' Decodes the color values for world points.

        Args:
            p_world (tensor): world point tensor
            c (tensor): latent conditioned code c
        '''
        if pointclouds.isempty():
            return pointclouds
        p_world = pointclouds.points_packed()
        # if not hasattr(self, 'texture') or self.texture is None:
        if c is not None:
            c = gather_batch_to_packed(c, pointclouds.packed_to_cloud_idx())
        colored_pointclouds = self.texture(pointclouds, c=c, only_texture=True, **kwargs)

        return colored_pointclouds

class Generator(ImplicitGenerator):
    """

    """
    def __init__(self, model, device='cpu', points_batch_size=400000,
                 refinement_step=0,
                 resolution=16, upsampling_steps=3,
                 simplify_nfaces=None,
                 with_normals=False, padding=0.1,
                 with_colors=False,
                 refine_max_faces=10000,
                 **kwargs):
        super().__init__(model, device=device, points_batch_size=points_batch_size,
                 refinement_step=refinement_step,
                 resolution=resolution, upsampling_steps=upsampling_steps,
                 simplify_nfaces=simplify_nfaces,
                 with_normals=with_normals, padding=padding,
                 with_colors=False, with_uncertainty=False,
                 refine_max_faces=refine_max_faces,
                 is_occupancy=True,
                 bbox_side_length=1.0,
                 **kwargs)
