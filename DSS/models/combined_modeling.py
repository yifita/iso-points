from typing import Optional, List, Tuple, Union
from pytorch3d.ops.utils import convert_pointclouds_to_tensor
from pytorch3d.structures import Pointclouds
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from pytorch3d.ops import padded_to_packed
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import padded_to_list, packed_to_list, list_to_padded
from ..utils import (intersection_with_unit_cube,
                     gather_batch_to_packed, get_tensor_values,
                     get_visible_points, mask_from_padding, mask_padded_to_list, mask_padded_to_list, num_points_2_packed_to_cloud_idx,
                     )
from ..utils.mathHelper import clip_norm, eps_sqrt
from ..utils.point_processing import upsample
from ..core.cloud import PointClouds3D, PointCloudsFilters
from ..core.texture import LightingTexture
from .implicit_modeling import Generator as ImplicitGenerator
from .implicit_modeling import Model as ImplicitModel
from .point_modeling import Model as PointModel
from .point_modeling import Generator as PointGenerator
from .levelset_sampling import (UniformProjection,
                                SampleNetwork, SphereTracing, DirectionalSamplingNetwork,
                                RayTracing)
from .. import logger_py, get_debugging_mode, get_debugging_tensor


def _save_grad_for_pcl_with_name(name, pcls=None, num_points_per_cloud=None):
    if num_points_per_cloud is None and pcls is not None:
        num_points_per_cloud = pcls.num_points_per_cloud().tolist()
    elif num_points_per_cloud is None and pcls is None:
        raise ValueError(
            "_save_grad_for_pcl_with_name requires either pcls or num_points_per_cloud")
    elif num_points_per_cloud is not None:
        if isinstance(num_points_per_cloud, torch.Tensor):
            num_points_per_cloud = num_points_per_cloud.tolist()

    def _save_grad(grad):
        if torch.all(grad == 0):
            return
        dbg_tensor = get_debugging_tensor()
        # a dict of list of tensors
        if grad.ndim == 3:
            grad = padded_to_list(
                grad, num_points_per_cloud)
        elif grad.ndim == 2:
            grad = packed_to_list(
                grad, num_points_per_cloud)
        dbg_tensor.pts_world_grad[name] = [
            grad[b].detach().cpu() for b in range(len(grad))]

    return _save_grad


class Model(ImplicitModel, PointModel):
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
                 renderer: Optional[nn.Module] = None,
                 texture: Optional[nn.Module] = None,
                 encoder: Optional[nn.Module] = None,
                 cameras: Optional[CamerasBase] = None,
                 device: Optional[torch.device] = 'cpu',
                 n_points_per_cloud: Optional[int] = 5000,
                 points: Optional[torch.Tensor] = None,
                 n_points_per_ray: Optional[int] = 100,
                 max_points_per_pass: Optional[int] = 120000,
                 proj_max_iters: Optional[int] = 10,
                 proj_tolerance: Optional[float] = 5e-5,
                 object_bounding_sphere: float = 1.0,
                 max_iso_per_batch: int = 1000,
                 **kwargs
                 ):
        nn.Module.__init__(self)
        self.decoder = decoder.to(device=device)
        self.encoder = encoder
        self.renderer = renderer
        self.texture = texture
        self.object_bounding_sphere = object_bounding_sphere
        self.cameras = cameras or renderer.cameras
        self.cfg = kwargs

        self.max_iso_per_batch = max_iso_per_batch

        self.hooks = []
        self.device = device
        self.n_points_per_cloud = n_points_per_cloud
        self.n_points_per_ray = n_points_per_ray
        self.max_points_per_pass = max_points_per_pass

        # sampled iso-surface points in the space
        if points is not None:
            self.points = points.to(device).view(1, -1, 3)
        else:
            # initialize points as sphere (n_batch, n_points_per_cloud,3)
            if n_points_per_cloud > 0:
                self.points = (torch.rand((1, n_points_per_cloud, 3), device=device) - 0.5) * 1.5
            else:
                self.points = torch.empty(
                    [1, 0, 3], dtype=torch.float, device=device)
        self._points = PointClouds3D(self.points)

        self.projection = UniformProjection(
            max_points_per_pass=max_points_per_pass, proj_max_iters=proj_max_iters,
            proj_tolerance=proj_tolerance,
            **kwargs,
        )

        self.sphere_tracing = SphereTracing(
            max_points_per_pass=max_points_per_pass, proj_max_iters=proj_max_iters,
            proj_tolerance=proj_tolerance, alpha=1.0, **kwargs)
        self.ray_tracing = RayTracing(object_bounding_sphere=object_bounding_sphere,
                                      sdf_threshold=proj_tolerance,
                                      line_search_step=0.5,
                                      line_step_iters=3,
                                      sphere_tracing_iters=proj_max_iters,
                                      n_steps=100,
                                      n_secant_steps=8,)
        self.sampling = SampleNetwork()
        self.directional_sampling = DirectionalSamplingNetwork()

    def _debugger_logging(self,
                          p_freespace=None,
                          n_free_per_batch=None,
                          p_occupancy=None,
                          n_occ_per_batch=None):

        dbg_tensor = get_debugging_tensor()

        # return hook function in closure
        def save_grad_sampled_points(n_pts_per_batch, name):
            def _save_grad(grad):
                dbg_tensor = get_debugging_tensor()
                if dbg_tensor is None:
                    logger_py.error("dbg_tensor is None")
                if grad is None:
                    logger_py.error('grad is None')
                # a dict of list of tensors
                dbg_tensor.pts_world_grad[name] = [x.detach().cpu()
                                                   for x in torch.split(grad, n_pts_per_batch.tolist())
                                                   ]
            return _save_grad

        if p_occupancy is not None and n_occ_per_batch is None:
            assert(p_occupancy.ndim == 2)
            n_occ_per_batch = p_occupancy.shape[1]

        if p_freespace is not None and n_free_per_batch is None:
            assert(p_freespace.ndim == 2)
            n_free_per_batch = p_freespace.shape[1]

        if p_occupancy is not None and n_occ_per_batch is not None:
            p_occupancy.requires_grad_(True)
            with torch.autograd.no_grad():
                dbg_tensor.pts_world['occupancy'] = [
                    x.cpu().detach() for x in torch.split(p_occupancy, n_occ_per_batch.tolist(), dim=0)]
            handle = p_occupancy.register_hook(
                save_grad_sampled_points(n_occ_per_batch, 'occupancy'))
            self.hooks.append(handle)

        if p_freespace is not None and n_free_per_batch is not None:
            p_freespace.requires_grad_(True)
            with torch.autograd.no_grad():
                dbg_tensor.pts_world['freespace'] = [
                    x.cpu().detach() for x in torch.split(p_freespace, n_free_per_batch.tolist(), dim=0)]
            handle = p_freespace.register_hook(
                save_grad_sampled_points(n_free_per_batch, 'freespace'))
            self.hooks.append(handle)

    def sample_onsurface_using_isopoints(self, iso_points: PointClouds3D, mask_img: torch.Tensor, cameras: CamerasBase, c=None):
        """
        Get on-surface and in-mask samples from iso-points
        Args:
            iso_points (PointClouds) iso points with normalized coordinates (top-left is (1, 1))
                bottom right is (-1, -1)
            mask (B, P) mask for incamera and inmask and successfully projected points
            cameras (BaseCameras)
            c: latent code (B,*,C)
        Returns:
            p_world (P, 3) ray-traced points
            num_onsurface_from_isopoints (B,): number of successful ray-tracing per batch
        """
        batch_size = cameras.R.shape[0]
        iso_points, num_points = convert_pointclouds_to_tensor(iso_points)
        _, P = iso_points.shape[:2]
        with autograd.no_grad():
            sample_points = iso_points
            p_screen_hat = cameras.transform_points(sample_points)
            iso_mask_gt = get_tensor_values(mask_img.float(),
                                            (-p_screen_hat[..., :2]
                                             ).clamp(-1.0, 1.0),
                                            squeeze_channel_dim=True).bool()

            mask = iso_mask_gt & mask_from_padding(num_points)
            iso_inmask = iso_points[mask]
            zero_number = torch.zeros(
                (batch_size,), dtype=torch.long, device=iso_inmask.device)
            if iso_inmask.nelement() == 0:
                return iso_inmask, zero_number

            num_ons_iso_per_batch = mask.sum(dim=-1)
            iso_visible = iso_inmask

        if self.training:
            if isinstance(self.texture, LightingTexture):
                iso_visible, _ = self.sampling(
                    self.decoder, iso_visible.detach(), return_eval=True)
            else:
                # First find initial ray0 as the intersection with the unit sphere
                cam_pos = cameras.get_camera_center()
                packed_to_cloud_idx = torch.arange(
                    batch_size).view(-1, 1).expand(batch_size, P)[mask].to(device=iso_inmask.device)
                cam_pos_packed = gather_batch_to_packed(
                    cam_pos, packed_to_cloud_idx)
                cam_ray = F.normalize(iso_inmask - cam_pos_packed,
                                      p=2, dim=-1, eps=1e-10)
                cam_ray = F.normalize(cam_ray, dim=-1, p=2)
                cam_ray = cam_ray.view_as(iso_visible)
                if batch_size != 1:
                    cam_pos_packed = cam_pos_packed.view_as(iso_visible)
                iso_visible, _ = self.directional_sampling(self.decoder, iso_visible.detach(),
                                                           cam_ray, cam_pos_packed, return_eval=True)

        return iso_visible, num_ons_iso_per_batch

    def sample_offsurface_using_isopoints(self, pixels: torch.Tensor,
                                          mask_img: torch.Tensor, cameras: CamerasBase,
                                          n_points_per_ray=64, max_insurface_per_batch=None,
                                          iso_pcl: Pointclouds = None):
        """
        Get ray between camera and the iso_points, find the smallest sdf value point
        between (iso-points, *)
        ----
        Render iso-points from the opposite view (-c), get "visible-iso-points", which are occluded points (M,3).
        Subsample R rays from camera to iso-points (R, 3), approximate intersection between (R,3) and (M,3).
        These intersections define *

        Args:
            iso_points: pointclouds structure representation of the visible iso-points (output of get_visible_iso_points)
            mask_img: (N,1,H,W)
            cameras
            n_points_per_ray: uniform sample candidates along segment
            iso_pcl: if None, then add to outsurface points
        Returns:
            offsurface_from_isopoints: (N2, 3)
            insurface_from_isopoints: (N1, 3)
            num_off_per_batch: (N,)
            num_in_per_batch: (N,)
        """
        with autograd.no_grad():

            batch_size = cameras.R.shape[0]

            cam_pos = cameras.get_camera_center()
            sample_points_padded = cameras.unproject_points(torch.cat(
                [-pixels, pixels.new_ones(pixels.shape[:-1] + (1,))], dim=-1), scaled_depth_input=False)
            cam_ray = F.normalize(sample_points_padded - cam_pos.unsqueeze(1), dim=-1)

            # use mask to separate offsurface and insurface rays
            # convert to padded to get mask
            p_screen_hat = cameras.transform_points(sample_points_padded)
            iso_mask = get_tensor_values(mask_img.float(),
                                            (-p_screen_hat[..., :2]
                                             ).clamp(-1.0, 1.0),
                                            squeeze_channel_dim=True).bool()

            # offsurface: sample randomly between the cube intersections
            section0, section1, has_intersection = intersection_with_unit_cube(
                cam_pos.view(batch_size, 1, 3),
                cam_ray,
                side_length=self.object_bounding_sphere*2)
            lengths = torch.norm(section1 - section0, dim=-1)
            p_offsurface = (torch.rand_like(lengths) * lengths).unsqueeze(-1) * cam_ray + section0
            mask_offsurface = (~iso_mask) & has_intersection
            p_offsurface = p_offsurface[mask_offsurface]
            num_off_per_batch = mask_offsurface.sum(dim=1)

            # + iso-points that are outside the 2D mask
            if iso_pcl is not None:
                p_screen_hat = cameras.transform_points(iso_pcl.points_padded())
                iso_mask_gt = get_tensor_values(mask_img.float(),
                                                (-p_screen_hat[..., :2]
                                                ).clamp(-1.0, 1.0),
                                                squeeze_channel_dim=True).bool()
                mask_offsurface = (~iso_mask_gt) & mask_from_padding(iso_pcl.num_points_per_cloud())
                p_offsurface_1 = iso_pcl.points_padded()[mask_offsurface]
                p_offsurface = torch.cat([p_offsurface, p_offsurface_1], dim=0)
                num_off_per_batch += mask_offsurface.sum(dim=-1)


            # Occluded points: sample between frontal and back intersections with iso-points
            # get the camera looking back
            mask_insurface = torch.full_like(iso_mask, False)
            if max_insurface_per_batch is not None:
                for b in range(batch_size):
                    sub_idx = iso_mask[b].nonzero(as_tuple=False)[:min(max_insurface_per_batch[b], iso_mask[b].sum())]
                    mask_insurface[b][sub_idx] = True
            sample_points_lst = [sample_points_padded[b][mask_insurface[b]] for b in range(batch_size)]

            # get frontal points (pointclouds)
            frontal_points = get_visible_points(self._points, cameras,
                                depth_merge_threshold=self.renderer.rasterizer.raster_settings.depth_merging_threshold)
            # get occluded points (pointclouds)
            cameras_back = cameras.clone()
            # for IDR data
            if hasattr(cameras, 'principal_point'):
                cameras_back.principal_point[:, 1] = -cameras.principal_point[:, 1]
            cameras_back.R[:, :, [0, 2]] = -cameras.R[:, :, [0, 2]]  # rotate around y axis
            cameras_back.T = -torch.bmm(cameras_back.R.transpose(1, 2), -cameras.get_camera_center()[:, :, None])[:, :, 0]
            occluded_points = get_visible_points(self._points, cameras_back,
                                                 depth_merge_threshold=self.renderer.rasterizer.raster_settings.depth_merging_threshold)

            # TODO: faster search
            # for each batch search for the intersetion based on point-to-ray distance
            ray_len0_lst = []  # first intersecting point
            ray_len1_lst = []  # distance between intersections
            for b in range(batch_size):
                # (R, 3)
                occ_batch = occluded_points.points_list()[b]
                fro_batch = frontal_points.points_list()[b]

                ray_batch = sample_points_lst[b] - cam_pos[b].view(1, 3)
                ray0 = F.normalize(ray_batch, dim=-1)

                # get closest occluded point to ray -> farther bound
                # (M, 3) p - C
                pC = occ_batch - cam_pos[b].view(1, 3)
                # (R, M)
                ray_sq = (pC[None, :, :] * ray0[:, None, :]).sum(-1) ** 2
                dist_to_ray = (pC ** 2).sum(-1).unsqueeze(0) - ray_sq
                _, nn_idx = torch.topk(dist_to_ray, k=1, dim=1, largest=False)
                ray_len1 = torch.gather(ray_sq, 1, nn_idx).view(ray0.shape[0], 1)

                # get closest frontal point to ray -> closer bound
                # (M, 3) p - C
                pC = fro_batch - cam_pos[b].view(1, 3)
                # (R, M)
                ray_sq = (pC[None, :, :] * ray0[:, None, :]).sum(-1) ** 2
                dist_to_ray = (pC ** 2).sum(-1).unsqueeze(0) - ray_sq
                _, nn_idx = torch.topk(dist_to_ray, k=1, dim=1, largest=False)
                ray_len0 = torch.gather(ray_sq, 1, nn_idx).view(ray0.shape[0], 1)

                valid = (ray_len0 < ray_len1).view(-1)
                mask_insurface[b][mask_insurface[b]] = (ray_len0 < ray_len1).view(-1)

                ray_len1_lst.append(eps_sqrt(ray_len1[valid]).sqrt())
                ray_len0_lst.append(eps_sqrt(ray_len0[valid]).sqrt())

            num_ins_per_batch = mask_insurface.sum(dim=-1)

            ray_len0 = torch.cat(ray_len0_lst, dim=0)
            ray_len1 = torch.cat(ray_len1_lst, dim=0)

            cam_pos_ins = gather_batch_to_packed(cam_pos, num_points_2_packed_to_cloud_idx(num_ins_per_batch))
            cam_ray = F.normalize(sample_points_padded[mask_insurface] - cam_pos_ins)

            # P,n_points_per_ray
            lengths_sampled = torch.linspace(0, 1.0, n_points_per_ray+2, device=lengths.device)[1:-1] * (ray_len1 - ray_len0) + ray_len0
            # P,n_points_per_ray,3
            insurface_candidates = lengths_sampled.unsqueeze(-1) * cam_ray.unsqueeze(-2) + cam_pos_ins.unsqueeze(-2)

            # 3. sample freespace and occupancy points
            # NOTE: focus on rays that intersect the unit sphere to limit the sampling space
            # to the unit sphere.
            p_split_list = torch.split(
                insurface_candidates.view(-1, 3), self.max_points_per_pass, dim=0)

            sdf_sampled = torch.cat([self.decoder.forward(p_split).sdf
                                     for p_split in p_split_list],
                                    dim=0)
            sdf_sampled = sdf_sampled.view(-1, n_points_per_ray)
            p_idx = torch.argmin(sdf_sampled, dim=-1, keepdim=True)

            p_insurface = torch.gather(
                insurface_candidates, -2, p_idx.unsqueeze(-1).expand(-1, -1, insurface_candidates.shape[-1])).squeeze(-2)

        return p_offsurface, p_insurface, num_off_per_batch, num_ins_per_batch

    def get_visible_iso_points(self, cameras, **proj_kwargs):
        """
        Returns not occuluded iso-points under the camera view (B, P ,3)
        """
        max_iso_per_batch = self.max_iso_per_batch
        if max_iso_per_batch == 0:
            return torch.zeros((1, 0, 3), device=self.device, dtype=torch.float)

        batch_size = cameras.R.shape[0]
        # need normals for visibility check
        if self._points.normals_packed() is None:
            normals = self.get_normals_from_grad(
                self._points.points_packed(), requires_grad=False)
            self._points.update_normals_(normals)

        # use rasterizer to find visible points
        ref_pcl = proj_kwargs.get('ref_pcl', None)
        if ref_pcl is not None:
            assert(len(ref_pcl)==1), "Currently support optimizing a single shape, with only one reference point cloud."
            if ref_pcl.normals_packed() is None:
                normals = self.get_normals_from_grad(
                    ref_pcl.points_packed(), requires_grad=False)
                ref_pcl.update_normals_(normals)
            _, mask = get_visible_points(ref_pcl, cameras,
                                            depth_merge_threshold=self.renderer.rasterizer.raster_settings.depth_merging_threshold,
                                            return_mask=True)
            pcl_filter = PointCloudsFilters(device=proj_kwargs['ref_pcl'].device, visibility=mask.any(dim=0, keepdim=True))
            ref_pcl = pcl_filter.filter(ref_pcl)
            proj_kwargs['ref_pcl'] = ref_pcl


        init_iso_points = self._points.clone()
        pcl_visible, mask = get_visible_points(init_iso_points.extend(batch_size), cameras,
                                         depth_merge_threshold=self.renderer.rasterizer.raster_settings.depth_merging_threshold,
                                         return_mask=True)

        # make sure that per batch at least #iso-points = max_iso_per_batch
        if max_iso_per_batch > 0:
            pts_visible_lst = []
            for b in range(batch_size):
                max_iso_per_batch_padded = max_iso_per_batch
                min_iso_per_batch_padded = 0.75 * max_iso_per_batch_padded
                if ref_pcl is not None:
                    min_iso_per_batch_padded = int(0.8*min_iso_per_batch_padded)
                    max_iso_per_batch_padded = int(0.8*max_iso_per_batch_padded)
                if pcl_visible.num_points_per_cloud()[b] > max_iso_per_batch_padded:
                    subsampled = pcl_visible[b].subsample_randomly(max_iso_per_batch_padded/pcl_visible.num_points_per_cloud().float()[b:b+1])
                    pts_visible_lst.append(subsampled.points_packed())
                elif pcl_visible.num_points_per_cloud()[b] < min_iso_per_batch_padded:
                    upsampled = upsample(pcl_visible[b], max_iso_per_batch_padded)
                    pts_visible_lst.append(upsampled.points_packed())
                else:
                    pts_visible_lst.append(pcl_visible[b].points_packed())
            pcl_visible = PointClouds3D(pts_visible_lst)
        else:
            pcl_filter = PointCloudsFilters(device=init_iso_points.device, visibility=mask.any(dim=0, keepdim=True))
            pcl_visible = pcl_filter.filter(init_iso_points)

        pcl_visible.offset_(0.05 * (torch.rand_like((pcl_visible.points_packed())) - 0.5))
        proj_results = self.projection.project_points(
            pcl_visible, self.decoder, skip_resampling=True, skip_upsampling=(ref_pcl is None), **proj_kwargs)

        iso_pcl = PointClouds3D(list(mask_padded_to_list(proj_results['levelset_points'], proj_results['mask'])),
                                normals=list(mask_padded_to_list(proj_results['levelset_normals'], proj_results['mask'])))
        # normals = self.get_normals_from_grad(
        #     iso_pcl.points_packed(), requires_grad=False)
        # iso_pcl.update_normals_(normals)
        iso_pcl = get_visible_points(
            iso_pcl, cameras, depth_merge_threshold=self.renderer.rasterizer.raster_settings.depth_merging_threshold)
        return iso_pcl

    def get_point_clouds(self, points: Optional[torch.Tensor] = None,
                         mask: Optional[torch.Tensor] = None,
                         with_colors=False, with_normals=True,
                         require_normals_grad=False,
                         project=False, debug_name_prefix='',
                         **kwargs):
        if points is None:
            if cameras := kwargs.get('cameras', None):
                points = self.points.view(1, -1, 3)
            else:
                points = self._points.points_padded()

            mask = None

        return super().get_point_clouds(points, mask=mask,
                                        with_colors=with_colors, with_normals=with_normals,
                                        require_normals_grad=require_normals_grad,
                                        project=project, debug_name_prefix=debug_name_prefix,
                                        **kwargs)

    def forward(self,
                mask_img: torch.Tensor,
                img_gt: torch.Tensor,
                cameras,
                pixels: Optional[torch.Tensor] = None,
                inputs: Optional[torch.Tensor] = None,
                lights=None,
                project: bool = True,
                sample_iso_offsurface: bool = False,
                proj_kwargs: dict = {},
                **kwargs
                ):
        '''
        1. render projected (from spatial point sampling) iso-points for DSS loss
        2. use ray-traced points in space for sdf loss (p_freespace and p_occupancy)
        3. projected and ray-traced points that are inside the mask_img for photo-consistency loss

        Args:
            mask_img (tensor): (N, 1, H, W) ground truth mask
                used to get mask value of the iso-points
            img_gt (tensor): mask values at pixels (N, C, H, W)
            pixels (tensor): sampled pixels (N, P, 2)
            masked_gt (tensor): mask values at pixels (N, P)
            inputs (tensor): input for encoder (like 2D image)
            cameras (Camera): used to project and unproject points in the current view
            it (int): training iteration (used for ray sampling scheduler)
        Returns:
            point_clouds (PointClouds3D): with rgb and normals for the iso-surface points, which are
                found via projection and pixel-to-space ray-tracing
            rgb_pred (tensor): (N3, 3) rgb of the visible iso points (from both project and ray-traced points)
            p_freespace (tensor): (N1, 3) combining detached iso-points that are outside the gt mask and
                unsuccessfully ray-traced points that are outside the gt mask
            p_occupancy (tensor): (N2, 3) combining detached iso-points that are inside the gt mask and
                unsuccessfully ray-traced points that are inside the gt mask
            sdf_freespace (tensor): (N1,)
            sdf_occupancy (tensor): (N2,)
            img_pred      (tensor): (N, H, W, 3)
            img_mask_pred (tensor): (N, H, W, 1)
            rgb_gt        (tensor): (N3, 3)
        '''
        batch_size = cameras.R.shape[0]

        # encode inputs
        c = self.encode_inputs(inputs)

        # 1.1) Sample from the iso-surface via differentiable points projection
        # iso_points: all projected iso-points
        # onsurface_from_isopoints: inmask, incamera, not occluded iso-points
        # offsurface_from_isopoints: off-surface points on the cam-isopoints ray that are outside the mask
        # insurface_from_isopoints: off-surface points on the cam-isopoints ray that are insidet the mask
        onsurface_from_isopoints = offsurface_from_isopoints = insurface_from_isopoints = torch.zeros(
            [0, 3], device=self.device, dtype=torch.float)
        num_ons_iso_per_batch = \
            num_iso_free_per_batch = \
            num_iso_occ_per_batch = \
            onsurface_from_isopoints.new_zeros(
                [batch_size], dtype=torch.long)

        if project:
            iso_points_pcl = self.get_visible_iso_points(cameras, **proj_kwargs)

            onsurface_from_isopoints, num_ons_iso_per_batch = self.sample_onsurface_using_isopoints(
                iso_points_pcl, mask_img, cameras)

        if sample_iso_offsurface:
            max_insurface_per_batch = num_ons_iso_per_batch // 10
            # max_insurface_per_batch[max_insurface_per_batch > (pixels.shape[1] // 10)] = (pixels.shape[1] // 10)
            offsurface_from_isopoints, \
                insurface_from_isopoints, \
                    num_iso_free_per_batch, num_iso_occ_per_batch = self.sample_offsurface_using_isopoints(
                pixels, mask_img, cameras, max_insurface_per_batch=max_insurface_per_batch, iso_pcl=iso_points_pcl)

        # 1.2) Sample points in the space via ray-tracing as freespace and occupancy points
        # a) combine ray-traced points with projected points to get the iso-points
        # b) combine the projected points to the freespace and occupancy points
        # project iso_points to image
        # p_world: (N0, 3)
        # mask_pred: (B, P) succesful ray-traced among p_world
        # p_freespace: (N1, 3) ray-traced offsurface points by sampling image that are outside the mask
        # p_occupancy: (N2, 3) ray-traced offsurface points by sampling image that are inside the mask
        onsurface_from_pixels = p_freespace = p_occupancy = torch.zeros(
            (0, 3), dtype=torch.float, device=self.device)
        num_ons_pixel_per_batch = num_occ_per_batch = num_free_per_batch = torch.zeros(
            (batch_size,), dtype=torch.long, device=self.device)

        if pixels is not None and pixels.nelement() > 0:
            # sphere-tracing find intersection
            mask_gt = get_tensor_values(
                mask_img.float(), pixels, squeeze_channel_dim=True).bool()

            # Fall back to using ray-tracing if no iso-points is found
            if onsurface_from_isopoints.nelement() == 0:
                # ############ DVR ##############
                # onsurface_from_pixels, mask_pred = self.pixels_to_world(
                #     pixels, cameras, c, it=it, **kwargs)
                # onsurface_from_pixels = onsurface_from_pixels[mask_pred & mask_gt]
                # p_freespace, num_free_per_batch, p_occupancy, num_occ_per_batch = self.sample_world_points(
                #     pixels, cameras, n_points_per_ray=self.n_points_per_ray,
                #     mask_gt=mask_gt, mask_pred=mask_pred, c=c)

                # ############ IDR ##############
                onsurface_from_pixels, p_freespace, p_occupancy, mask_pred, num_free_per_batch, num_occ_per_batch = self.sample_from_pixels(
                    pixels, cameras, mask_gt, **kwargs)
                onsurface_from_pixels = onsurface_from_pixels[mask_pred & mask_gt]

                if self.training and isinstance(self.texture, LightingTexture):
                    onsurface_from_pixels = self.sampling(self.decoder, onsurface_from_pixels.detach())
                num_ons_pixel_per_batch = (mask_pred & mask_gt).sum(-1)

            if offsurface_from_isopoints.nelement() == 0:
                p_freespace, num_free_per_batch, p_occupancy, num_occ_per_batch = self.sample_world_points(
                    pixels, cameras, n_points_per_ray=self.n_points_per_ray,
                    mask_gt=mask_gt, mask_pred=mask_pred, c=c)

            # # b) combine the projected points to the freespace and occupancy points
            if offsurface_from_isopoints.nelement() > 0:
                p_freespace = torch.cat([torch.cat([x1, x2], dim=0) for x1, x2 in zip(torch.split(
                    p_freespace, num_free_per_batch.tolist(), dim=0), torch.split(offsurface_from_isopoints, num_iso_free_per_batch.tolist(), dim=0))], dim=0)
            if insurface_from_isopoints.nelement() > 0:
                p_occupancy = torch.cat([torch.cat([x1, x2], dim=0) for x1, x2 in zip(torch.split(
                    p_occupancy, num_occ_per_batch.tolist(), dim=0), torch.split(insurface_from_isopoints, num_iso_occ_per_batch.tolist(), dim=0))], dim=0)
            num_free_per_batch = num_iso_free_per_batch + num_free_per_batch
            num_occ_per_batch = num_iso_occ_per_batch + num_occ_per_batch

        elif sample_iso_offsurface:
            p_freespace = offsurface_from_isopoints
            p_occupancy = insurface_from_isopoints
            num_occ_per_batch = num_iso_occ_per_batch
            num_free_per_batch = num_iso_free_per_batch

        if self.training and get_debugging_mode():
            self._debugger_logging(
                p_freespace, num_free_per_batch, p_occupancy, num_occ_per_batch)


        # 3. losses: SDF losses and photo-consistency loss
        # 3.1) photo-consistency loss for iso-points that are visible and inside the image masks
        # one part from the sampled pointclouds the other part from ray-tracing
        # 3.1.1) from projection
        # 3.1.2) from ray-tracing
        rgb_gt = rgb_pred = img_gt.new_zeros((0, 3))
        if self.training and get_debugging_mode():
            dbg_tensor = get_debugging_tensor()
            if onsurface_from_isopoints.nelement() > 0:
                dbg_tensor.pts_world['dvr_iso_points'] = torch.split(
                    onsurface_from_isopoints.cpu().detach(), num_ons_iso_per_batch.cpu().tolist())
                handle = onsurface_from_isopoints.register_hook(_save_grad_for_pcl_with_name(
                    'dvr_iso_points', num_points_per_cloud=num_ons_iso_per_batch.cpu().tolist()))
                self.hooks.append(handle)
            if onsurface_from_pixels.nelement() > 0:
                dbg_tensor.pts_world['dvr_ray'] = torch.split(
                    onsurface_from_pixels.cpu().detach(), num_ons_pixel_per_batch.cpu().tolist())
                handle = onsurface_from_pixels.register_hook(_save_grad_for_pcl_with_name(
                    'dvr_ray', num_points_per_cloud=num_ons_pixel_per_batch.tolist()))
                self.hooks.append(handle)

        pointclouds_iso_visible = PointClouds3D([], ).to(self.device)
        rgb_gt = onsurface_from_pixels.new_empty((0, 3))
        pixel_pred = onsurface_from_pixels.new_empty((0, 2))
        # merge two sets of onsurface points
        # get appearance from texture model
        if onsurface_from_pixels.nelement() > 0 or onsurface_from_isopoints.nelement() > 0:
            # iso_visible_list = torch.split(onsurface_from_isopoints, num_ons_iso_per_batch.tolist(), dim=0)
            # iso_visible_padded = list_to_padded(iso_visible_list)
            # iso_visible_mask = torch.full(
            #     (batch_size, iso_visible_padded.shape[1]), False, dtype=torch.bool, device=iso_visible_padded.device)
            # for b in range(batch_size):
            #     iso_visible_mask[b, :iso_visible_list[b].shape[0]] = True
            iso_visible_list = [torch.cat([x1, x2], dim=0) for x1, x2 in zip(torch.split(
                onsurface_from_pixels, num_ons_pixel_per_batch.tolist(), dim=0), torch.split(onsurface_from_isopoints, num_ons_iso_per_batch.tolist(), dim=0))]
            iso_visible_padded = list_to_padded(iso_visible_list)
            iso_visible_mask = torch.full(
                (batch_size, iso_visible_padded.shape[1]), False, dtype=torch.bool, device=iso_visible_padded.device)
            for b in range(batch_size):
                iso_visible_mask[b, :iso_visible_list[b].shape[0]] = True

            if iso_visible_padded.requires_grad:
                # iso_visible_padded.register_hook(lambda x: x.clamp(-0.1, 0.1))
                iso_visible_padded.register_hook(
                    lambda x: clip_norm(x, dim=-1, max_value=0.1))

            pointclouds_iso_visible = self.get_point_clouds(
                iso_visible_padded, mask=iso_visible_mask, c=c, with_colors=True, with_normals=True, require_normals_grad=True,
                debug_name_prefix='texture_all_', cameras=cameras, lights=lights)
            rgb_pred = pointclouds_iso_visible.features_packed()
            # get ground truth
            p_screen_hat = cameras.transform_points(
                pointclouds_iso_visible.points_padded())
            pixel_pred = -p_screen_hat[..., :2]
            rgb_gt = get_tensor_values(
                img_gt.float(), pixel_pred)
            rgb_gt = padded_to_packed(rgb_gt, pointclouds_iso_visible.cloud_to_packed_first_idx(
            ), pointclouds_iso_visible.num_points_per_cloud().sum().item())
            pixel_pred = padded_to_packed(pixel_pred, pointclouds_iso_visible.cloud_to_packed_first_idx(
            ), pointclouds_iso_visible.num_points_per_cloud().sum().item())
            assert(rgb_gt.shape[0] == rgb_pred.shape[0])

        self.points = pointclouds_iso_visible.points_packed().detach().view(1, -1, 3)

        # 3.2) Calc SDF values for mask losses (freespace and occupancy losses)
        sdf_freespace = p_freespace.new_empty(p_freespace.shape[:-1] + (1,))
        sdf_occupancy = p_occupancy.new_empty(p_freespace.shape[:-1] + (1,))
        if p_freespace is not None and p_freespace.nelement() > 0:
            out_freespace = self.decode(p_freespace, only_base=True)
            sdf_freespace = out_freespace.sdf
        if p_occupancy is not None and p_occupancy.nelement() > 0:
            out_occupancy = self.decode(p_occupancy, only_base=True)
            sdf_occupancy = out_occupancy.sdf

        return {'iso_pcl': pointclouds_iso_visible,
                'iso_pixel': pixel_pred,
                'iso_rgb': rgb_pred,
                'p_freespace': p_freespace,
                'p_occupancy': p_occupancy,
                'sdf_freespace': sdf_freespace,
                'sdf_occupancy': sdf_occupancy,
                'iso_rgb_gt': rgb_gt}

    def render(self, p_world=None, cameras=None, lights=None, mode='both', **kwargs):
        """ Render predicted intersections
        Args:
            cameras: pytorch3d cameras

        Returns:
            rgba_imgs: (N,H,W,4)
        """
        cameras = cameras or self.cameras
        batch_size = cameras.R.shape[0]

        pointclouds = self.get_point_clouds(
            p_world, with_colors=False, with_normals=True, cameras=cameras, lights=lights)

        if batch_size != len(pointclouds) and len(pointclouds) == 1:
            pointclouds = pointclouds.extend(len(cameras))

        colored_pointclouds = self.decode_color(
            pointclouds, cameras=cameras, lights=lights)

        # render
        rgba_imgs = self.renderer(
            colored_pointclouds, cameras=cameras, **kwargs)

        return rgba_imgs

    def sample_uniform_iso_points(self, init_points:Optional[Pointclouds]=None):
        """ Sample uniformly distributed iso-points
        Args:
            init_points (1, P, 3), if not given, use self._points
        Returns:
            sampled_points
        """
        return


class Generator(ImplicitGenerator, PointGenerator):
    """ Generate mesh from implicit model """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval_points(self, p, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points (N, 3)
            c (tensor): latent conditioned code c (1, C)
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.decode(pi, c, **kwargs).sdf.squeeze(-1)

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def generate_meshes(self, *args, **kwargs):
        # only return the mesh from the marching cube
        outputs = super().generate_meshes(*args, **kwargs)
        return outputs[1:]
