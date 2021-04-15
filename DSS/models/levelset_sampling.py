from typing import Optional, Tuple, Union, Callable
from collections import namedtuple
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import frnn
from pytorch3d.ops import (
    knn_points, knn_gather, convert_pointclouds_to_tensor, packed_to_padded)
from pytorch3d.ops.knn import _KNN
from pytorch3d.renderer.utils import convert_to_tensors_and_broadcast
from pytorch3d.structures import Pointclouds, list_to_padded, list_to_packed, padded_to_list
from .. import logger_py
from ..utils import gather_batch_to_packed, reduce_mask_padded, mask_padded_to_list, intersection_with_unit_sphere, num_points_2_cloud_to_packed_first_idx
from ..utils.mathHelper import eps_denom, eps_sqrt
from ..utils.point_processing import upsample, wlop
"""
Various functions for iso-surface projection
"""

ProjectionResult = namedtuple('ProjectionResult', ('points', 'normals', 'mask'))


def _convert_batched_to_packed_args(*args, latent=None):
    """
    Convert possible batched inputs to packed
    Args:
          list of broadcastable (N, *, cj) tensors and latent: (N, C)
    Returns:
          list of reshaped (M, cj) 2-dim inputs and latent (M, C)
    """
    if len(args) == 0:
        if latent is not None:
            latent.squeeze_()
        return args + (latent,)
    device = args[0].device
    args = convert_to_tensors_and_broadcast(*args, device=device)
    assert(all([x.shape[0] == args[0].shape[0] for x in args]))

    if latent is not None:
        latent.squeeze_()
        assert(latent.ndim == 2)
        if args[0].ndim > 2:
            batch_size = args[0].shape[0]
            num_intermediate = math.prod(args[0].shape[1:-1])
            args = [x.view(-1, x.shape[-1]) for x in args]

            first_idx = torch.repeat_interleave(
                torch.arange(batch_size, device=device),
                num_intermediate,
                dim=0)
            latent = gather_batch_to_packed(latent, first_idx)

    args = [x.view(-1, x.shape[-1]) for x in args]
    return args + [latent]

def _filter_projection_result(result: ProjectionResult) -> ProjectionResult:
    """ filter the invalid projection result """
    points, normals, mask = result
    points = reduce_mask_padded(points, mask)
    normals = reduce_mask_padded(normals, mask)
    mask = reduce_mask_padded(mask, mask)
    return ProjectionResult(points, normals, mask)

class LevelSetProjection(object):

    def __init__(self, proj_max_iters=10, proj_tolerance=5.0e-5, max_points_per_pass=120000,
                 ):
        self.proj_max_iters = proj_max_iters
        self.proj_tolerance = proj_tolerance
        self.max_points_per_pass = max_points_per_pass

    def project_points(self, points_init, network, latent, levelset):
        raise NotImplementedError


class UniformProjection(LevelSetProjection):
    """
    Project sampled points uniformly to the current zero-levelset
    First repulse points, then project to iso-surface `proj_max_iters` times.
    Repeat `sample_iters` times.
    Attributes:
        knn_k: used to find neighborhood
        sigma_p: variance weight spatial kernel (the larger the smoother)
        sigma_n: variance for normal kernel (the smaller the more feature-focused)
        total_iters: one iteration consissts of proj_max_iters projections +
            sample_iters resampling
    """

    def __init__(self, proj_max_iters=10, proj_tolerance=5e-5,
                 max_points_per_pass=120000,
                 sample_iters=1,
                 knn_k=8, resampling_clip=0.02, **kwargs):
        """
        Args:
            knn_k: used for neighborhood search for resampling and upsampling
            sample_iters: number of iterations per resampling
            resampling_clip: larger -> allow more feature prominant, but can be more unstable
        """
        super().__init__(proj_max_iters=proj_max_iters,
                         proj_tolerance=proj_tolerance,
                         max_points_per_pass=max_points_per_pass,
                         )
        self.knn_k = knn_k
        self.sample_iters = sample_iters
        self.resampling_clip = resampling_clip

    def _create_tree(self, points_padded: torch.Tensor, refresh_tree=True, num_points_per_cloud=None):
        """
        create a data structure, per-point cache knn-neighbor
        Args:
            points_padded (N,P,D)
            num_points_per_cloud list
        """
        if not refresh_tree and hasattr(self, '_knn_idx') and self._knn_idx is not None:
            return self._knn_idx
        assert(points_padded.ndim == 3)
        if num_points_per_cloud is None:
            num_points_per_cloud = torch.tensor([points_padded.shape[1]] * points_padded.shape[0],
                                                device=points_padded.device, dtype=torch.long)
        # knn_result = knn_points(
        #     points_padded, points_padded, num_points_per_cloud, num_points_per_cloud,
        #     K=self.knn_k + 1, return_nn=True, return_sorted=True)
        # self._knn_idx = knn_result.idx[..., 1:]
        # self._knn_dists = knn_result.dists[..., 1:]
        # self._knn_nn = knn_result.knn[..., 1:, :]
        diag = (points_padded.max(dim=1).values -
                points_padded.min(dim=1).values).norm(dim=-1)
        search_radius = torch.sqrt(diag / num_points_per_cloud.float()) * self.knn_k
        dists, idxs, nn, grid = frnn.frnn_grid_points(points_padded, points_padded,
                                                      num_points_per_cloud, num_points_per_cloud, K=self.knn_k + 1, r=search_radius,
                                                      grid=None, return_nn=True)
        self._knn_gather = frnn.frnn_gather
        self._knn_idx = idxs[..., 1:]
        self._knn_dists = dists[..., 1:]
        self._knn_nn = nn[..., 1:, :]
        self.knn_gather = frnn.frnn_gather
        return self._knn_idx

    def _compute_sdf_and_grad(self, points, model, latent=None, **forward_kwargs) -> Tuple[torch.Tensor]:
        """
        Evalute sdf and compute grad in splits
        Args:
            points: (N, *, 3)
        Returns sdf (N, *) and grad (N,*,D)
        """
        shp = points.shape
        points_packed = points.view(-1, 3)

        grad_packed = []
        eval_packed = []
        with autograd.no_grad():
            model.eval()
            for sub_points in torch.split(points_packed, self.max_points_per_pass, dim=0):
                net_input = sub_points
                with autograd.enable_grad():
                    net_input.detach_().requires_grad_(True)
                    network_eval = model.forward(
                        net_input, **forward_kwargs).sdf
                    input_grad = autograd.grad([network_eval], [net_input], torch.ones_like(
                        network_eval), retain_graph=False)[0]

                grad_packed.append(input_grad)
                eval_packed.append(network_eval)

            grad_packed = torch.cat(grad_packed, dim=0).view(shp)
            eval_packed = torch.cat(eval_packed, dim=0).view(shp[:-1])
        return eval_packed, grad_packed

    def insert(self, ref_pcl, points, num_points, current_knn_result=None):
        """ Insert points around high value ref_pcl points """
        batch_size = points.shape[0]
        diag = (points.view(-1, 3).max(dim=0).values -
                points.view(-1, 3).min(0).values).norm().item()
        avg_spacing = math.sqrt(diag / ref_pcl.num_points_per_cloud().item())
        patch_size = 8
        knn_k = patch_size
        search_radius = min(avg_spacing * knn_k, 0.2)
        if current_knn_result is None:
            dists, idxs, nn, _ = frnn.frnn_grid_points(points, points,
                                                          num_points, num_points, K=knn_k+1, r=search_radius,
                                                          grid=None, return_nn=True)
            current_knn_result = _KNN(
                dists=dists[..., 1:], idx=idxs[..., 1:], knn=nn[..., 1:, ...])
        try:
            metrics = ref_pcl.features_packed()
            num_ref = metrics.shape[0]
            threshold = min(metrics.median() * 2, metrics.max() * 0.5)
            assert(len(ref_pcl) == 1), "Support only 1 point cloud"

            # assert(num_points.shape[0] == 1), "Support only 1 point cloud"
            ref_pts = ref_pcl.points_packed()[(
                metrics > threshold).squeeze(-1)].view(1, -1, 3)
            if ref_pts.shape[1] == 0 or ref_pts.shape[1] > min(50, int(num_ref / 20)):
                ref_pts = ref_pcl.points_packed()[metrics.sort(
                    dim=0).indices[-max(min(50, int(num_ref / 20)), 1):, 0]].view(1, -1, 3).expand(batch_size, -1, -1)

            dists_to_ref, idxs_to_ref, nn_to_ref, _ = frnn.frnn_grid_points(
                points, ref_pts.expand(batch_size, -1, -1), lengths1=num_points, lengths2=None,
                K=1, return_nn=True, grid=None, r=search_radius*4)
            dists_to_ref = dists_to_ref.view(batch_size, -1)

            # current_knn_result.dists[current_knn_result.idx < 0] = 0.0
            # # B,
            # dist_threshold = torch.sum(current_knn_result.dists[...,0], dim=1)/torch.sum(current_knn_result.idx[...,0] >= 0, dim=1).float()
            dist_threshold = avg_spacing**2
            # B, P
            father_pts_mask = (dists_to_ref < 4*dist_threshold) & (dists_to_ref > 0)
            # P_all, 3
            father_pts = points[father_pts_mask]
            mother_pts = frnn.frnn_gather(
                points, current_knn_result.idx[..., -patch_size:])
            mother_pts = mother_pts[father_pts_mask]

            child_pts = 2 * father_pts.unsqueeze(-2) / 3 + mother_pts / 3
            child_per_batch = father_pts_mask.sum(-1) * mother_pts.shape[-2]
            child_pts = child_pts.view(-1, 3)
            cloud_to_packed_first_idx = F.pad(
                child_per_batch, (1, 0), 'constant', 0)
            cloud_to_packed_first_idx = cloud_to_packed_first_idx.cumsum(0)
            child_pts = packed_to_padded(
                child_pts, cloud_to_packed_first_idx[:-1], child_per_batch.max().item())

        except Exception as e:
            logger_py.error("Error occurred during insertion {}".format(e))
            child_pts = points.new_zeros((batch_size, 0, 3))
            child_per_batch = num_points.new_zeros((batch_size,))
        finally:
            points = torch.cat((points, child_pts), dim=1)
            num_points = num_points + child_per_batch
            return points, num_points, child_pts, child_per_batch

    def upsample(self, points, n_points, model, num_points=None, **forward_kwargs):
        points, num_points = upsample(points, n_points, num_points=num_points, neighborhood_size=31)
        return points, num_points

    def resample(self, model, points_init, normals_init, num_points, sample_iters=None, **forward_kwargs) -> ProjectionResult:
        """ resample sample_iters times """
        sample_iters = sample_iters or self.sample_iters

        batch_size = points_init.shape[0]
        if num_points is None:
            num_points = torch.full(
                (batch_size, ), points_init.shape[1], dtype=torch.long, device=points_init.device)

        if sample_iters == 0:
            return ProjectionResult(points_init, normals_init, points_init.new_full(points_init.shape[:-1], True, dtype=torch.bool))

        if points_init.nelement() < 2 * (self.knn_k + 1):
            return ProjectionResult(points_init, normals_init, points_init.new_full(points_init.shape[:-1], True, dtype=torch.bool))

        diag = (points_init.view(-1, 3).max(dim=0).values -
                points_init.view(-1, 3).min(0).values).norm().item()
        inv_sigma_spatial = num_points / diag

        points = points_init
        normals = F.normalize(normals_init, dim=-1)
        for sample_iter in range(sample_iters):
            if sample_iter % 2 == 0:
                # assume repulsion doesn't change neighborhood
                self._create_tree(points, refresh_tree=True,
                                  num_points_per_cloud=num_points)
                current_knn_result = _KNN(
                    dists=self._knn_dists, idx=self._knn_idx, knn=self._knn_nn)

            nn_normals = self.knn_gather(
                normals, current_knn_result.idx, num_points)
            knn_nn = self.knn_gather(
                points, current_knn_result.idx, num_points)
            knn_diff = points[:, :, None, :] - knn_nn
            knn_dists = torch.sum(knn_diff**2, dim=-1)
            spatial_w = torch.exp(-knn_dists * inv_sigma_spatial)
            spatial_w[current_knn_result.idx < 0] = 0
            density_w = torch.sum(spatial_w, dim=-1, keepdim=True) + 1.0

            pts_diff_proj = knn_diff - \
                (knn_diff * nn_normals).sum(dim=-1, keepdim=True) * nn_normals

            move = torch.sum(spatial_w[..., None] * pts_diff_proj, dim=-2)
            move = density_w * torch.sum(spatial_w[..., None] * pts_diff_proj, dim=-2) / eps_denom(torch.sum(spatial_w, dim=-1, keepdim=True))

            points = points + move
            projection_result = self._project_points(
                    model, points, num_points, proj_max_iters=3, **forward_kwargs)

        return projection_result

    def _project_points(self, model: Callable, points: torch.Tensor, num_points: torch.Tensor,
                        proj_max_iters: int=None, proj_tolerance:float = None, **forward_kwargs) -> ProjectionResult:
        """
        Args:
            model: SDF model
            points: (B, P, 3) padded tensor
            num_points: (B,) number of points in each batch
            proj_max_iters: maximal number of projection
            proj_tolerance: tolerace as a successful projection
        Returns:
            points (B, P, 3)
            normals (B, P, 3)
            mask (B, P)
        """
        proj_max_iters = proj_max_iters or self.proj_max_iters
        proj_tolerance = proj_tolerance or self.proj_tolerance

        points_packed, _, _, _ = list_to_packed(
            padded_to_list(points, num_points.tolist()))
        not_converged = points.new_full(
            points_packed.shape[:-1], True, dtype=torch.bool)
        normals_packed = torch.zeros_like(points_packed)
        it = 0
        while True:
            # 4. project points to iso surface
            curr_points = points_packed[not_converged]

            # compute normal grad and project
            # torch.cuda.synchronize()
            # t0 = time.time()
            curr_sdf, curr_grad = self._compute_sdf_and_grad(
                curr_points, model, **forward_kwargs)
            normals_packed[not_converged] = curr_grad
            # torch.cuda.synchronize()
            # t1 = time.time()
            # print(time.time() - t0)
            curr_not_converged = curr_sdf.squeeze(
                -1).abs() > proj_tolerance
            not_converged[not_converged] = curr_not_converged
            if (~not_converged).all() or it == proj_max_iters:
                break
            it += 1
            # Eq.4
            active_grad = curr_grad[curr_not_converged]
            active_sdf = curr_sdf[curr_not_converged]
            active_pts = curr_points[curr_not_converged]
            sum_square_grad = torch.sum(
                active_grad ** 2, dim=-1, keepdim=True)
            move = active_sdf.view(active_pts.shape[:-1] + (1,)) * \
                (active_grad / eps_denom(sum_square_grad, 1.0e-17))
            move = F.normalize(move, dim=-1, eps=1e-15) * \
                move.norm(dim=-1, keepdim=True).clamp_max(0.1)
            points_packed[not_converged] = active_pts - move

        valid_packed = ~not_converged

        points = packed_to_padded(points_packed, num_points_2_cloud_to_packed_first_idx(num_points), num_points.max().item())
        normals = packed_to_padded(normals_packed, num_points_2_cloud_to_packed_first_idx(num_points), num_points.max().item())

        valid_mask = packed_to_padded(valid_packed.view(-1, 1).float(),
            num_points_2_cloud_to_packed_first_idx(num_points), num_points.max().item()).squeeze(-1).bool()
        return ProjectionResult(points, normals, valid_mask)

    def project_points(self, point_clouds: Union[Pointclouds, torch.Tensor],
                       model: nn.Module,
                       normals_init: Optional[torch.Tensor] = None,
                       skip_resampling: bool = False,
                       skip_upsampling: bool = False,
                       ref_pcl: Optional['Pointclouds'] = None,
                       proj_max_iters: Optional[int] = None,
                       sample_iters: Optional[int] = None,
                       **forward_kwargs):
        """
        repulse and project points, no early convergence because the repulsion term measures
        Args:
            point_clouds: (N,P,D) padded points
            model: nn.Module
            latent: (N,C) latent code in minibatches or None
        Args:
            levelset_points                  projected iso-points (N, P, D)
            levelset_points_Dx               gradient of the projected iso-points (N, P, D)
            network_eval_on_levelset_points  sdf value (N, P)
            mask                             iso-points is valid (sdf < threshold)
        """
        points_init, num_points = convert_pointclouds_to_tensor(point_clouds)
        num_points_init = num_points

        proj_max_iters = proj_max_iters or self.proj_max_iters
        sample_iters = sample_iters or self.sample_iters

        if normals_init is None and isinstance(point_clouds, Pointclouds):
            normals_init = point_clouds.normals_padded()

        from collections import defaultdict
        # times = defaultdict(lambda: 0.0)
        # t00 = time.time()
        with autograd.no_grad():
            points_projected = points_init
            valid_projection = torch.full(points_projected.shape[:-1], True, device=points_projected.device, dtype=torch.bool)
                # torch.cuda.synchronize()
                # t0 = time.time()
            # TODO return padded
            points_projected, normals_projected, valid_projection = self._project_points(
                model, points_projected, num_points, proj_max_iters=proj_max_iters, **forward_kwargs)
            # torch.cuda.synchronize()
            # times['projection'] += time.time() - t0
            if not valid_projection.any():
                return {
                    'levelset_points': points_projected,
                    'mask': valid_projection}

            # resample uniformly
            if not skip_resampling:
                points_projected, normals_projected, valid_projection = _filter_projection_result(
                    ProjectionResult(points_projected, normals_projected, valid_projection))
                num_points = valid_projection.sum(dim=-1)

                points_projected, normals_projected, valid_projection = self.resample(model, points_projected, normals_projected, num_points,
                                                                      sample_iters=sample_iters, **forward_kwargs)
                num_points = valid_projection.sum(dim=-1)

            if not skip_upsampling and ref_pcl is not None:
                points_projected, normals_projected, valid_projection = _filter_projection_result(
                    ProjectionResult(points_projected, normals_projected, valid_projection))
                num_points = valid_projection.sum(dim=-1)

                _, _, new_points, num_new_points = self.insert(
                    ref_pcl, points_projected, num_points)

                new_points_projected, new_normals_projected, new_valid_projection = self._project_points(
                    model, new_points, num_new_points, proj_max_iters=10, **forward_kwargs)

                points_projected = torch.cat([points_projected, new_points_projected], dim=1)
                normals_projected = torch.cat([normals_projected, new_normals_projected], dim=1)
                valid_projection = torch.cat([valid_projection, new_valid_projection], dim=1)

            elif not skip_upsampling:
                points_projected, normals_projected, valid_projection = _filter_projection_result(
                    ProjectionResult(points_projected, normals_projected, valid_projection))
                num_points = valid_projection.sum(dim=-1)

                points_projected, num_points = self.upsample(
                    points_projected, num_points_init, model, num_points, **forward_kwargs)
                points_projected, normals_projected, valid_projection = self._project_points(
                    model, points_projected, num_points, proj_max_iters=10, **forward_kwargs)

            return {'levelset_points': points_projected,
                    'levelset_normals': normals_projected,
                    'mask': valid_projection
                    }


class EdgeAwareProjection(UniformProjection):
    """
    Sample and project away from the edge
    """

    def __init__(self, proj_max_iters=10, proj_tolerance=5e-5,
                 max_points_per_pass=120000,
                 knn_k=31, repulsion_mu=0.5,
                 sample_iters=5, total_iters=1,
                 sharpness_angle=15, edge_sensitivity=1, resampling_clip=0.02,
                 upsample_ratio=1.5, **kwargs):
        """
        Args:
            sigma_p, sigma_n: used to compound a feature x = [p/sigma_p, n/sigma_n]
            resampling_clip: larger -> allow more feature prominant, but can be more unstable
        """
        super().__init__(sample_iters=sample_iters,
                         total_iters=total_iters,
                         resampling_clip=resampling_clip,
                         knn_k=knn_k,
                         proj_max_iters=proj_max_iters,
                         proj_tolerance=proj_tolerance,
                         max_points_per_pass=max_points_per_pass,
                         )

        self.sharpness_sigma = 1 - math.cos(sharpness_angle / 180 * math.pi)
        self.repulsion_mu = repulsion_mu
        self.edge_sensitivity = edge_sensitivity
        self.upsample_ratio = upsample_ratio

    def _create_tree(self, points_padded: torch.Tensor, refresh_tree=True, num_points_per_cloud=None):
        """
        create a data structure, per-point cache knn-neighbor
        Args:
            points_padded (N,P,D)
            num_points_per_cloud list
        """
        if not refresh_tree and hasattr(self, '_knn_idx') and self._knn_idx is not None:
            return self._knn_idx
        assert(points_padded.ndim == 3)
        if num_points_per_cloud is None:
            num_points_per_cloud = torch.tensor([points_padded.shape[1]] * points_padded.shape[0],
                                                device=points_padded.device, dtype=torch.long)
        knn_result = knn_points(
            points_padded, points_padded, num_points_per_cloud, num_points_per_cloud,
            K=self.knn_k + 1, return_nn=True, return_sorted=True)
        self._knn_idx = knn_result.idx[..., 1:]
        self._knn_dists = knn_result.dists[..., 1:]
        self._knn_nn = knn_result.knn[..., 1:, :]
        self.knn_gather = knn_gather
        # search_radius = torch.min(4 * math.sqrt(2.0 / num_points_per_cloud) * self.knn_k, 0.2)
        # dists, idxs, nn, grid = frnn.frnn_grid_points(points_padded, points_padded,
        #                                               num_points_per_cloud, num_points_per_cloud, K=self.knn_k + 1, r=search_radius, grid=None, return_nn=True)
        # self._knn_idx = idxs
        # self._knn_dists = dists
        # self._knn_nn = nn
        return self._knn_idx

    def denoise_normals(self, points, normals, num_points, **kwargs):
        """
        Weights exp(-(1-<n, n_i>)/(1-cos(sharpness_sigma))), for i in a local neighborhood
        """
        normals = F.normalize(normals, dim=-1)
        # filter out
        knn_normals = self.knn_gather(normals, self._knn_idx, num_points)
        # knn_normals = frnn.frnn_gather(normals, self._knn_idx, num_points)
        self.sharpness_sigma = kwargs.get(
            'sharpness_sigma', self.sharpness_sigma)
        weights_n = ((1 - torch.sum(knn_normals *
                                    normals[:, :, None, :], dim=-1)) / self.sharpness_sigma)**2
        weights_n = torch.exp(-weights_n)

        inv_sigma_spatial = num_points / 2.0
        spatial_dist = 16 / inv_sigma_spatial
        deltap = self._knn_nn - points[:, :, None, :]
        deltap = torch.sum(deltap * deltap, dim=-1)
        weights_p = torch.exp(-deltap * inv_sigma_spatial)
        weights_p[deltap > spatial_dist] = 0
        weights = weights_p * weights_n
        # weights[self._knn_idx < 0] = 0
        normals_denoised = torch.sum(knn_normals * weights[:, :, :, None], dim=-2) / \
            eps_denom(torch.sum(weights, dim=-1, keepdim=True))
        normals_denoised = F.normalize(normals_denoised, dim=-1)
        return normals_denoised, weights_p, weights_n

    def upsample(self, points, n_points, model, num_points=None, **forward_kwargs):
        """
        points_proj: (N,P,3)
        normals_denoised: (N,P,3) remains fixed
        inv_sigma_spatial: (N,) density constant from the original point clouds
        """
        upsample_ratio = forward_kwargs.get(
            'upsample_ratio', self.upsample_ratio)
        n_points = n_points * upsample_ratio
        if isinstance(n_points, torch.Tensor):
            n_points = n_points.ceil().long()
        else:
            n_points = int(math.ceil(n_points))

        batch_size = points.shape[0]
        if num_points is None:
            num_points = torch.full(
                (batch_size, ), points.shape[1], dtype=torch.long, device=points.device)

        # 0. create knn for input points (keep fixed)
        self._create_tree(points, refresh_tree=True,
                          num_points_per_cloud=num_points)

        inv_sigma_spatial = num_points / 2.0
        spatial_dist = 16 / inv_sigma_spatial

        # 1. compute normals
        _, normals = self._compute_sdf_and_grad(
            points, model, **forward_kwargs)
        normals = F.normalize(normals, dim=-1, eps=1e-15)

        normals_denoised, weights_p, weights_n = self.denoise_normals(
            points, normals, num_points)
        normals = normals_denoised

        # 2. LOP projection
        current_knn_result = _KNN(
            dists=self._knn_dists, idx=self._knn_idx, knn=self._knn_nn)
        # dists, idxs, nn, grid = frnn.frnn_grid_points(points_proj, points_proj, num_points, num_points, K=self.knn_k + 1,
        #                       r=torch.sqrt(spatial_dist), return_nn=True)
        # current_knn_result = _KNN(dists=dists, idx=idxs, knn=nn)
        move_clip = current_knn_result.dists[..., 0].mean().sqrt()
        # (optional) search knn in the original points
        # e(-(<n, p-pi>)^2/sigma_p)
        weight_lop = torch.exp(-torch.sum(normals[:, :, None, :] *
                                          (points[:, :, None, :] - self._knn_nn), dim=-1)**2 * inv_sigma_spatial)
        weight_lop[self._knn_dists > spatial_dist] = 0
        # weight_lop[self._knn_idx < 0] = 0
        deltap = current_knn_result.dists
        spatial_w = torch.exp(-deltap * inv_sigma_spatial)
        spatial_w[deltap > spatial_dist] = 0
        # spatial_w[self._knn_idx[...,1:] <0] = 0
        density_w = torch.sum(spatial_w, dim=-1) + 1.0
        move_data = torch.sum(
            weight_lop[..., None] * (points[:, :, None, :] - self._knn_nn), dim=-2) / \
            eps_denom(torch.sum(weight_lop, dim=-1, keepdim=True))
        move_repul = self.repulsion_mu * density_w[..., None] * torch.sum(spatial_w[..., None] * (
            current_knn_result.knn - points[:, :, None, :]), dim=-2) / \
            eps_denom(torch.sum(spatial_w, dim=-1, keepdim=True))
        move_repul = F.normalize(
            move_repul) * move_repul.norm(dim=-1, keepdim=True).clamp_max(move_clip)
        move_data = F.normalize(
            move_data) * move_data.norm(dim=-1, keepdim=True).clamp_max(move_clip)
        move = move_data + move_repul

        # import trimesh
        # trimesh.Trimesh(vertices=points_proj.cpu().detach()[0].numpy(), vertex_normals=-move_data.cpu().detach()[0].numpy(), process=False).export('move_data.ply', vertex_normal=True)
        # trimesh.Trimesh(vertices=points_proj.cpu().detach()[0].numpy(), vertex_normals=-move_repul.cpu().detach()[0].numpy(), process=False).export('move_repul.ply', vertex_normal=True)
        # trimesh.Trimesh(vertices=points_proj.cpu().detach()[0].numpy(), vertex_normals=-move.cpu().detach()[0].numpy(), process=False).export('move.ply', vertex_normal=True)
        # import pdb
        # pdb.set_trace()
        points = points - move

        n_remaining = n_points - num_points
        batch_size, P = points.shape[:2]
        max_P = P // 10
        while True:
            if (n_remaining == 0).all():
                break
            knn_pts = self.knn_gather(
                points, current_knn_result.idx, num_points)
            knn_normals = self.knn_gather(
                normals, current_knn_result.idx, num_points)

            # sparse_knn_normals = frnn.frnn_gather(
            #     normals_init, knn_result.idx, num_points)[:, 1:]
            # get all mid points
            mid_points = (knn_pts + 2 * points[..., None, :]) / 3
            # N,P,K,K,3
            mid_nn_diff = mid_points.unsqueeze(-2) - knn_pts.unsqueeze(-3)
            # neighborhood edge
            dot_product = (2 - torch.sum(normals.unsqueeze(-2)
                                         * knn_normals, dim=-1))**self.edge_sensitivity
            # minimize among all the neighbors
            min_dist2 = torch.norm(mid_nn_diff, dim=-1)  # N,P,K,K
            min_dist2 = min_dist2 - \
                torch.sum(
                    (mid_nn_diff * knn_normals.unsqueeze(-2))**2, dim=-1)
            min_dist2 = min_dist2.min(dim=-1)[0]  # N,P,K
            min_dist2 = eps_sqrt(min_dist2).sqrt()
            father_sparsity, father_nb = (
                dot_product * min_dist2).max(dim=-1)  # N,P
            sparsity_sorted = father_sparsity.sort(dim=1).indices
            n_new_points = n_remaining.clone()
            n_new_points[n_new_points > max_P] = max_P
            sparsity_sorted = sparsity_sorted[:, -max_P:]

            new_pts = torch.gather(mid_points[torch.arange(mid_points.shape[0]), torch.arange(mid_points.shape[1]), father_nb], 1,
                                   sparsity_sorted.unsqueeze(-1).expand(-1, -1, 3))
            # import trimesh; trimesh.Trimesh(vertices=new_pts.cpu().detach()[0].numpy(), process=False).export('new_pts.ply')
            # import trimesh; from ..utils import scaler_to_color; colors = scaler_to_color(father_sparsity[0].cpu().numpy().reshape(-1));trimesh.Trimesh(vertices=points.cpu().detach()[0].numpy(), vertex_colors=colors, process=False).export('priority.ply')
            # import pdb; pdb.set_trace()

            total_pts_list = []
            for b, pts_batch in enumerate(padded_to_list(points, num_points.tolist())):
                total_pts_list.append(
                    torch.cat([new_pts[b][-n_new_points[b]:], pts_batch], dim=0))

            points = list_to_padded(total_pts_list)
            n_remaining = n_remaining - n_new_points
            num_points = n_new_points + num_points

            # update tree
            self._create_tree(
                points, num_points_per_cloud=num_points, refresh_tree=True)
            current_knn_result = _KNN(
                dists=self._knn_dists, idx=self._knn_idx, knn=self._knn_nn)

            # compute normals
            _, normals = self._compute_sdf_and_grad(
                points, model, **forward_kwargs)
            normals = F.normalize(normals, dim=-1)

        return points, num_points


class SphereTracing(LevelSetProjection):
    def __init__(self, proj_max_iters=10, proj_tolerance=5e-5,
                 max_points_per_pass=120000,
                 alpha=1.0, radius=1.0, padding=0.1, **kwargs):
        """
        Args:
            sigma_p, sigma_n: used to compound a feature x = [p/sigma_p, n/sigma_n]
        """
        super().__init__(proj_max_iters=proj_max_iters,
                         proj_tolerance=proj_tolerance,
                         max_points_per_pass=max_points_per_pass,
                         )
        self.alpha = alpha
        self.radius = radius
        self.padding = padding

    def project_points(self, ray0: torch.Tensor, ray_direction: torch.Tensor,
                       model: nn.Module,
                       latent: Optional[torch.Tensor] = None,
                       **forward_kwargs):
        """
        Args:
            ray0: (N,*,D) points
            ray_direction: (N,*,D) normalized ray direction
            model: nn.Module
            latent: (N, C) latent code or None

        Returns:
            {
            'levelset_points':      projected levelset points
            'levelset_points_Dx':   moore-penrose pseudo inverse (will be used in the sampling layer again)
            'network_eval_on_levelset_points':  network prediction of the projected levelset points
            'mask':                 mask of valid projection (SDF value< threshold) (N,P)
            }
        """
        shp = ray0.shape
        # to packed form
        ray0, ray_direction, latent = _convert_batched_to_packed_args(
            ray0, ray_direction, latent=latent)

        N, D = ray0.shape

        ray0_list = torch.split(ray0, self.max_points_per_pass, dim=0)
        ray_direction_list = torch.split(
            ray_direction, self.max_points_per_pass, dim=0)

        # change c from batch to per point
        if latent is not None and latent.nelement() > 0:
            latent_list = torch.split(latent, self.max_points_per_pass, dim=0)
        else:
            latent_list = [None] * len(ray0_list)

        levelset_points = []
        levelset_points_Dx = []
        eval_on_levelset_points = []

        with autograd.no_grad():
            # process points in sub-batches
            for sub_points_init, sub_ray_direction, sub_latent in zip(
                    ray0_list, ray_direction_list, latent_list):
                curr_projection = sub_points_init.clone()
                num_points, points_dim = sub_points_init.shape[:2]
                active_mask = sub_points_init.new_full(
                    [num_points, ], True, dtype=torch.bool)

                trials = 0
                model.eval()

                # mask the points that are still inside a unit sphere *after*
                # ray-marching
                inside_sphere = torch.full(
                    sub_ray_direction.shape[:-1], True, dtype=torch.bool,
                    device=sub_ray_direction.device)
                while True:
                    # evaluate sdf
                    net_input = curr_projection[active_mask]
                    if sub_latent is not None:
                        current_latent = sub_latent[active_mask]
                    else:
                        current_latent = sub_latent
                    with autograd.enable_grad():
                        if trials == 0:
                            net_input.detach_().requires_grad_(True)
                            network_eval = model.forward(
                                net_input, c=current_latent, **forward_kwargs).sdf
                            grad = autograd.grad(
                                network_eval, net_input, torch.ones_like(network_eval), retain_graph=True
                            )[0].detach()
                        else:
                            net_input.detach_().requires_grad_(True)
                            network_eval_active = model.forward(
                                net_input, c=current_latent, **forward_kwargs).sdf
                            grad_active = autograd.grad([network_eval_active], [net_input], torch.ones_like(
                                network_eval_active), retain_graph=True)[0]

                            grad[active_mask] = grad_active.detach()
                            network_eval[active_mask] = network_eval_active.detach()

                    active_mask = (network_eval.abs() > 1e-1 * self.proj_tolerance).squeeze(1) \
                        & inside_sphere

                    # project not converged points to iso-surface
                    if ((active_mask).any() and trials < self.proj_max_iters):
                        network_eval_active = network_eval[active_mask]
                        points_active = curr_projection[active_mask]

                        # Advance by alpha*sdf
                        move = self.alpha * network_eval_active * \
                            sub_ray_direction[active_mask]
                        move = F.normalize(move, dim=-1, eps=1e-15) * \
                            move.norm(dim=-1, keepdim=True).clamp_max(0.1)
                        points_active += move
                        inside_sphere[active_mask] = (points_active.norm(
                            dim=-1) < (self.padding + self.radius))
                        curr_projection[active_mask &
                                        inside_sphere] = points_active[inside_sphere[active_mask]]
                    else:
                        break

                    trials = trials + 1

                curr_projection.detach_()
                levelset_points.append(curr_projection)
                eval_on_levelset_points.append(network_eval.detach())
                levelset_points_Dx.append(grad.detach())

        levelset_points = torch.cat(levelset_points, dim=0)
        eval_on_levelset_points = torch.cat(eval_on_levelset_points, dim=0)
        levelset_points_Dx = torch.cat(levelset_points_Dx, dim=0)

        valid_projection = eval_on_levelset_points.abs() <= self.proj_tolerance

        # if (not valid_projection.all() and self.proj_max_iters > 0):
        #     abs_sdf = eval_on_levelset_points.abs()
        #     logger_py.info("[SphereTracing] success rate: {:.2f}% ,  max : {:.3g} , min {:.3g} , mean {:.3g} , std {:.3g}"
        #                    .format(valid_projection.sum().item() * 100 / valid_projection.nelement(),
        #                            torch.max(torch.abs(abs_sdf)),
        #                            torch.min(torch.abs(abs_sdf)),
        #                            torch.mean(abs_sdf),
        #                            torch.std(abs_sdf)))

        return {'levelset_points': levelset_points.view(shp),
                'network_eval_on_levelset_points': eval_on_levelset_points.view(shp[:-1]),
                'levelset_points_Dx': levelset_points.view(shp),
                'mask': valid_projection.view(shp[:-1])}


class RayTracing(nn.Module):
    def __init__(
            self,
            object_bounding_sphere=1.0,
            sdf_threshold=5.0e-5,
            line_search_step=0.5,
            line_step_iters=1,
            sphere_tracing_iters=10,
            n_steps=100,
            n_secant_steps=8,
    ):
        super().__init__()

        self.object_bounding_sphere = object_bounding_sphere
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_steps = n_steps
        self.n_secant_steps = n_secant_steps

    def forward(self,
                sdf,
                cam_loc,
                object_mask,
                ray_directions
                ):
        """ cam_loc (B, 3) """
        batch_size, num_pixels, _ = ray_directions.shape

        # use sphere intersections as the boundary of the the rays.
        # sphere_intersections, mask_intersect = get_sphere_intersection(cam_loc, ray_directions, r=self.object_bounding_sphere)
        section0, section1, mask_intersect = intersection_with_unit_sphere(
            cam_loc, ray_directions, radius=self.object_bounding_sphere)
        # range of valid ray-length
        sphere_intersections = (torch.stack([section0, section1], dim=-2) - cam_loc.view(batch_size, 1, 3).unsqueeze(-2)).norm(
            dim=-1) / ray_directions.unsqueeze(-2).norm(dim=-1)

        curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis = \
            self.sphere_tracing(batch_size, num_pixels, sdf, cam_loc,
                                ray_directions, mask_intersect, sphere_intersections)

        network_object_mask = (acc_start_dis < acc_end_dis)

        # The non convergent rays should be handled by the sampler
        sampler_mask = unfinished_mask_start
        sampler_net_obj_mask = torch.zeros_like(sampler_mask).bool().cuda()
        if sampler_mask.sum() > 0:
            sampler_min_max = torch.zeros((batch_size, num_pixels, 2)).cuda()
            sampler_min_max.reshape(-1, 2)[sampler_mask,
                                           0] = acc_start_dis[sampler_mask]
            sampler_min_max.reshape(-1, 2)[sampler_mask,
                                           1] = acc_end_dis[sampler_mask]

            sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(sdf,
                                                                                cam_loc,
                                                                                object_mask,
                                                                                ray_directions,
                                                                                sampler_min_max,
                                                                                sampler_mask
                                                                                )

            curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
            acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]

        # print('----------------------------------------------------------------')
        # print('RayTracing: object = {0}/{1}, secant on {2}/{3}.'
        #       .format(network_object_mask.sum(), len(network_object_mask), sampler_net_obj_mask.sum(), sampler_mask.sum()))
        # print('----------------------------------------------------------------')

        if not self.training:
            return curr_start_points, \
                network_object_mask, \
                acc_start_dis

        ray_directions = ray_directions.reshape(-1, 3)
        mask_intersect = mask_intersect.reshape(-1)

        # in ground truth mask, no intersection detected but sphere tracing has converged
        in_mask = ~network_object_mask & object_mask & ~sampler_mask
        # not in ground truth mask but sphere tracing has converged
        out_mask = ~object_mask & ~sampler_mask

        mask_left_out = (in_mask | out_mask) & ~mask_intersect
        if mask_left_out.sum() > 0:  # project the origin to the not intersect points on the sphere
            cam_left_out = cam_loc.unsqueeze(1).repeat(
                1, num_pixels, 1).reshape(-1, 3)[mask_left_out]
            rays_left_out = ray_directions[mask_left_out]
            acc_start_dis[mask_left_out] = -torch.bmm(
                rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).squeeze()
            curr_start_points[mask_left_out] = cam_left_out + \
                acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out

        mask = (in_mask | out_mask) & mask_intersect

        if mask.sum() > 0:
            min_dis[network_object_mask &
                    out_mask] = acc_start_dis[network_object_mask & out_mask]

            min_mask_points, min_mask_dist = self.minimal_sdf_points(
                num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis)

            curr_start_points[mask] = min_mask_points
            acc_start_dis[mask] = min_mask_dist

        return curr_start_points, \
            network_object_mask, \
            acc_start_dis

    def sphere_tracing(self, batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections):
        ''' Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection '''

        sphere_intersections_points = cam_loc.reshape(
            batch_size, 1, 1, 3) + sphere_intersections.unsqueeze(-1) * ray_directions.unsqueeze(2)
        unfinished_mask_start = mask_intersect.reshape(-1).clone()
        unfinished_mask_end = mask_intersect.reshape(-1).clone()

        # Initialize start current points
        curr_start_points = torch.zeros(
            batch_size * num_pixels, 3).cuda().float()
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[:,
                                                                               :, 0, :].reshape(-1, 3)[unfinished_mask_start]
        acc_start_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1, 2)[
            unfinished_mask_start, 0]

        # Initialize end current points
        curr_end_points = torch.zeros(
            batch_size * num_pixels, 3).cuda().float()
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[:,
                                                                           :, 1, :].reshape(-1, 3)[unfinished_mask_end]
        acc_end_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1, 2)[
            unfinished_mask_end, 1]

        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start[unfinished_mask_start] = sdf(
            curr_start_points[unfinished_mask_start])

        next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
        next_sdf_end[unfinished_mask_end] = sdf(
            curr_end_points[unfinished_mask_end])

        while True:
            # Update sdf
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            curr_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (
                curr_sdf_start > self.sdf_threshold)
            unfinished_mask_end = unfinished_mask_end & (
                curr_sdf_end > self.sdf_threshold)

            if (unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end

            # Update points
            curr_start_points = (cam_loc.unsqueeze(
                1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)
            curr_end_points = (cam_loc.unsqueeze(
                1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)

            # Fix points which wrongly crossed the surface
            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = sdf(
                curr_start_points[unfinished_mask_start])

            next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            next_sdf_end[unfinished_mask_end] = sdf(
                curr_end_points[unfinished_mask_end])

            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (
                    2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(
                    batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_start]

                acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (
                    2 ** not_proj_iters)) * curr_sdf_end[not_projected_end]
                curr_end_points[not_projected_end] = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(
                    batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_end]

                # Calc sdf
                next_sdf_start[not_projected_start] = sdf(
                    curr_start_points[not_projected_start])
                next_sdf_end[not_projected_end] = sdf(
                    curr_end_points[not_projected_end])

                # Update mask
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (
                acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (
                acc_start_dis < acc_end_dis)

        return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis

    def ray_sampler(self, sdf, cam_loc, object_mask, ray_directions, sampler_min_max, sampler_mask):
        ''' Sample the ray in a given range and run secant on rays which have sign transition '''

        batch_size, num_pixels, _ = ray_directions.shape
        n_total_pxl = batch_size * num_pixels
        sampler_pts = torch.zeros(n_total_pxl, 3).cuda().float()
        sampler_dists = torch.zeros(n_total_pxl).cuda().float()

        intervals_dist = torch.linspace(
            0, 1, steps=self.n_steps).cuda().view(1, 1, -1)

        pts_intervals = sampler_min_max[:, :, 0].unsqueeze(-1) + intervals_dist * (
            sampler_min_max[:, :, 1] - sampler_min_max[:, :, 0]).unsqueeze(-1)
        points = cam_loc.reshape(
            batch_size, 1, 1, 3) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)

        # Get the non convergent rays
        mask_intersect_idx = torch.nonzero(
            sampler_mask, as_tuple=False).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 80000, dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).cuda().float(
        ).reshape((1, self.n_steps))  # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1)
        sampler_pts[mask_intersect_idx] = points[torch.arange(
            points.shape[0]), sampler_pts_ind, :]
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(
            pts_intervals.shape[0]), sampler_pts_ind]

        true_surface_pts = object_mask[sampler_mask]
        net_surface_pts = (sdf_val[torch.arange(
            sdf_val.shape[0]), sampler_pts_ind] < 0)

        # take points with minimal SDF value for P_out pixels
        p_out_mask = ~(true_surface_pts & net_surface_pts)
        n_p_out = p_out_mask.sum()
        if n_p_out > 0:
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask,
                                                                 :, :][torch.arange(n_p_out), out_pts_idx, :]
            sampler_dists[mask_intersect_idx[p_out_mask]
                          ] = pts_intervals[p_out_mask, :][torch.arange(n_p_out), out_pts_idx]

        # Get Network object mask
        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False

        # Run Secant method
        secant_pts = net_surface_pts & true_surface_pts if self.training else net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[torch.arange(
                pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            sdf_high = sdf_val[torch.arange(
                sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][torch.arange(
                n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][torch.arange(
                n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            cam_loc_secant = cam_loc.unsqueeze(1).repeat(
                1, num_pixels, 1).reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            ray_directions_secant = ray_directions.reshape(
                (-1, 3))[mask_intersect_idx[secant_pts]]
            z_pred_secant = self.secant(
                sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant, sdf)

            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = cam_loc_secant + \
                z_pred_secant.unsqueeze(-1) * ray_directions_secant
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions, sdf):
        ''' Runs the secant method for interval [z_low, z_high] for n_secant_steps '''

        z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for i in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid)
            ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]

            z_pred = - sdf_low * (z_high - z_low) / \
                (sdf_high - sdf_low) + z_low

        return z_pred

    def minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis):
        ''' Find points with minimal SDF value on rays for P_out pixels '''

        n_mask_points = mask.sum()

        n = self.n_steps
        # steps = torch.linspace(0.0, 1.0,n).cuda()
        steps = torch.empty(n).uniform_(0.0, 1.0).cuda()
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(
            n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        mask_points = cam_loc.unsqueeze(1).repeat(
            1, num_pixels, 1).reshape(-1, 3)[mask]
        mask_rays = ray_directions[mask, :]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []
        for pnts in torch.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts))

        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        min_vals, min_idx = mask_sdf_all.min(-1)
        min_mask_points = mask_points_all.reshape(
            -1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1,
                                      n)[torch.arange(0, n_mask_points), min_idx]

        return min_mask_points, min_mask_dist


class SampleNetwork(nn.Module):
    """
    Eq.13 in the paper
    """

    def forward(self, network: nn.Module,
                levelset_points: torch.Tensor,
                return_eval: bool = False):
        """
        Args:
            levelset_points: (n, *, d) leaf nodes on the level set (from projection), packed points
            levelset_points_Dx: (n, *, d) grad(network, levelset_points)
            network_eval_on_levelset_points: (n, *, 1) the SDF value of the levelset points
        """
        levelset_points = levelset_points.detach()
        with autograd.enable_grad():
            levelset_points = levelset_points.requires_grad_(True)
            network_eval = network.forward(levelset_points).sdf
            levelset_points_Dx = autograd.grad([network_eval], [levelset_points], torch.ones_like(
                network_eval), retain_graph=True)[0]

        levelset_points = levelset_points.detach()
        levelset_points_Dx = levelset_points_Dx.detach()

        # is it necessary to pass in network_eval_on_levelset_points?
        network_eval = network.forward(levelset_points).sdf
        sum_square_grad = torch.sum(
            levelset_points_Dx ** 2, dim=-1, keepdim=True)

        # network_eval_on_levelset_points (bxnxl)   := c, independent of theta (constant)
        # network_eval                    (bxnxl)   := F(p; theta)
        # levelset_points_Dx              (bxlxnxd) := D_xF(p;theta_0)^{+} moore-penrose pseudo-inverse Eq.5
        sampled_points = levelset_points - (
            network_eval - network_eval.detach()).view(levelset_points.shape[:-1] + (1,)) * (
            levelset_points_Dx / eps_denom(sum_square_grad, 1e-17))
        if return_eval:
            return sampled_points, network_eval
        return sampled_points


def find_zero_crossing_between_point_pairs(p0: Optional[torch.Tensor], p1: Optional[torch.Tensor],
                                           network: torch.nn.Module,
                                           n_secant_steps=8,
                                           n_steps=100,
                                           is_occupancy=True,
                                           max_points=80000,
                                           c: Optional[torch.Tensor] = None,
                                           allow_in_to_out=False,
                                           **forward_kwargs):
    '''
    Args:
        p0 (tensor): (N, *, 3)
        p1 (tensor): (N, *, 3)
        network (nn.Module): sdf evaluator
        n_steps (int): number of evaluation steps; if the difference between
            n_steps[0] and n_steps[1] is larger then 1, the value is sampled
            in the range
        n_secant_steps (int): number of secant refinement steps
        max_points (int): max number of points loaded to GPU memory
        c (tensor): (N,C)
    Returns:
        pt_pred (tensor): (N, *, 3)
        mask (tensor): (N, *) boolean tensor mask valid zero crossing (sign change
            & from outside to inside & doesn't start from being inside )
    '''
    def _compare_func(is_occupancy, tau_logit=0.0):
        def less_than(data):
            return data < tau_logit

        def greater_than(data):
            return data > tau_logit
        if is_occupancy:
            return less_than
        else:
            return greater_than

    compare_func = _compare_func(is_occupancy)
    device = p0.device
    shp = p0.shape
    p0, p1, c = _convert_batched_to_packed_args(p0, p1, latent=c)
    n_pts, D = p0.shape

    # Prepare d_proposal and p_proposal in form (b_size, n_pts, n_steps, 3)
    # d_proposal are "proposal" depth values and p_proposal the
    # corresponding "proposal" 3D points
    ray_direction = F.normalize(
        p1 - p0, p=2, dim=-1, eps=1e-10)

    d_proposal = torch.linspace(0, 1, steps=n_steps).view(
        1, n_steps).to(device) * torch.norm(p1 - p0, p=2, dim=-1).unsqueeze(-1)

    p_proposal = p0.unsqueeze(-2) + \
        ray_direction.unsqueeze(-2) * d_proposal.unsqueeze(-1)

    # Evaluate all proposal points in parallel
    with torch.no_grad():
        p_proposal = p_proposal.view(-1, 3)
        p_proposal_list = torch.split(p_proposal, max_points, dim=0)
        if c is not None:
            c = c.view(n_pts, 1, c.shape[-1]).expand(n_pts,
                                                     p_proposal.shape[1], c.shape[-1]).view(-1, c.shape[-1])
            c_list = torch.split(c, max_points, dim=0)
        else:
            c_list = [None] * len(p_proposal_list)

        val = torch.cat([network.forward(p_split, c=c_split, **forward_kwargs).sdf
                         for p_split, c_split in zip(p_proposal_list, c_list)],
                        dim=0).view(n_pts, n_steps)

    # Create mask for valid points where the first point is not occupied
    mask_0_not_occupied = compare_func(val[..., 0])

    # Calculate if sign change occurred and concat 1 (no sign change) in
    # last dimension
    sign_matrix = torch.cat([torch.sign(val[..., :-1] * val[..., 1:]),
                             torch.ones(n_pts, 1).to(device)],
                            dim=-1)
    cost_matrix = sign_matrix * torch.arange(n_steps, 0, -1).float().to(device)

    # Get first sign change and mask for values where
    # a.) a sign changed occurred and
    # b.) no a neg to pos sign change occurred (meaning from inside surface to outside)
    # NOTE: for sdf value b.) becomes from pos to neg
    values, indices = torch.min(cost_matrix, -1)
    mask_sign_change = values < 0
    mask_out_to_in = compare_func(val[torch.arange(n_pts), indices])

    # Define mask where a valid depth value is found
    if allow_in_to_out:
        mask = mask_sign_change
    else:
        mask = mask_sign_change & mask_out_to_in

    # Get depth values and function values for the interval
    # to which we want to apply the Secant method
    # Again, for SDF decoder d_low is actually d_high
    d_start = d_proposal[torch.arange(n_pts), indices.view(n_pts)][mask]
    f_start = val[torch.arange(n_pts), indices.view(n_pts)][mask]
    indices = torch.clamp(indices + 1, max=n_steps - 1)
    d_end = d_proposal[torch.arange(n_pts), indices.view(n_pts)][mask]
    f_end = val[torch.arange(n_pts), indices.view(n_pts)][mask]

    p0_masked = p0[mask]
    ray_direction_masked = ray_direction[mask]

    # write c in pointwise format
    if c is not None and c.shape[-1] != 0:
        c = c.unsqueeze(1).repeat(1, n_pts, 1).view(-1, c.shape[-1])[mask]

    # Apply surface depth refinement step (e.g. Secant method)
    p_pred = run_Secant_method(
        f_start, f_end, d_start, d_end, n_secant_steps, p0_masked,
        ray_direction_masked, network, c, **forward_kwargs)

    # for sanity
    pt_pred = torch.ones(mask.shape + (3,)).to(device)
    pt_pred[mask] = p_pred
    pt_pred = pt_pred.view(shp)
    return pt_pred, mask.view(shp[:-1])


def run_Secant_method(f_start, f_end, d_start, d_end, n_secant_steps,
                      p0, ray_direction, decoder, c,
                      **forward_kwargs):
    ''' Runs the secant method for interval [d_start, d_end].

    Args:
        f_start(tensor): (N, *)
        f_end(tensor): (N, *)
        d_start (tensor): (N,*) start values for the interval
        d_end (tensor): (N,*) end values for the interval
        n_secant_steps (int): number of steps
        p0 (tensor): masked ray start points
        ray_direction_masked (tensor): masked ray direction vectors
        decoder (nn.Module): decoder model to evaluate point occupancies
        c (tensor): latent conditioned code c
    '''
    d_pred = - f_start * (d_end - d_start) / (f_end - f_start) + d_start
    for i in range(n_secant_steps):
        p_mid = p0 + d_pred.unsqueeze(-1) * ray_direction
        with torch.no_grad():
            f_mid = decoder.forward(p_mid, c, **forward_kwargs).sdf
            f_mid = f_mid.squeeze(-1)
        # ind_start masks f_mid has the same sign as d_start
        # if decoder outputs sdf, d_start (start) is > 0,
        ind_start = torch.eq(torch.sign(f_mid), torch.sign(f_start))
        if ind_start.sum() > 0:
            d_start[ind_start] = d_pred[ind_start]
            f_start[ind_start] = f_mid[ind_start]
        if (ind_start == 0).sum() > 0:
            d_end[ind_start == 0] = d_pred[ind_start == 0]
            f_end[ind_start == 0] = f_mid[ind_start == 0]

        d_pred = - f_start * (d_end - d_start) / (f_end - f_start) + d_start

    p_pred = p0 + \
        d_pred.unsqueeze(-1) * ray_direction
    return p_pred


class DirectionalSamplingNetwork(SampleNetwork):
    def forward(self, network, iso_points, ray, cam_pos, c=None, return_eval: bool = False):
        """ points can be (N,*,3) """
        shp = ray.shape
        iso_points = iso_points.detach()

        # compute Dx
        with autograd.enable_grad():
            iso_points.requires_grad_(True)
            network_eval = network.forward(iso_points).sdf
            iso_points_Dx = autograd.grad([network_eval], [iso_points], torch.ones_like(
                network_eval), retain_graph=True)[0]

        iso_points_Dx = iso_points_Dx.detach()
        iso_points = iso_points.detach()

        surface_dists = (iso_points - cam_pos).norm(dim=-1, keepdim=True)
        # (N, *, 1)
        network_eval = network.forward(iso_points).sdf
        ray = F.normalize(ray, dim=-1, p=2)
        ray0 = ray.detach()
        # t -> t(theta)
        surface_points_dot = torch.sum(
            iso_points_Dx * ray0, dim=-1, keepdim=True)
        surface_dists_theta = surface_dists - \
            (network_eval - network_eval.detach()) / \
            eps_denom(surface_points_dot, 1e-10)

        # t(theta) -> x(theta,c,v)
        surface_points_theta_c_v = cam_pos + surface_dists_theta * ray

        if return_eval:
            return surface_points_theta_c_v, network_eval
        return surface_points_theta_c_v

def sample_uniform_iso_points(model: nn.Module, n_points: int,
                              init_points: Optional[torch.Tensor]=None,
                              bounding_sphere_radius: float = 1.0
                              ) -> Pointclouds:
    """
    Sample uniformly distributed iso-points
    Returns:
        iso_points (1, n_points, 3)
    """
    # project
    projector = UniformProjection(
        max_points_per_pass=16000, proj_max_iters=10,
        proj_tolerance=5e-5, knn_k=8)

    # only projection
    if init_points is None:
        init_points = (torch.rand((1, n_points*4, 3)) - 0.5) * 2 * bounding_sphere_radius
        init_points = init_points.cuda()

    # first projection
    proj_results = projector.project_points(init_points, model, skip_resampling=True, skip_upsampling=True)
    # remove out-of-boundary points
    boundary_mask = proj_results['levelset_points'].norm(dim=-1) < bounding_sphere_radius
    # padded_to_list returns tuple
    proj_pcl = Pointclouds(list(mask_padded_to_list(proj_results['levelset_points'], proj_results['mask'] & boundary_mask.view_as(proj_results['mask']))))

    # wlop uniform resample
    wlop_result = wlop(proj_pcl, min(0.5, n_points/proj_pcl.num_points_per_cloud().item()))
    proj_results = projector.project_points(wlop_result, model, skip_resampling=True, skip_upsampling=False)

    # upsample
    proj_pcl = Pointclouds(list(mask_padded_to_list(proj_results['levelset_points'], proj_results['mask'])))
    upsampled_pcl = upsample(proj_pcl, n_points)
    proj_results = projector.project_points(upsampled_pcl, model, skip_resampling=True, skip_upsampling=False)

    # NOTE to do exactly what the paper does, could also run the line below, which does not downsample and upsample
    # proj_results = projector.project_points(wlop_result, model, max_proj_iters=0,
    #                                         skip_resampling=False, skip_upsampling=False,
    #                                         sample_iters=3)

    return Pointclouds(list(mask_padded_to_list(proj_results['levelset_points'], proj_results['mask'])))