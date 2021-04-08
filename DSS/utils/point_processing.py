from typing import Union, Tuple
from pytorch3d.structures.pointclouds import Pointclouds
import torch
import torch.nn.functional as F
import frnn
import math
from pytorch3d.ops import (convert_pointclouds_to_tensor, is_pointclouds)
from pytorch3d.ops.knn import _KNN, knn_points
from pytorch3d.structures import list_to_padded, padded_to_list
from .. import logger_py
from ..core.cloud import PointClouds3D, PointCloudsFilters
from . import mask_from_padding, num_points_2_packed_to_cloud_idx
from .mathHelper import eps_denom, estimate_pointcloud_normals, estimate_pointcloud_local_coord_frames


def remove_outliers(pointclouds, neighborhood_size=16, tolerance=0.05):
    """
    Identify a point as outlier if the ratio of the smallest and largest
    variance is > than a threshold
    """
    points, num_points = convert_pointclouds_to_tensor(pointclouds)
    mask_padding = mask_from_padding(num_points)
    variance, local_frame = estimate_pointcloud_local_coord_frames(
        points, neighborhood_size=neighborhood_size)
    # thres = variance[..., -1].median(dim=1)[0] * 16
    # largest
    mask = (variance[...,0] / torch.sum(variance, dim=-1)) < tolerance
    # mask = variance[...,-1] < thres
    pointclouds_filtered = PointCloudsFilters(
        device=pointclouds.device, activation=mask & mask_padding).filter(pointclouds)
    return pointclouds_filtered



def wlop(pointclouds: PointClouds3D, ratio:float=0.5, neighborhood_size=16, iters=3, repulsion_mu=0.5) -> PointClouds3D:
    """
    Consolidation of Unorganized Point Clouds for Surface Reconstruction
    Args:
        pointclouds containing max J points per cloud
        ratio: downsampling ratio (0, 1]
    """
    P, num_points_P = convert_pointclouds_to_tensor(pointclouds)
    # (N, 3, 2)
    bbox = pointclouds.get_bounding_boxes()
    # (N,)
    diag = torch.norm(bbox[..., 0] - bbox[..., 1], dim=-1)
    h = 4 * torch.sqrt(diag / num_points_P.float())
    search_radius = min(h * neighborhood_size, 0.2)
    theta_sigma_inv = 16/h/h

    if ratio < 1.0:
        X0 = farthest_sampling(pointclouds, ratio=ratio)
    elif ratio == 1.0:
        X0 = pointclouds.clone()
    else:
        raise ValueError('ratio must be less or equal to 1.0')

    # slightly perturb so that we don't find the same point when searching NN XtoP
    offset = torch.randn_like(X0.points_packed()) * h * 0.1
    X0.offset_(offset)
    X, num_points_X = convert_pointclouds_to_tensor(X0)


    def theta(r2):
        return torch.exp(-r2 * theta_sigma_inv)

    def eta(r):
        return -r

    def deta(r):
        return torch.ones_like(r)

    grid = None
    dists, idxs, _, grid = frnn.frnn_grid_points(P, P, num_points_P, num_points_P, K=neighborhood_size+1, r=search_radius, grid=grid, return_nn=False)
    knn_PtoP = _KNN(dists=dists[...,1:], idx=idxs[...,1:], knn=None)

    deltapp = torch.norm(P.unsqueeze(-2) - frnn.frnn_gather(P, knn_PtoP.idx, num_points_P), dim=-1)
    theta_pp_nn = theta(deltapp**2)  # (B, P, K)
    theta_pp_nn[knn_PtoP.idx < 0] = 0
    density_P = torch.sum(theta_pp_nn, dim=-1) + 1

    for it in range(iters):
        # from each x find closest neighbors in pointclouds
        dists, idxs, _, grid = frnn.frnn_grid_points(X, P, num_points_X, num_points_P, K=neighborhood_size, r=search_radius, grid=grid, return_nn=False)
        knn_XtoP = _KNN(dists=dists, idx=idxs, knn=None)

        dists, idxs, _, _ = frnn.frnn_grid_points(X, X, num_points_X, num_points_X, K=neighborhood_size+1, r=search_radius, grid=None, return_nn=False)
        knn_XtoX = _KNN(dists=dists[...,1:], idx=idxs[...,1:], knn=None)

        # LOP local optimal projection
        nn_XtoP = frnn.frnn_gather(P, knn_XtoP.idx, num_points_P)
        epsilon = X.unsqueeze(-2) - frnn.frnn_gather(P, knn_XtoP.idx, num_points_P)
        delta = X.unsqueeze(-2) - frnn.frnn_gather(X, knn_XtoX.idx, num_points_X)

        # (B, I, I)
        deltaxx2 = (delta ** 2).sum(dim=-1)
        # (B, I, K)
        deltaxp2 = (epsilon ** 2).sum(dim=-1)

        # (B, I, K)
        alpha = theta(deltaxp2) / eps_denom(epsilon.norm(dim=-1))
        # (B, I, K)
        beta = theta(deltaxx2) * deta(delta.norm(dim=-1)) / eps_denom(delta.norm(dim=-1))

        density_X = torch.sum(theta(deltaxx2), dim=-1) + 1

        new_alpha = alpha / frnn.frnn_gather(density_P.unsqueeze(-1), knn_XtoP.idx, num_points_P).squeeze(-1)
        new_alpha[knn_XtoP.idx < 0] = 0

        new_beta = density_X.unsqueeze(-1) * beta
        new_beta[knn_XtoX.idx < 0] = 0

        term_data = torch.sum(new_alpha[..., None] * nn_XtoP, dim=-2) / \
            eps_denom(torch.sum(new_alpha, dim=-1, keepdim=True))
        term_repul = repulsion_mu * torch.sum(new_beta[..., None] * delta, dim=-2) / \
            eps_denom(torch.sum(new_beta, dim=-1, keepdim=True))

        X = term_data + term_repul

    if is_pointclouds(X0):
        return X0.update_padded(X)
    return X



def resample_uniformly(pointclouds: Union[Pointclouds, torch.Tensor], neighborhood_size:int=8, knn=None, normals=None,
                       shrink_ratio:float=0.5, repulsion_mu:float=1.0) -> Union[Pointclouds, Tuple[torch.Tensor, torch.Tensor]]:
    """
    resample first use wlop to consolidate point clouds to a smaller point clouds (halve the points)
    then upsample with ear
    Returns:
        Pointclouds or padded points and number of points per batch
    """
    import math
    import frnn
    points_init, num_points = convert_pointclouds_to_tensor(pointclouds)
    batch_size = num_points.shape[0]

    diag = (points_init.view(-1, 3).max(dim=0).values -
            points_init.view(-1, 3).min(0).values).norm().item()
    avg_spacing = math.sqrt(diag / points_init.shape[1])
    search_radius = min(
        4 * avg_spacing * neighborhood_size, 0.2)
    if knn is None:
        dists, idxs, _, grid = frnn.frnn_grid_points(points_init, points_init,
                                                    num_points, num_points, K=neighborhood_size + 1, r=search_radius, grid=None, return_nn=False)
        knn = _KNN(dists=dists[...,1:], idx=idxs[...,1:], knn=None)

    # estimate normals
    if isinstance(pointclouds, torch.Tensor):
        normals = normals
    else:
        normals = pointclouds.normals_padded()

    if normals is None:
        normals = estimate_pointcloud_normals(points_init, neighborhood_size=neighborhood_size, disambiguate_directions=False)
    else:
        normals = F.normalize(normals, dim=-1)

    points = points_init
    wlop_result = wlop(pointclouds, ratio=shrink_ratio, repulsion_mu=repulsion_mu)
    up_result = upsample(wlop_result, num_points)

    if is_pointclouds(pointclouds):
        return up_result
    return up_result.points_padded(), up_result.num_points_per_cloud()

def project_to_latent_surface(points, normals, sharpness_angle=60, neighborhood_size=31, max_proj_iters=10, max_est_iter=5):
    """
    RIMLS
    """
    points, num_points = convert_pointclouds_to_tensor(points)
    normals = F.normalize(normals, dim=-1)
    sharpness_sigma = 1 - math.cos(sharpness_angle / 180 * math.pi)
    diag = (points.max(dim=-2)[0] - points.min(dim=-2)[0]).norm(dim=-1)
    avg_spacing = math.sqrt(diag / points.shape[1])
    search_radius = min(16 * avg_spacing * neighborhood_size, 0.2)

    dists, idxs, _, grid = frnn.frnn_grid_points(points, points,
                                                num_points, num_points, K=neighborhood_size + 1, r=search_radius, grid=None, return_nn=False)
    knn_result = _KNN(dists=dists[...,1:], idx=idxs[...,1:], knn=None)

    # knn_normals = knn_gather(normals, knn_result.idx, num_points)
    knn_normals = frnn.frnn_gather(normals, knn_result.idx, num_points)

    inv_sigma_spatial = 1/knn_result.dists[...,0]/16
    # spatial_dist = 16 / inv_sigma_spatial
    not_converged = torch.full(points.shape[:-1], True, device=points.device, dtype=torch.bool)
    itt = 0
    it = 0
    while True:
        knn_pts = frnn.frnn_gather(points, knn_result.idx, num_points)
        pts_diff = points[not_converged].unsqueeze(-2) - knn_pts[not_converged]
        fx = torch.sum(pts_diff*knn_normals[not_converged], dim=-1)
        not_converged_1 = torch.full(fx.shape[:-1], True, dtype=torch.bool, device=fx.device)
        knn_normals_1 = knn_normals[not_converged]
        inv_sigma_spatial_1 = inv_sigma_spatial[not_converged]
        f = points.new_zeros(points[not_converged].shape[:-1], device=points.device)
        grad_f = points.new_zeros(points[not_converged].shape, device=points.device)
        alpha = torch.ones_like(fx)
        for itt in range(max_est_iter):
            if itt > 0:
                alpha_old = alpha
                weights_n = ((knn_normals_1[not_converged_1] - grad_f[not_converged_1].unsqueeze(-2)).norm(dim=-1) / 0.5)**2
                weights_n = torch.exp(-weights_n)
                weights_p = torch.exp(-((fx[not_converged_1] - f[not_converged_1].unsqueeze(-1))**2*inv_sigma_spatial_1[not_converged_1].unsqueeze(-1)/4))
                alpha[not_converged_1] = weights_n * weights_p
                not_converged_1[not_converged_1] = (alpha[not_converged_1] - alpha_old[not_converged_1]).abs().max(dim=-1)[0] < 1e-4
                if not not_converged_1.any():
                    break

            deltap = torch.sum(pts_diff[not_converged_1] * pts_diff[not_converged_1], dim=-1)
            phi = torch.exp(-deltap * inv_sigma_spatial_1[not_converged_1].unsqueeze(-1))
            # phi[deltap > spatial_dist] = 0
            dphi = inv_sigma_spatial_1[not_converged_1].unsqueeze(-1)*phi

            weights = phi * alpha[not_converged_1]
            grad_weights = 2*pts_diff*(dphi * weights).unsqueeze(-1)

            sum_grad_weights = torch.sum(grad_weights, dim=-2)
            sum_weight = torch.sum(weights, dim=-1)
            sum_f = torch.sum(fx[not_converged_1] * weights, dim=-1)
            sum_Gf = torch.sum(grad_weights*fx[not_converged_1].unsqueeze(-1), dim=-2)
            sum_N = torch.sum(weights.unsqueeze(-1) * knn_normals_1[not_converged_1], dim=-2)

            tmp_f = sum_f / eps_denom(sum_weight)
            tmp_grad_f = (sum_Gf - tmp_f.unsqueeze(-1)*sum_grad_weights + sum_N) / eps_denom(sum_weight).unsqueeze(-1)
            grad_f[not_converged_1] = tmp_grad_f
            f[not_converged_1] = tmp_f

        move = f.unsqueeze(-1) * grad_f
        points[not_converged] = points[not_converged]-move
        mask = move.norm(dim=-1) > 5e-4
        not_converged[not_converged] = mask
        it = it + 1
        if not not_converged.any() or it >= max_proj_iters:
            break

    return points

def denoise_normals(points, normals, sharpness_sigma=30, knn_result=None, neighborhood_size=16):
    """
    Weights exp(-(1-<n, n_i>)/(1-cos(sharpness_sigma))), for i in a local neighborhood
    """
    points, num_points = convert_pointclouds_to_tensor(points)
    normals = F.normalize(normals, dim=-1)
    if knn_result is None:
        diag = (points.max(dim=-2)[0] - points.min(dim=-2)[0]).norm(dim=-1)
        avg_spacing = math.sqrt(diag / points.shape[1])
        search_radius = min(
            4 * avg_spacing * neighborhood_size, 0.2)

        dists, idxs, _, grid = frnn.frnn_grid_points(points, points,
                                                    num_points, num_points, K=neighborhood_size + 1, r=search_radius, grid=None, return_nn=True)
        knn_result = _KNN(dists=dists[...,1:], idx=idxs[...,1:], knn=None)
    if knn_result.knn is None:
        knn = frnn.frnn_gather(points, knn_result.idx, num_points)
        knn_result = _KNN(idx=knn_result.idx, knn=knn, dists=knn_result.dists)

    # filter out
    knn_normals = frnn.frnn_gather(normals, knn_result.idx, num_points)
    # knn_normals = frnn.frnn_gather(normals, self._knn_idx, num_points)
    weights_n = ((1 - torch.sum(knn_normals *
                                normals[:, :, None, :], dim=-1)) / sharpness_sigma)**2
    weights_n = torch.exp(-weights_n)

    inv_sigma_spatial = num_points / 2.0
    spatial_dist = 16 / inv_sigma_spatial
    deltap = knn - points[:, :, None, :]
    deltap = torch.sum(deltap * deltap, dim=-1)
    weights_p = torch.exp(-deltap * inv_sigma_spatial)
    weights_p[deltap > spatial_dist] = 0
    weights = weights_p * weights_n
    # weights[self._knn_idx < 0] = 0
    normals_denoised = torch.sum(knn_normals * weights[:, :, :, None], dim=-2) / \
        eps_denom(torch.sum(weights, dim=-1, keepdim=True))
    normals_denoised = F.normalize(normals_denoised, dim=-1)
    return normals_denoised.view_as(normals)


def upsample(pcl: Union[Pointclouds, torch.Tensor], n_points: Union[int, torch.Tensor],
             num_points=None, neighborhood_size=16, knn_result=None) -> Union[Pointclouds, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Iteratively add points to the sparsest region
    Args:
        points (tensor of [N, P, 3] or Pointclouds)
        n_points (tensor of [N] or integer): target number of points per cloud
    Returns:
        Pointclouds or (padded_points, num_points)
    """
    def _return_value(points, num_points, return_pcl):
        if return_pcl:
            points_list = padded_to_list(points, num_points.tolist())
            return pcl.__class__(points_list)
        else:
            return points, num_points

    return_pcl = is_pointclouds(pcl)
    points, num_points = convert_pointclouds_to_tensor(pcl)

    knn_k = neighborhood_size

    if not ((num_points - num_points[0]) == 0).all():
        logger_py.warn(
            "Upsampling operation may encounter unexpected behavior for heterogeneous batches")

    if num_points.sum() == 0:
        return _return_value(points, num_points, return_pcl)

    n_remaining = (n_points - num_points).to(dtype=torch.long)
    if (n_remaining <= 0).all():
        return _return_value(points, num_points, return_pcl)

    if knn_result is None:
        knn_result = knn_points(
            points, points, num_points, num_points,
            K=knn_k + 1, return_nn=True, return_sorted=True)

        knn_result = _KNN(dists=knn_result.dists[..., 1:], idx=knn_result.idx[..., 1:], knn=knn_result.knn[..., 1:, :])

    while True:
        if (n_remaining == 0).all():
            break
        # half of the points per batch
        sparse_pts = points
        sparse_dists = knn_result.dists
        sparse_knn = knn_result.knn
        batch_size, P, _ = sparse_pts.shape
        max_P = (P // 8)
        # sparse_knn_normals = frnn.frnn_gather(
        #     normals_init, knn_result.idx, num_points)[:, 1:]
        # get all mid points
        mid_points = (sparse_knn + 2 * sparse_pts[..., None, :]) / 3
        # N,P,K,K,3
        mid_nn_diff = mid_points.unsqueeze(-2) - sparse_knn.unsqueeze(-3)
        # minimize among all the neighbors
        min_dist2 = torch.norm(mid_nn_diff, dim=-1)  # N,P,K,K
        min_dist2 = min_dist2.min(dim=-1)[0]  # N,P,K
        father_sparsity, father_nb = min_dist2.max(dim=-1)  # N,P
        # neighborhood to insert
        sparsity_sorted = father_sparsity.sort(dim=1).indices
        n_new_points = n_remaining.clone()
        n_new_points[n_new_points > max_P] = max_P
        sparsity_sorted = sparsity_sorted[:, -max_P:]
        new_pts = torch.gather(mid_points[torch.arange(mid_points.shape[0]).view(-1, 1, 1), torch.arange(mid_points.shape[1]).view(1, -1, 1), father_nb.unsqueeze(-1)].squeeze(-2), 1,
                               sparsity_sorted.unsqueeze(-1).expand(-1, -1, 3))

        sparse_selected = torch.gather(sparse_pts, 1, sparsity_sorted.unsqueeze(-1).expand(-1, -1, 3))

        total_pts_list = []
        for b, pts_batch in enumerate(padded_to_list(points, num_points.tolist())):
            total_pts_list.append(
                torch.cat([new_pts[b][-n_new_points[b]:], pts_batch], dim=0))

        points = list_to_padded(total_pts_list)
        n_remaining = n_remaining - n_new_points
        num_points = n_new_points + num_points
        knn_result = knn_points(
            points, points, num_points, num_points, K=knn_k + 1, return_nn=True)
        knn_result = _KNN(dists=knn_result.dists[..., 1:], idx=knn_result.idx[..., 1:], knn=knn_result.knn[..., 1:, :])

    return _return_value(points, num_points, return_pcl)

def upsample_ear(points, normals, n_points: Union[int, torch.Tensor], num_points=None, neighborhood_size=16, repulsion_mu=0.4, edge_sensitivity=1.0):
    """
    Args:
        points (N, P, 3)
        n_points (tensor of [N] or integer): target number of points per cloud

    """
    batch_size = points.shape[0]
    knn_k = neighborhood_size
    if num_points is None:
        num_points = torch.tensor([points.shape[1]] * points.shape[0],
                                  device=points.device, dtype=torch.long)
    if not ((num_points - num_points[0]) == 0).all():
        logger_py.warn(
            "May encounter unexpected behavior for heterogeneous batches")
    if num_points.sum() == 0:
        return points, num_points

    point_cloud_diag = (points.max(dim=-2)[0] - points.min(dim=-2)[0]).norm(dim=-1)
    inv_sigma_spatial = num_points / point_cloud_diag
    spatial_dist = 16 / inv_sigma_spatial

    knn_result = knn_points(
        points, points, num_points, num_points,
        K=knn_k + 1, return_nn=True, return_sorted=True)
    # dists, idxs, nn, grid = frnn.frnn_grid_points(points_proj, points_proj, num_points, num_points, K=self.knn_k + 1,
    #                                               r=torch.sqrt(spatial_dist), return_nn=True)
    # knn_result = _KNN(dists=dists, idx=idxs, knn=nn)
    _knn_idx = knn_result.idx[..., 1:]
    _knn_dists = knn_result.dists[..., 1:]
    _knn_nn = knn_result.knn[..., 1:, :]
    move_clip = knn_result.dists[..., 1].mean().sqrt()

    # 2. LOP projection
    if denoise_normals:
        normals_denoised, weights_p, weights_n = denoise_normals(
            points, normals, num_points, knn_result=knn_result)
        normals = normals_denoised

    # (optional) search knn in the original points
    # e(-(<n, p-pi>)^2/sigma_p)
    weight_lop = torch.exp(-torch.sum(normals[:, :, None, :] *
                                        (points[:, :, None, :] - _knn_nn), dim=-1)**2 * inv_sigma_spatial)
    weight_lop[_knn_dists > spatial_dist] = 0
        # weight_lop[self._knn_idx < 0] = 0

    # spatial weight
    deltap = _knn_dists
    spatial_w = torch.exp(-deltap * inv_sigma_spatial)
    spatial_w[deltap > spatial_dist] = 0
    # spatial_w[self._knn_idx[..., 1:] < 0] = 0
    density_w = torch.sum(spatial_w, dim=-1) + 1.0
    move_data = torch.sum(
        weight_lop[..., None] * (points[:, :, None, :] - _knn_nn), dim=-2) / \
        eps_denom(torch.sum(weight_lop, dim=-1, keepdim=True))
    move_repul = repulsion_mu * density_w[..., None] * torch.sum(spatial_w[..., None] * (
        knn_result.knn[:, :, 1:, :] - points[:, :, None, :]), dim=-2) / \
        eps_denom(torch.sum(spatial_w, dim=-1, keepdim=True))
    move_repul = F.normalize(
        move_repul) * move_repul.norm(dim=-1, keepdim=True).clamp_max(move_clip)
    move_data = F.normalize(
        move_data) * move_data.norm(dim=-1, keepdim=True).clamp_max(move_clip)
    move = move_data + move_repul
    points = points - move

    n_remaining = n_points - num_points
    while True:
        if (n_remaining == 0).all():
            break
        # half of the points per batch
        sparse_pts = points
        sparse_dists = _knn_dists
        sparse_knn = _knn_nn
        batch_size, P, _ = sparse_pts.shape
        max_P = (P // 10)
        # sparse_knn_normals = frnn.frnn_gather(
        #     normals_init, knn_result.idx, num_points)[:, 1:]
        # get all mid points
        mid_points = (sparse_knn + 2 * sparse_pts[..., None, :]) / 3
        # N,P,K,K,3
        mid_nn_diff = mid_points.unsqueeze(-2) - sparse_knn.unsqueeze(-3)
        # minimize among all the neighbors
        min_dist2 = torch.norm(mid_nn_diff, dim=-1)  # N,P,K,K
        min_dist2 = min_dist2.min(dim=-1)[0]  # N,P,K
        father_sparsity, father_nb = min_dist2.max(dim=-1)  # N,P
        # neighborhood to insert
        sparsity_sorted = father_sparsity.sort(dim=1).indices
        n_new_points = n_remaining.clone()
        n_new_points[n_new_points > max_P] = max_P
        sparsity_sorted = sparsity_sorted[:, -max_P:]
        # N, P//2, 3, sparsest at the end
        new_pts = torch.gather(mid_points[torch.arange(mid_points.shape[0]), torch.arange(mid_points.shape[1]), father_nb], 1,
                               sparsity_sorted.unsqueeze(-1).expand(-1, -1, 3))
        total_pts_list = []
        for b, pts_batch in enumerate(padded_to_list(points, num_points.tolist())):
            total_pts_list.append(
                torch.cat([new_pts[b][-n_new_points[b]:], pts_batch], dim=0))

        points_proj = list_to_padded(total_pts_list)
        n_remaining = n_remaining - n_new_points
        num_points = n_new_points + num_points
        knn_result = knn_points(
            points_proj, points_proj, num_points, num_points, K=knn_k + 1, return_nn=True)
        _knn_idx = knn_result.idx[..., 1:]
        _knn_dists = knn_result.dists[..., 1:]
        _knn_nn = knn_result.knn[..., 1:, :]

    return points_proj, num_points

def farthest_sampling(point_clouds: PointClouds3D, ratio: float) -> PointClouds3D:
    """
    Args:
        point_clouds: Pointclouds object
    """
    points_packed = point_clouds.points_packed()

    from torch_cluster import fps
    packed_to_cloud_idx = num_points_2_packed_to_cloud_idx(
        point_clouds.num_points_per_cloud())
    fps_idx = fps(points_packed, packed_to_cloud_idx, ratio)
    sampled_points = points_packed[fps_idx]

    # back to pointclouds object
    point_lst = [sampled_points[packed_to_cloud_idx[fps_idx] == b]
                 for b in range(len(point_clouds))]
    sampled_point_clouds = point_clouds.__class__(point_lst)

    if (normals_packed := point_clouds.normals_packed()) is not None:
        normals_packed = normals_packed[fps_idx]
        sampled_point_clouds.update_normals_(normals_packed)

    if (features_packed := point_clouds.features_packed()) is not None:
        features_packed = features_packed[fps_idx]
        sampled_point_clouds.update_features_(features_packed)

    return sampled_point_clouds
