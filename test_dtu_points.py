from DSS.core.cloud import PointClouds3D
import os
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import argparse
from collections import defaultdict
from im2mesh.checkpoints import CheckpointIO
from DSS.models.common import Siren, SDF
from DSS.models.levelset_sampling import UniformProjection, EdgeAwareProjection
from pytorch3d.ops import knn_points, knn_gather
from DSS.utils import tolerating_collate, get_surface_high_res_mesh, scaler_to_color, valid_value_mask
from DSS.utils.point_processing import resample_uniformly, denoise_normals
from DSS.utils.io import save_ply, read_ply
from DSS.utils.mathHelper import eps_sqrt, to_homogen, estimate_pointcloud_normals, pinverse
from DSS.misc.visualize import plot_cuts
from DSS.training.losses import NormalLengthLoss
from DSS import set_deterministic_
import plotly.graph_objs as go
import frnn

set_deterministic_()


import argparse

class FooAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest+'_nondefault', True)

class Shape(object):
    """
    docstring
    """
    pass

    def __init__(self, points, n_points, normals=None):
        super().__init__()
        B, P, _ = points.shape
        assert(B==1)
        self.projection = UniformProjection(
            proj_max_iters=10, proj_tolerance=1e-5, total_iters=1, sample_iters=5, knn_k=16)
        self.ear_projection = EdgeAwareProjection(proj_max_iters=10, knn_k=16,
            proj_tolerance=1e-5, total_iters=1, resampling_clip=0.02, sample_iters=2, repulsion_mu=0.4,
            sharpness_angle=20, edge_sensitivity=1.0)
        rnd_idx = torch.randperm(P)[:n_points]
        points = points.view(-1, 3)[rnd_idx].view(1, -1, 3)
        if normals is not None:
            normals = normals.view(-1, 3)[rnd_idx].view(1, -1, 3)
        self.points = resample_uniformly(PointClouds3D(points, normals=normals), shrink_ratio=0.25, repulsion_mu=0.65, neighborhood_size=31).points_padded()

    def get_iso_points(self, points, sdf_net, ear=False, outlier_tolerance=0.01):
        if not ear:
            projection = self.projection
        else:
            # first resample uniformly
            projection = self.ear_projection
        with autograd.no_grad():
            proj_results = projection.project_points(points.view(1, -1, 3), sdf_net)
            mask_iso = proj_results['mask'].view(1, -1)
            iso_points = proj_results['levelset_points'].view(1, -1, 3)
            iso_points = iso_points[mask_iso].view(1, -1, 3)
            # iso_points = remove_outliers(iso_points, tolerance=outlier_tolerance, neighborhood_size=31).points_padded()
            return iso_points

def get_iso_bilateral_weights(points, normals, iso_points, iso_normals):
    """ find closest iso point, compute bilateral weight """
    search_radius = 0.1
    dim = iso_points.view(-1,3).norm(dim=-1).max()*2
    avg_spacing = iso_points.shape[1] / dim / 16
    dists, idxs, nn, _ = frnn.frnn_grid_points(
            points, iso_points, K=1,
            return_nn=True, grid=None, r=search_radius)
    iso_normals = F.normalize(iso_normals, dim=-1)
    iso_normals = frnn.frnn_gather(iso_normals, idxs).view(1, -1, 3)
    dists = torch.sum((nn.view_as(points) - points)*iso_normals,dim=-1)**2
    # dists[idxs<0] = 10 * search_radius **2
    # dists = dists.squeeze(-1)
    spatial_w = torch.exp(-dists*avg_spacing)
    normals = F.normalize(normals, dim=-1)
    normal_w = torch.exp(-((1-torch.sum(normals * iso_normals, dim=-1))/(1-np.cos(np.deg2rad(60))))**2)
    weight = spatial_w * normal_w
    weight[idxs.view_as(weight)<0] = 0
    if not valid_value_mask(weight).all():
        print("Illegal weights")
        breakpoint()
    return weight



def get_laplacian_weights(points, normals, iso_points, iso_normals, neighborhood_size=8):
    """
    compute distance based on iso local neighborhood
    """
    with autograd.no_grad():
        P, _ = points.view(-1, 3).shape
        search_radius = 0.15
        dim = iso_points.view(-1,3).norm(dim=-1).max()*2
        avg_spacing = iso_points.shape[1] / dim / 16
        dists, idxs, nn, _ = frnn.frnn_grid_points(
                points, iso_points, K=1,
                return_nn=True, grid=None, r=search_radius)
        nn_normals = frnn.frnn_gather(iso_normals, idxs)
        dists = torch.sum((points - nn.view_as(points))*(normals + nn_normals.view_as(normals)), dim=-1)
        dists = dists * dists
        spatial_w = torch.exp(-dists*avg_spacing)
        spatial_w[idxs.view_as(spatial_w)<0] = 0
    return spatial_w.view(points.shape[:-1])

def get_heat_kernel_weights(points, normals, iso_points, iso_normals, neighborhood_size=8, sigma_p=0.4, sigma_n=0.7):
    """
    find closest k points, compute point2face distance, and normal distance
    """
    P, _ = points.view(-1, 3).shape
    search_radius = 0.15
    dim = iso_points.view(-1,3).norm(dim=-1).max()
    avg_spacing = iso_points.shape[1] / (dim*2**2) / 16
    dists, idxs, nn, _ = frnn.frnn_grid_points(
            points, iso_points, K=neighborhood_size,
            return_nn=True, grid=None, r=search_radius)

    # features
    with autograd.no_grad():
        # normalize just to be sure
        iso_normals = F.normalize(iso_normals, dim=-1, eps=1e-15)
        normals = F.normalize(normals, dim=-1, eps=1e-15)

        # features are composite of points and normals
        features = torch.cat([points / sigma_p, normals / sigma_n], dim=-1)
        features_iso = torch.cat([iso_points / sigma_p, iso_normals / sigma_n], dim=-1)

        # compute kernels (N,P,K) k(x,xi), xi \in Neighbor(x)
        knn_idx = idxs
        # features_nb = knn_gather(features_iso, knn_idx)
        features_nb = frnn.frnn_gather(features_iso, knn_idx)
        # (N,P,K,D)
        features_diff = features.unsqueeze(2) - features_nb
        features_dist = torch.sum(features_diff**2, dim=-1)
        kernels = torch.exp(-features_dist)
        kernels[knn_idx < 0] = 0

        # N,P,K,K,D
        features_diff_ij = features_nb[:, :, :,
                                        None, :] - features_nb[:, :, None, :, :]
        features_dist_ij = torch.sum(features_diff_ij**2, dim=-1)
        kernel_matrices = torch.exp(-features_dist_ij)
        kernel_matrices[knn_idx < 0] = 0
        kernel_matrices[knn_idx.unsqueeze(-2).expand_as(kernel_matrices) < 0]
        kernel_matrices_inv = pinverse(kernel_matrices)

        weight = kernels.unsqueeze(-2) @ kernel_matrices_inv @ kernels.unsqueeze(-1)
        weight.clamp_max_(1.0)

    return weight.view(points.shape[:-1])

def gradient(points, net):
    points.requires_grad_(True)
    sdf_value = net(points)
    grad = torch.autograd.grad(sdf_value, [points], [
        torch.ones_like(sdf_value)], create_graph=True)[0]
    return grad


def run(pointcloud_path, out_dir, decoder_type='siren',
        resume=True, **kwargs):
    """
    test_implicit_siren_noisy_wNormals
    """
    device = torch.device('cuda:0')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # data
    points, normals = np.split(
        read_ply(pointcloud_path).astype('float32'), (3,), axis=1)

    pmax, pmin = points.max(axis=0), points.min(axis=0)
    scale = (pmax - pmin).max()
    pcenter = (pmax + pmin) /2
    points = (points - pcenter) / scale * 1.5
    scale_mat = scale_mat_inv = np.identity(4)
    scale_mat[[0,1,2], [0,1,2]] = 1/scale * 1.5
    scale_mat[[0,1,2], [3,3,3]] = - pcenter / scale * 1.5
    scale_mat_inv = np.linalg.inv(scale_mat)
    normals = normals @ np.linalg.inv(scale_mat[:3, :3].T)
    object_bounding_sphere = np.linalg.norm(points, axis=1).max()
    pcl = trimesh.Trimesh(
        vertices=points, vertex_normals=normals, process=False)
    pcl.export(os.path.join(out_dir, "input_pcl.ply"), vertex_normal=True)

    assert(np.abs(points).max() < 1)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(points), torch.from_numpy(normals))
    batch_size = 5000
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True,
        collate_fn=tolerating_collate,
    )
    gt_surface_pts_all = torch.from_numpy(points).unsqueeze(0).float()
    gt_surface_normals_all = torch.from_numpy(normals).unsqueeze(0).float()
    gt_surface_normals_all = F.normalize(gt_surface_normals_all, dim=-1)

    if kwargs['use_off_normal_loss']:
        # subsample from pointset
        sub_idx = torch.randperm(gt_surface_normals_all.shape[1])[:20000]
        gt_surface_pts_sub = torch.index_select(gt_surface_pts_all, 1, sub_idx).to(device=device)
        gt_surface_normals_sub = torch.index_select(gt_surface_normals_all, 1, sub_idx).to(device=device)
        gt_surface_normals_sub = denoise_normals(gt_surface_pts_sub, gt_surface_normals_sub, neighborhood_size=30)

    if decoder_type == 'siren':
        decoder_params = {
            'dim': 3,
            "out_dims": {'sdf': 1},
            "c_dim": 0,
            "hidden_size": 256,
            'n_layers': 3,
            "first_omega_0": 30,
            "hidden_omega_0": 30,
            "outermost_linear": True,
        }
        decoder = Siren(**decoder_params)
        # pretrained_model_file = os.path.join('data', 'trained_model', 'siren_l{}_c{}_o{}.pt'.format(
        #                     decoder_params['n_layers'], decoder_params['hidden_size'], decoder_params['first_omega_0']))
        # loaded_state_dict = torch.load(pretrained_model_file)
        # decoder.load_state_dict(loaded_state_dict)
    elif decoder_type == 'sdf':
        decoder_params = {
            'dim': 3,
            "out_dims": {'sdf': 1},
            "c_dim": 0,
            "hidden_size": 512,
            'n_layers': 8,
            'bias': 1.0,
        }
        decoder = SDF(**decoder_params)
    else:
        raise ValueError
    print(decoder)
    decoder = decoder.to(device)

    # training
    total_iter = 30000
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10000, 20000], gamma=0.5)


    shape = Shape(gt_surface_pts_all.cuda(), n_points=gt_surface_pts_all.shape[1]//16, normals=gt_surface_normals_all.cuda())
    # initialize siren with sphere_initialization
    checkpoint_io = CheckpointIO(out_dir, model=decoder, optimizer=optimizer)
    load_dict = dict()
    if resume:
        models_avail = [f for f in os.listdir(out_dir) if f[-3:] == '.pt']
        if len(models_avail) > 0:
            models_avail.sort()
            load_dict = checkpoint_io.load(models_avail[-1])

    it = load_dict.get('it', 0)
    if it > 0:
        try:
            iso_point_files = [f for f in os.listdir(out_dir) if f[-7:] == 'iso.ply']
            iso_point_iters = [int(os.path.basename(f[:-len('_iso.ply')])) for f in iso_point_files]
            iso_point_iters = np.array(iso_point_iters)
            idx = np.argmax(iso_point_iters[(iso_point_iters - it)<=0])
            iso_point_file = np.array(iso_point_files)[(iso_point_iters - it)<=0][idx]
            iso_points = torch.from_numpy(read_ply(os.path.join(out_dir, iso_point_file))[...,:3])
            shape.points = iso_points.to(device=shape.points.device).view(1, -1 ,3)
            print('Loaded iso-points from %s' % iso_point_file)
        except Exception as e:
            pass

    # loss
    eikonal_loss = NormalLengthLoss(reduction='mean')

    # start training
    # save_ply(os.path.join(out_dir, 'in_iso_points.ply'), (to_homogen(shape.points).cpu().detach().numpy() @ scale_mat_inv.T)[...,:3].reshape(-1,3))
    save_ply(os.path.join(out_dir, 'in_iso_points.ply'), shape.points.cpu().view(-1,3))
    # autograd.set_detect_anomaly(True)
    iso_points = shape.points
    iso_points_normal = None
    while True:
        if (it > total_iter):
            checkpoint_io.save('model_{:04d}.pt'.format(it), it=it)
            mesh = get_surface_high_res_mesh(
                lambda x: decoder(x).sdf.squeeze(), resolution=512)
            mesh.apply_transform(scale_mat_inv)
            mesh.export(os.path.join(out_dir, "final.ply"))
            break

        for batch in data_loader:

            gt_surface_pts, gt_surface_normals = batch
            gt_surface_pts.unsqueeze_(0)
            gt_surface_normals.unsqueeze_(0)
            gt_surface_pts = gt_surface_pts.to(device=device).detach()
            gt_surface_normals = gt_surface_normals.to(
                device=device).detach()

            optimizer.zero_grad()
            decoder.train()
            loss = defaultdict(float)

            lambda_surface_sdf = 1e3
            lambda_surface_normal = 1e2
            if kwargs['warm_up'] >= 0 and it >= kwargs['warm_up']:
                lambda_surface_sdf = kwargs['lambda_surface_sdf']
                lambda_surface_normal = kwargs['lambda_surface_normal']

            # debug
            if (it - kwargs['warm_up']) % 1000 == 0:
                # generate iso surface
                with torch.autograd.no_grad():
                    box_size = (object_bounding_sphere * 2 + 0.2, ) * 3
                    imgs = plot_cuts(lambda x: decoder(x).sdf.squeeze().detach(),
                                     box_size=box_size, max_n_eval_pts=10000, thres=0.0,
                                     imgs_per_cut=1, save_path=os.path.join(out_dir, '%010d_iso.html' % it))
                    mesh = get_surface_high_res_mesh(
                        lambda x: decoder(x).sdf.squeeze(), resolution=200)
                    mesh.apply_transform(scale_mat_inv)
                    mesh.export(os.path.join(out_dir, '%010d_mesh.ply' % it))

            if it % 2000 == 0:
                checkpoint_io.save('model.pt', it=it)

            pred_surface_grad = gradient(gt_surface_pts.clone(), lambda x: decoder(x).sdf)

            # every once in a while update shape and points
            # sample points in space and on the shape
            # use iso points to weigh data points loss
            weights = 1.0
            if kwargs['warm_up'] >= 0 and it >= kwargs['warm_up']:
                if it == kwargs['warm_up'] or kwargs['resample_every'] > 0 and (it - kwargs['warm_up']) % kwargs['resample_every'] == 0:
                    # if shape.points.shape[1]/iso_points.shape[1] < 1.0:
                    #     idx = fps(iso_points.view(-1,3), torch.zeros(iso_points.shape[1], dtype=torch.long, device=iso_points.device), shape.points.shape[1]/iso_points.shape[1])
                    #     iso_points = iso_points.view(-1,3)[idx].view(1,-1,3)

                    iso_points = shape.get_iso_points(iso_points+0.1*(torch.rand_like(iso_points)-0.5), decoder, ear=kwargs['ear'], outlier_tolerance=kwargs['outlier_tolerance'])
                    # iso_points = shape.get_iso_points(shape.points, decoder, ear=kwargs['ear'], outlier_tolerance=kwargs['outlier_tolerance'])
                    iso_points_normal = estimate_pointcloud_normals(iso_points.view(1,-1,3), 8, False)
                    if kwargs['denoise_normal']:
                        iso_points_normal = denoise_normals(iso_points, iso_points_normal, num_points=None)
                        iso_points_normal = iso_points_normal.view_as(iso_points)
                elif iso_points_normal is None:
                    iso_points_normal = estimate_pointcloud_normals(iso_points.view(1,-1,3), 8, False)

                # iso_points = resample_uniformly(iso_points.view(1,-1,3))
                # TODO: use gradient from network or neighborhood?
                iso_points_g = gradient(iso_points.clone(), lambda x: decoder(x).sdf)
                if it == kwargs['warm_up'] or kwargs['resample_every'] > 0 and (it - kwargs['warm_up']) % kwargs['resample_every'] == 0:
                    # save_ply(os.path.join(out_dir, '%010d_iso.ply' % it), (to_homogen(iso_points).cpu().detach().numpy() @ scale_mat_inv.T)[...,:3].reshape(-1,3), normals=iso_points_g.view(-1,3).detach().cpu())
                    save_ply(os.path.join(out_dir, '%010d_iso.ply' % it), iso_points.cpu().detach().view(-1,3), normals=iso_points_g.view(-1,3).detach().cpu())

                if kwargs['weight_mode'] == 1:
                    weights = get_iso_bilateral_weights(gt_surface_pts, gt_surface_normals, iso_points, iso_points_g).detach()
                elif kwargs['weight_mode'] == 2:
                    weights = get_laplacian_weights(gt_surface_pts, gt_surface_normals, iso_points, iso_points_g).detach()
                elif kwargs['weight_mode'] == 3:
                    weights = get_heat_kernel_weights(gt_surface_pts, gt_surface_normals, iso_points, iso_points_g).detach()

                if (it - kwargs['warm_up']) % 1000 == 0 and kwargs['weight_mode'] != -1:
                    print("min {:.4g}, max {:.4g}, std {:.4g}, mean {:.4g}".format(weights.min(), weights.max(), weights.std(), weights.mean()))
                    colors = scaler_to_color(1-weights.view(-1).cpu().numpy(), cmap='Reds')
                    save_ply(os.path.join(out_dir, '%010d_batch_weight.ply' % it), (to_homogen(gt_surface_pts).cpu().detach().numpy() @ scale_mat_inv.T)[...,:3].reshape(-1,3),
                        colors=colors)

                sample_idx = torch.randperm(iso_points.shape[1])[:min(batch_size, iso_points.shape[1])]
                iso_points_sampled = iso_points.detach()[:, sample_idx, :]
                # iso_points_sampled = iso_points.detach()
                iso_points_sdf = decoder(iso_points_sampled.detach()).sdf
                loss_iso_points_sdf = iso_points_sdf.abs().mean()* kwargs['lambda_iso_sdf'] * iso_points_sdf.nelement() / (iso_points_sdf.nelement()+8000)
                loss['loss_sdf_iso'] = loss_iso_points_sdf.detach()
                loss['loss'] += loss_iso_points_sdf

                # TODO: predict iso_normals from local_frame
                iso_normals_sampled = iso_points_normal.detach()[:, sample_idx, :]
                iso_g_sampled = iso_points_g[:, sample_idx, :]
                loss_normals = torch.mean((1 - F.cosine_similarity(iso_normals_sampled, iso_g_sampled, dim=-1).abs())) * kwargs['lambda_iso_normal'] * iso_points_sdf.nelement() / (iso_points_sdf.nelement()+8000)
                # loss_normals = torch.mean((1 - F.cosine_similarity(iso_points_normal, iso_points_g, dim=-1).abs())) * kwargs['lambda_iso_normal']
                loss['loss_normal_iso'] = loss_normals.detach()
                loss['loss'] += loss_normals


            idx = torch.randperm(gt_surface_pts.shape[1]).to(device=gt_surface_pts.device)[:(gt_surface_pts.shape[1]//2)]
            tmp = torch.index_select(gt_surface_pts, 1, idx)
            space_pts = torch.cat(
                [torch.rand_like(tmp) * 2 - 1,
                 torch.randn_like(tmp, device=tmp.device, dtype=tmp.dtype) * 0.1+tmp], dim=1)

            space_pts.requires_grad_(True)
            pred_space_sdf = decoder(space_pts).sdf
            pred_space_grad = torch.autograd.grad(pred_space_sdf, [space_pts], [
                torch.ones_like(pred_space_sdf)], create_graph=True)[0]

            # 1. eikonal term
            loss_eikonal = (eikonal_loss(pred_surface_grad) +
                            eikonal_loss(pred_space_grad)) * kwargs['lambda_eikonal']
            loss['loss_eikonal'] = loss_eikonal.detach()
            loss['loss'] += loss_eikonal

            # 2. SDF loss
            # loss on iso points
            pred_surface_sdf = decoder(gt_surface_pts).sdf

            loss_sdf = torch.mean(weights * pred_surface_sdf.abs()) * lambda_surface_sdf
            if kwargs['warm_up'] >= 0 and it >= kwargs['warm_up'] and kwargs['lambda_iso_sdf'] != 0:
                # loss_sdf = 0.5 * loss_sdf
                loss_sdf = loss_sdf * pred_surface_sdf.nelement() / (pred_surface_sdf.nelement() + iso_points_sdf.nelement())

            if kwargs['use_sal_loss'] and iso_points is not None:
                dists, idxs, _ = knn_points(space_pts.view(1,-1,3), iso_points.view(1,-1,3).detach(), K=1)
                dists = dists.view_as(pred_space_sdf)
                idxs = idxs.view_as(pred_space_sdf)
                loss_inter = ((eps_sqrt(dists).sqrt() - pred_space_sdf.abs())**2).mean() * kwargs['lambda_inter_sal']
            else:
                alpha = (it / total_iter  + 1)*100
                loss_inter = torch.exp(-alpha * pred_space_sdf.abs()).mean() * kwargs['lambda_inter_sdf']

            loss_sald = torch.tensor(0.0).cuda()
            # prevent wrong closing for open mesh
            if kwargs['use_off_normal_loss'] and it < 1000:
                dists, idxs, _ = knn_points(space_pts.view(1,-1,3), gt_surface_pts_sub.view(1,-1,3).cuda(), K=1)
                knn_normal = knn_gather(gt_surface_normals_sub.cuda().view(1,-1,3), idxs).view(1,-1,3)
                direction_correctness = -F.cosine_similarity(knn_normal, pred_space_grad, dim=-1)
                direction_correctness[direction_correctness < 0] = 0
                loss_sald = torch.mean(direction_correctness*torch.exp(-2*dists)) * 2

            # 3. normal direction
            loss_normals = torch.mean(
                weights * (1 - F.cosine_similarity(gt_surface_normals, pred_surface_grad, dim=-1))) * lambda_surface_normal
            if kwargs['warm_up'] >= 0 and it >= kwargs['warm_up'] and kwargs['lambda_iso_normal'] != 0:
                # loss_normals = 0.5 * loss_normals
                loss_normals = loss_normals * gt_surface_normals.nelement() / (gt_surface_normals.nelement() + iso_normals_sampled.nelement())

            loss['loss_sdf'] = loss_sdf.detach()
            loss['loss_inter'] = loss_inter.detach()
            loss['loss_normals'] = loss_normals.detach()
            loss['loss_sald'] = loss_sald
            loss['loss'] += loss_sdf
            loss['loss'] += loss_inter
            loss['loss'] += loss_sald
            loss['loss'] += loss_normals

            loss['loss'].backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.)

            optimizer.step()
            scheduler.step()
            if it % 20 == 0:
                print("iter {:05d} {}".format(it, ', '.join(
                    ['{}: {}'.format(k, v.item()) for k, v in loss.items()])))

            it += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train from DTU point clouds'
    )
    parser.add_argument('pcl_file', type=str, nargs=1,
                        default='/home/mnt/points/data/DTU_MVS/Points_in_Mask/furu/scan106.ply',
                        help='Path to point cloud file.')
    parser.add_argument('--out_dir', '-o', type=str,
                        nargs='?', help='Directory for output.')
    parser.add_argument('--use_sal_loss', action='store_true',
                        help='Use Distance value.')
    parser.add_argument('--use_off_normal_loss', action='store_true',
                        help='penalize flipped normal outside the shape.')
    parser.add_argument('--resume', action='store_true',
                        help='resume training.')
    parser.add_argument('--decoder', '-d', type=str, nargs=1, default=['siren'], choices=['siren', 'sdf'],
                        help='Decoder type.')
    parser.add_argument('--warm_up', type=int, action=FooAction, default=500)
    parser.add_argument('--weight_mode', type=int, action=FooAction, default=1, dest='w', choices=[1, 2, 3, -1])
    parser.add_argument('--resample_every', type=int, action=FooAction, dest='i',default=2000)
    parser.add_argument('--ear', action='store_true')
    parser.add_argument('--denoise_normal', action='store_true')
    parser.add_argument('--outlier_tolerance', type=float, action=FooAction, default=0.1)
    parser.add_argument('--lambda_eikonal', type=float, action=FooAction, dest='eik', default=5e1)
    parser.add_argument('--lambda_iso_sdf', type=float, action=FooAction, dest='isoSDF', default=1e3)
    parser.add_argument('--lambda_iso_normal', type=float, action=FooAction, dest='isoN', default=1e2)
    parser.add_argument('--lambda_surface_sdf', type=float, action=FooAction, dest='onSDF', default=1e3)
    parser.add_argument('--lambda_inter_sdf', type=float, action=FooAction, dest='offSDF', default=1e2)
    parser.add_argument('--lambda_inter_sal', type=float, action=FooAction, dest='offSal', default=10)
    parser.add_argument('--lambda_surface_normal', type=float, action=FooAction, dest='onN', default=1e2)

    args = parser.parse_args()

    pcl_file = args.pcl_file[0]
    decoder = args.decoder[0]
    use_sal_loss = args.use_sal_loss
    use_off_normal_loss = args.use_off_normal_loss
    out_dir = args.out_dir
    resume = args.resume
    warm_up = args.warm_up
    ear = args.ear

    if out_dir is None:
        out_dir = os.path.join('exp', 'points_3d_outputs',
                               os.path.basename(pcl_file).split('.')[0])
        check_opts = ["eik","isoSDF","isoN","onSDF","offSDF","offSal","onN","i","w"]
        for opt_name in check_opts:
            if hasattr(args, opt_name+'_nondefault'):
                out_dir += '_'+opt_name+str(getattr(args, opt_name))

    run(pcl_file, out_dir, decoder_type=decoder,
        use_sal_loss=use_sal_loss, use_off_normal_loss=use_off_normal_loss,
        resume=resume, warm_up=warm_up, weight_mode=args.w, resample_every=args.i,
        ear=ear,
        lambda_eikonal=args.eik, lambda_iso_sdf=args.isoSDF, lambda_iso_normal=args.isoN,
        lambda_surface_sdf=args.onSDF, lambda_surface_normal=args.onN,
        lambda_inter_sdf=args.offSDF, lambda_inter_sal=args.offSal,
        outlier_tolerance=args.outlier_tolerance, denoise_normal=args.denoise_normal)
