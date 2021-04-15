from collections import OrderedDict, defaultdict
import datetime
import os
import numpy as np
import time
import trimesh
import imageio
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
import plotly.graph_objs as go
from pytorch3d.ops import (knn_points, knn_gather)
from ..core.cloud import PointClouds3D
from ..utils.mathHelper import (
    decompose_to_R_and_t, estimate_pointcloud_local_coord_frames, eps_denom)
from ..training.losses import (IouLoss, NormalLengthLoss, L2Loss,
                               L1Loss, SDF2DLoss)
from .scheduler import TrainerScheduler
from ..models import CombinedModel
from .. import get_debugging_mode, set_debugging_mode_, get_debugging_tensor, logger_py
from ..misc import Thread
from ..misc.visualize import plot_2D_quiver, plot_3D_quiver
from ..utils import slice_dict, scaler_to_color, check_weights, arange_pixels, sample_patch_points
from ..utils.io import save_ply
from ..utils.point_processing import farthest_sampling
from ..models.levelset_sampling import sample_uniform_iso_points
from pytorch3d.loss import chamfer_distance


class BaseTrainer(object):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 *args, **kwargs):
        self.model = model
        self.optimizer = optimizer

    def forward(self, *args, mode="train", **kwargs):
        """
        One forward pass, returns all necessary outputs to getting the losses or evaluation
        """
        raise NotImplementedError

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        """ Performs a evaluation step """
        raise NotImplementedError

    def compute_loss(self, *args, **kwargs):
        """ Returns the training loss (a scalar)  """
        raise NotImplementedError

    def evaluate(self, val_dataloader, reduce=True, **kwargs):
        """Make models eval mode during test time"""
        eval_list = defaultdict(list)

        for data in tqdm(val_dataloader):
            eval_step_dict = self.eval_step(data, **kwargs)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: torch.stack(v) for k, v in eval_list.items()}
        if reduce:
            eval_dict = {k: torch.mean(v) for k, v in eval_dict.items()}
        return eval_dict

    def update_learning_rate(self):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, generator, train_loader, val_loader, device='cpu',
                 cameras=None, log_dir=None, vis_dir=None, debug_dir=None, val_dir=None,
                 threshold=0.0, n_training_points=2048, n_eval_points=4000,
                 lambda_occupied=1., lambda_freespace=1., lambda_rgb=1.,
                 lambda_eikonal=0.01,
                 patch_size=1, clip_grad=True,
                 reduction_method='sum', sample_continuous=False,
                 overwrite_visualization=True,
                 n_debug_points=-1, saliency_sampling_3d=False,
                 resample_every=-1, refresh_metric_every=-1,
                 gamma_n_points_dss=2.0, gamma_n_rays=0.6, gamma_lambda_rgb=1.0,
                 steps_n_points_dss=-1, steps_n_rays=-1, steps_lambda_rgb=-1,
                 limit_n_points_dss=24000, limit_n_rays=1024,
                 steps_proj_tolerance=-1, gamma_proj_tolerance=0.5, limit_proj_tolerance=5e-5,
                 steps_lambda_sdf=-1, gamma_lambda_sdf=1.0,
                 warm_up_iters=0,
                 sdf_alpha=5.0, limit_sdf_alpha=100, gamma_sdf_alpha=2, steps_sdf_alpha=-1,
                 limit_lambda_freespace=1.0, limit_lambda_occupied=1.0, limit_lambda_rgb=1.0,
                 **kwargs):
        """Initialize the BaseModel class.
        Args:
            model (nn.Module)
            optimizer: optimizer
            scheduler: scheduler
            device: device
        """
        self.cfg = kwargs
        self.device = device
        self.model = model
        self.cameras = cameras

        self.val_loader = val_loader
        self.train_loader = train_loader

        self.tb_logger = SummaryWriter(
            log_dir + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S"))

        # implicit function model
        self.vis_dir = vis_dir
        self.val_dir = val_dir
        self.threshold = threshold

        self.lambda_eikonal = lambda_eikonal
        self.lambda_occupied = lambda_occupied
        self.lambda_freespace = lambda_freespace
        self.lambda_rgb = lambda_rgb
        self.sdf_alpha = sdf_alpha

        self.generator = generator
        self.n_eval_points = n_eval_points
        self.patch_size = patch_size
        self.reduction_method = reduction_method
        self.sample_continuous = sample_continuous
        self.overwrite_visualization = overwrite_visualization
        self.saliency_sampling_3d = saliency_sampling_3d
        self.resample_every = resample_every
        self.refresh_metric_every = refresh_metric_every
        self.warm_up_iters = warm_up_iters

        #  tuple (score, mesh)
        self._mesh_cache = None
        self._pcl_cache = {}
        self.ref_pcl = None

        n_points_per_cloud = 0
        init_proj_tolerance = 0
        if isinstance(self.model, CombinedModel):
            n_points_per_cloud = self.model.n_points_per_cloud
            init_proj_tolerance = self.model.projection.proj_tolerance
        self.training_scheduler = TrainerScheduler(init_n_points_dss=n_points_per_cloud,
                                                   init_n_rays=n_training_points,
                                                   init_proj_tolerance=init_proj_tolerance,
                                                   init_lambda_rgb=lambda_rgb,
                                                   init_lambda_freespace=lambda_freespace,
                                                   init_lambda_occupied=lambda_occupied,
                                                   init_sdf_alpha=self.sdf_alpha,
                                                   steps_n_points_dss=steps_n_points_dss,
                                                   steps_n_rays=steps_n_rays,
                                                   steps_proj_tolerance=steps_proj_tolerance,
                                                   steps_sdf_alpha=steps_sdf_alpha,
                                                   steps_lambda_rgb=steps_lambda_rgb,
                                                   steps_lambda_sdf=steps_lambda_sdf,
                                                   warm_up_iters=self.warm_up_iters,
                                                   gamma_n_points_dss=gamma_n_points_dss,
                                                   gamma_n_rays=gamma_n_rays,
                                                   gamma_proj_tolerance=gamma_proj_tolerance,
                                                   gamma_lambda_rgb=gamma_lambda_rgb,
                                                   gamma_sdf_alpha=gamma_sdf_alpha,
                                                   gamma_lambda_sdf=gamma_lambda_sdf,
                                                   limit_n_points_dss=limit_n_points_dss,
                                                   limit_n_rays=limit_n_rays,
                                                   limit_proj_tolerance=limit_proj_tolerance,
                                                   limit_sdf_alpha=limit_sdf_alpha,
                                                   limit_lambda_rgb=limit_lambda_rgb,
                                                   limit_lambda_occupied=limit_lambda_occupied,
                                                   limit_lambda_freespace=limit_lambda_freespace
                                                   )

        self.debug_dir = debug_dir
        self.hooks = []

        self.n_training_points = n_training_points
        self.n_debug_points = n_debug_points

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad = clip_grad

        self.iou_loss = IouLoss(
            reduction=self.reduction_method, channel_dim=None)
        self.eikonal_loss = NormalLengthLoss(
            reduction=self.reduction_method)
        self.l1_loss = L1Loss(reduction=self.reduction_method)
        self.l2_loss = L2Loss(reduction=self.reduction_method)
        self.sdf_loss = SDF2DLoss(reduction=self.reduction_method)

    def _query_mesh(self):
        """
        Generate mesh at the current training (it), evaluate
        """
        if self._mesh_cache is None:
            try:
                mesh = self.generator.generate_mesh({}, with_colors=False,
                                                    with_normals=False)
            except Exception as e:
                return logger_py.error("Couldn\'t generate mesh {}".format(e))
            else:
                if self.cfg['model_selection_mode'] == 'maximize':
                    model_selection_sign = 1
                elif self.cfg['model_selection_mode'] == 'minimize':
                    model_selection_sign = -1

                self._mesh_cache = (-model_selection_sign * float('inf'), mesh)
                self._pcl_cache.clear()

        if self._mesh_cache is not None:
            return self._mesh_cache[1]

    def _query_pcl(self, n_points=-1):
        """
        get a uniform point cloud on the iso-surface, save in a cache
        """
        if n_points < 0 or n_points is None and len(self._pcl_cache) > 0:
            n_points = list(self._pcl_cache.keys())[0]
        iso_pcl = self._pcl_cache.get(n_points, None)
        new_pcl = False
        if iso_pcl is None:
            t0 = time.time()
            iso_pcl = sample_uniform_iso_points(self.model.decoder, n_points, bounding_sphere_radius=self.model.object_bounding_sphere,
                                                init_points=self.model._points.points_padded())
            t1 = time.time()

            normals = self.model.get_normals_from_grad(iso_pcl.points_padded(), requires_grad=False)
            logger_py.debug('[Sample from Mesh] time ellapsed {}s'.format(t1 - t0))
            iso_pcl = PointClouds3D(iso_pcl.points_padded(), normals=normals)
            self._pcl_cache.clear()
            self._pcl_cache[n_points] = iso_pcl
            new_pcl = True
        return new_pcl, iso_pcl

    def evaluate_mesh(self, val_dataloader, it, **kwargs):
        logger_py.info("[Mesh Evaluation]")
        t0 = time.time()
        if not os.path.exists(self.val_dir):
            os.makedirs(self.val_dir)

        eval_list = defaultdict(list)

        mesh_gt = val_dataloader.dataset.get_meshes()
        assert(mesh_gt is not None)
        mesh_gt = mesh_gt.to(device=self.device)

        pointcloud_tgt = val_dataloader.dataset.get_pointclouds(
            num_points=self.n_eval_points)

        mesh = self.generator.generate_mesh({}, with_colors=False, with_normals=False)
        points_pred = trimesh.sample.sample_surface_even(mesh, pointcloud_tgt.points_packed().shape[0])
        chamfer_dist = chamfer_distance(pointcloud_tgt.points_padded(), torch.from_numpy(points_pred).view(1,-1,3).to(device=pointcloud_tgt.points_padded().device, dtype=torch.float32)
                        )
        eval_dict_mesh = {'chamfer': chamfer_dist.item()}

        # save to "val" dict
        t1 = time.time()
        logger_py.info('[Mesh Evaluation] time ellapsed {}s'.format(t1 - t0))
        if not mesh.is_empty:
            mesh.export(os.path.join(self.val_dir, "%010d.ply" % it))
        return eval_dict_mesh

    def eval_step(self, data, **kwargs):
        """
        evaluate with image mask iou or image rgb psnr
        """
        lights_model = kwargs.get('lights', self.val_loader.dataset.get_lights())
        cameras_model = kwargs.get('cameras', self.val_loader.dataset.get_cameras())
        img_size = self.generator.img_size
        eval_dict = {'iou': 0.0, 'psnr': 0.0}
        with autograd.no_grad():
            self.model.eval()
            data = self.process_data_dict(data, cameras_model, lights=lights_model)
            img_mask = data['mask_img']
            img = data['img']
            # render image
            rgbas = self.generator.raytrace_images(
                img_size, img_mask, cameras=data['camera'], lights=data['light'])
            assert(len(rgbas) == 1)
            rgba = rgbas[0]
            rgba = torch.tensor(
                rgba[None, ...], dtype=torch.float, device=img_mask.device).permute(0, 3, 1, 2)

            # compare iou
            mask_gt = F.interpolate(
                img_mask.float(), img_size, mode='bilinear', align_corners=False).squeeze(1)
            mask_pred = rgba[:, 3, :, :]
            eval_dict['iou'] += self.iou_loss(mask_gt.float(),
                                              mask_pred.float(), reduction='mean')

            # compare psnr
            rgb_gt = F.interpolate(
                img, img_size, mode='bilinear', align_corners=False)
            rgb_pred = rgba[:, :3, :, :]
            eval_dict['psnr'] += self.l2_loss(
                rgb_gt, rgb_pred, channel_dim=1, reduction='mean', align_corners=False).detach()

        return eval_dict

    def train_step(self, data, cameras, **kwargs):
        """
        Args:
            data (dict): contains img, img.mask and img.depth and camera_mat
            cameras (Cameras): Cameras object from pytorch3d
        Returns:
            loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        it = kwargs.get("it", None)
        lights = kwargs.get('lights', None)
        if hasattr(self, 'training_scheduler'):
            self.training_scheduler.step(self, it)

        if isinstance(self.model, CombinedModel) and it > self.warm_up_iters:
            if self.resample_every > 0 and (it - self.warm_up_iters) % self.resample_every == 0:
                self._pcl_cache.clear()
            is_new = self.sample_from_mesh(self.model.n_points_per_cloud)
            if self.saliency_sampling_3d:
                refresh_per_point_metric = is_new or (self.refresh_metric_every > 0) and \
                    ((it - 1) % self.refresh_metric_every == 0)
                self.model.eval()
                if refresh_per_point_metric:
                    self.ref_pcl = self.ref_per_point_metric(mode=self.cfg['ref_metric'])
                    colors = scaler_to_color(
                        self.ref_pcl.features_packed().cpu().numpy().reshape(-1))
                    save_ply(os.path.join(self.vis_dir, '%010d_refpcl.ply' % it), self.ref_pcl.points_packed().cpu().numpy(),
                             colors=colors, normals=self.ref_pcl.normals_packed().cpu().numpy())

        data = self.process_data_dict(data, cameras, lights=lights)
        self.model.train()
        # autograd.set_detect_anomaly(True)
        loss = self.compute_loss(data['img'], data['mask_img'], data['input'],
                                 data['camera'], data['light'], it=it, ref_pcl=self.ref_pcl)
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.)
        self.optimizer.step()
        check_weights(self.model.state_dict())

        return loss.item()

    def process_data_dict(self, data, cameras, lights=None):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        device = self.device

        # Get "ordinary" data
        img = data.get('img.rgb').to(device)
        assert(img.min() >= 0 and img.max() <=
               1), "Image must be a floating number between 0 and 1."
        mask_img = data.get('img.mask').to(device)

        camera_mat = data.get('camera_mat', None)

        # inputs for SVR
        inputs = data.get('inputs', torch.empty(0, 0)).to(device)

        # set camera matrix to cameras
        if camera_mat is None:
            logger_py.warning(
                "Camera matrix is not provided! Using the default matrix")
        else:
            cameras.R, cameras.T = decompose_to_R_and_t(camera_mat)
            cameras._N = cameras.R.shape[0]
            cameras.to(device)

        if lights is not None:
            lights_params = data.get('lights', None)
            if lights_params is not None:
                lights = type(lights)(**lights_params).to(device)

        return {'img': img, 'mask_img': mask_img, 'input': inputs, 'camera': cameras, 'light': lights}

    def sample_pixels(self, n_rays: int, batch_size:int, h: int, w: int):

        if n_rays >= h * w:
            p = arange_pixels((h, w), batch_size)[1].to(self.device)
        else:
            p = sample_patch_points(batch_size, n_rays,
                                    patch_size=self.patch_size,
                                    image_resolution=(h, w),
                                    continuous=self.sample_continuous,
                                    ).to(self.device)
        return p

    def sample_from_mesh(self, n_points):
        """
        Construct mesh from implicit model and sample from the mesh to get iso-surface points,
        which is used for projection in the combined model
        Return True is a new pcl is sampled
        """
        try:
            new_pcl, pcl = self._query_pcl(n_points)
            self.model._points = pcl
            self.model.points = pcl.points_padded()
            if not os.path.exists(self.vis_dir):
                os.makedirs(self.vis_dir)
        except Exception as e:
            logger_py.error("Couldn't sample points from mesh: {}".format(e))
            return False

        return new_pcl

    def compute_loss(self, img, mask_img, inputs, cameras, lights, n_points=None, eval_mode=False, it=None, ref_pcl=None):
        ''' Compute the loss.
        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        # Initialize loss dictionary and other values
        loss = {}

        # overwrite n_points
        if n_points is None:
            n_points = self.n_eval_points if eval_mode else self.n_training_points

        # Shortcuts
        device = self.device
        patch_size = self.patch_size
        reduction_method = self.reduction_method
        batch_size, _, h, w = img.shape

        # Assertions
        assert(((h, w) == mask_img.shape[2:4]) and
               (patch_size > 0))

        # Apply losses
        # 1.) Initialize loss
        loss['loss'] = 0
        if isinstance(self.model, CombinedModel):
            # 1.) Sample points on image plane ("pixels")
            p = None
            if n_points > 0:
                p = self.sample_pixels(n_points, batch_size, h, w)

            project = (it - self.warm_up_iters > 0) and not eval_mode
            sample_iso_offsurface = project
            # TODO: check insertion fix: start using insertion after 5000 iterations
            saliency_sampling_3d = self.saliency_sampling_3d
            model_outputs = self.model(mask_img, img, cameras, pixels=p, inputs=inputs, lights=lights,
                                       it=it, eval_mode=eval_mode, project=project, sample_iso_offsurface=sample_iso_offsurface,
                                       proj_kwargs={'ref_pcl': ref_pcl if saliency_sampling_3d else None, 'insert': saliency_sampling_3d})
        else:
            # 1.) Sample points on image plane ("pixels")
            p = self.sample_pixels(n_points, batch_size, h, w)

            model_outputs = self.model(
                p, img, mask_img, inputs=inputs, cameras=cameras, lights=lights, it=it)

        point_clouds = model_outputs.get('iso_pcl')
        rgb_gt = model_outputs.get('iso_rgb_gt')
        sdf_freespace = model_outputs.get('sdf_freespace')
        sdf_occupancy = model_outputs.get('sdf_occupancy')


        if it % 50 == 0 and not eval_mode and not get_debugging_mode():
            logger_py.debug('# iso: {}, # occ_off: {}, # free_off: {}'.format(
                model_outputs['iso_rgb_gt'].shape[0], model_outputs['p_occupancy'].shape[0], model_outputs['p_freespace'].shape[0]))

        if not point_clouds.isempty():
            # Photo Consistency Loss
            normalizing_value = 1.0
            if reduction_method == 'sum':
                total_p = batch_size*self.training_scheduler.init_n_rays
                normalizing_value = 1.0 / total_p * min((sdf_freespace.nelement()+sdf_occupancy.nelement())/float(rgb_gt.size(0)), 1.0)
            self.calc_photoconsistency_loss(point_clouds, rgb_gt, reduction_method, loss, normalizing_value=normalizing_value)

        # Occupancy and Freespace losses
        if self.lambda_occupied > 0 or self.lambda_freespace > 0:
            normalizing_value = 1.0
            if reduction_method == 'sum':
                total_p = batch_size*self.training_scheduler.init_n_rays
                normalizing_value = 1.0 / total_p
            self.calc_sdf_mask_loss(
                sdf_freespace, sdf_occupancy, self.sdf_alpha, reduction_method, loss, normalizing_value=normalizing_value)

        # Eikonal loss
        # Random samples in the space
        total_p = batch_size*self.training_scheduler.init_n_rays
        space_pts = torch.empty(total_p, 3).uniform_(-self.model.object_bounding_sphere,
                                                          self.model.object_bounding_sphere).to(device=device)
        # space_pts = torch.cat([model_outputs.get('p_freespace'), model_outputs.get('p_occupancy'), point_clouds.points_packed()]).detach()
        eikonal_normals = self.model.get_normals_from_grad(
            space_pts, c=inputs, requires_grad=True)
        normalizing_value = 1.0
        if reduction_method == 'sum':
            normalizing_value = 1.0 / total_p
        self.calc_eikonal_loss(eikonal_normals, reduction_method, loss, normalizing_value=normalizing_value)

        for k, v in loss.items():
            mode = 'val' if eval_mode else 'train'
            if isinstance(v, torch.Tensor):
                self.tb_logger.add_scalar('%s/%s' % (mode, k), v.item(), it)
            else:
                self.tb_logger.add_scalar('%s/%s' % (mode, k), v, it)

        return loss if eval_mode else loss['loss']

    def ref_per_point_metric(self, ref_pcl: PointClouds3D = None, mode='curvature'):
        """
        Computes the metric used for sampling or weighting, e.g. curvature / loss,
        and update to pcl's features
        When mode == 'curvature', estimates the shape curvature from the point cloud using
        a local neighborhood of 12 points,
        When mode == 'loss', average the per point per view RGB loss over the entire training images
        """
        if ref_pcl is None:
            ref_pcl = self.model._points

        if (n_points:=ref_pcl.points_packed().shape[0]) > 5000:
            ref_pcl = farthest_sampling(ref_pcl, 5000/n_points)

        with autograd.no_grad():
            if mode == 'loss':
                self.model.eval()

                cameras = self.val_loader.dataset.get_cameras()
                lights = self.val_loader.dataset.get_lights()

                # project all init_points to the surface
                proj_result = self.model.projection.project_points(ref_pcl,
                    self.model.decoder, skip_resampling=False, skip_upsampling=False, sample_iters=2)
                ref_pcl = PointClouds3D(proj_result['levelset_points'][proj_result['mask']].view(1,-1,3),
                                                   normals=proj_result['levelset_normals'][proj_result['mask']].view(1,-1,3))
                num_points2 = ref_pcl.num_points_per_cloud()
                logger_py.info('[Per Point Loss Metric] evaluating ref point cloud ({}) on all training images'.format(
                    num_points2.item()))
                assert(len(num_points2) == 1)
                from ..utils.mathHelper import RunningStat
                runStat = RunningStat(num_points2.item(),
                                      device=ref_pcl.device)

                # set max_iso_per_batch to -1
                max_iso_per_batch = self.model.max_iso_per_batch
                self.model.max_iso_per_batch = -1
                # If loss is used, transfer the computed loss in the current point cloud to the reference point cloud
                for batch in tqdm(self.val_loader):
                    data = self.process_data_dict(
                        batch, cameras=cameras, lights=lights)
                    mask_img, img, cameras, lights = data['mask_img'], data['img'], data['camera'], data['light']
                    # set proj_max_iters to 0 because we already projected the points before hand
                    model_outputs = self.model(mask_img, img, cameras, mask_gt=None, pixels=None, inputs=None, lights=lights,
                                               project=True, sample_iso_offsurface=False,
                                               )

                    point_clouds = model_outputs['iso_pcl']
                    pixel_pred = model_outputs['iso_pixel']
                    rgb_gt = model_outputs['iso_rgb_gt']
                    sig = num_points2.item() / self.model.object_bounding_sphere
                    dist_thres = 4 / sig
                    if not point_clouds.isempty():
                        num_points1 = point_clouds.num_points_per_cloud().sum(0, keepdim=True)
                        loss = {'loss': 0.0}
                        self.calc_photoconsistency_loss(point_clouds,
                                                        rgb_gt, 'none', loss,
                                                        )
                        loss_per_point = loss['loss_rgb'] / self.lambda_rgb

                        query_points = point_clouds.points_packed()
                        dists, idxs, _ = knn_points(
                            ref_pcl.points_padded().contiguous(), query_points.unsqueeze(0).contiguous(), num_points2, num_points1, K=1,
                            return_nn=False)
                        loss_ref_point = knn_gather(loss_per_point.view(
                            1, num_points1.item(), 1), idxs, num_points1).view(-1, 1)
                        mask = (dists < dist_thres) & (dists > 0)
                        runStat.add(loss_ref_point.view(-1), mask.view(-1))

                per_point_metric = runStat.mean().view(-1, 1)
                logger_py.debug('[Per Point Loss Metric] ref point cloud metric min {} max {} median {}'.format(
                    per_point_metric.min(), per_point_metric.max(), per_point_metric.median()))

                self.model.max_iso_per_batch = max_iso_per_batch
            # or curvature is used
            elif mode == 'curvature':
                curvatures, _ = estimate_pointcloud_local_coord_frames(ref_pcl, neighborhood_size=12,
                                                                       disambiguate_directions=False, return_knn_result=False)
                # high curvature area : variance in the local frame is large
                curvatures = curvatures[..., 0] / \
                    eps_denom(curvatures[..., -1])
                per_point_metric = curvatures.view(-1, 1)

        ref_pcl = ref_pcl.update_features_(per_point_metric)

        return ref_pcl

    def calc_eikonal_loss(self, normals, reduction_method, loss={}, normalizing_value=1.0):
        """ Implicit function gradient norm == 1 """
        if self.lambda_eikonal > 0:
            eikonal_loss = self.eikonal_loss(
                normals, reduce=reduction_method) * self.lambda_eikonal * normalizing_value
            loss['loss_eikonal'] = eikonal_loss
            loss['loss'] = eikonal_loss + loss['loss']

    def calc_sdf_mask_loss(self, sdf_freespace, sdf_occupancy, alpha, reduction_method, loss={}, normalizing_value=1.0):
        """
        [1] Multiview Neural Surface Reconstruction with Implicit Lighting and Material (eq. 7)
        Penalize occupancy, different to [1], penalize only if sdf sign is wrong,
        TODO: Use point clouds to find the closest point for sdf?
        Args:
            sdf_freespace (tensor): (N1,)
            sdf_occupancy (tensor): (N2,)
            alpha (float):
        """
        sdf_freespace = sdf_freespace.squeeze(-1)
        sdf_occupancy = sdf_occupancy.squeeze(-1)
        n_free = float(sdf_freespace.nelement())
        n_occp = float(sdf_occupancy.nelement())
        if n_free + n_occp == 0:
            return

        loss_freespace = self.sdf_loss(-alpha * sdf_freespace, True,
                                        reduction=reduction_method)
        loss_freespace = loss_freespace * self.lambda_freespace * normalizing_value

        # balance sample counts, if occupancy samples << freespace, increase occupancy sample weight
        if reduction_method == 'mean':
            normalizing_value /= (n_free/5/max(n_occp, 1))
        elif reduction_method == 'sum':
            normalizing_value *= (n_free/5/max(n_occp, 1))
        loss_occupancy = self.sdf_loss(-alpha * sdf_occupancy, False,
                                        reduction=reduction_method)
        loss_occupancy = loss_occupancy * self.lambda_occupied * normalizing_value

        loss['loss'] = loss_freespace + loss['loss']
        loss['loss_freespace'] = loss_freespace
        loss['loss'] = loss_occupancy + loss['loss']
        loss['loss_occupancy'] = loss_occupancy

    def calc_photoconsistency_loss(self, colored_pcl, rgb_gt, reduction_method, loss, normalizing_value=1.0):
        ''' Calculates the photo-consistency loss.

        Args:
            colored_pcl: (PointClouds)
            rgb_gt (tensor): ground truth color (N,3)
            reduction_method (string): how to reduce the loss tensor
            loss (dict): loss dictionary
        '''
        if self.lambda_rgb != 0:
            rgb_pred = colored_pcl.features_packed()
            assert(rgb_pred.ndim == 2)
            _, dim = rgb_pred.shape
            assert(dim == 3)

            loss_rgb = self.l1_loss(
                rgb_pred, rgb_gt, reduction=reduction_method) * self.lambda_rgb * normalizing_value

            loss['loss'] = loss_rgb + loss['loss']
            loss['loss_rgb'] = loss_rgb

    def visualize(self, data, cameras, lights=None, it=0, vis_type='mesh', **kwargs):
        ''' Visualized the data.

        Args:
            data (dict): data dictionary
            it (int): training iteration
            vis_type (string): visualization type
        '''
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        # naming
        if self.overwrite_visualization:
            prefix = ''
        else:
            prefix = '%010d_' % it

        # use only one mini-batch
        data = slice_dict(data, [0, ])

        with torch.autograd.no_grad():
            device = self.device
            data = self.process_data_dict(data, cameras, lights)
            cameras = data['camera']
            lights = data['light']
            # visualize the rendered image and pointcloud
            try:
                if vis_type == 'image':
                    img_list = self.generator.generate_images(
                        data, cameras=cameras, lights=lights, **kwargs)
                    for i, img in enumerate(img_list):
                        out_file = os.path.join(
                            self.vis_dir, '%s%03d' % (prefix, i))
                        if isinstance(img, go.Figure):
                            img.write_html(out_file + '.html')
                        else:
                            # self.tb_logger.add_image('train/vis/render', img[...,:3], global_step=it, dataformats='HWC')
                            img = img * 255
                            imageio.imwrite(out_file + '.png',
                                            img.astype(np.uint8))

                    # visualize ground truth image and mask
                    img_gt = data.get('img').permute(0, 2, 3, 1)
                    mask_gt = data.get('mask_img').float().permute(0, 2, 3, 1)
                    rgba_gt = torch.cat([img_gt, mask_gt], dim=-1)
                    for i in range(rgba_gt.shape[0]):
                        # self.tb_logger.add_image('train/vis/gt', img[...,:3], global_step=it, dataformats='HWC')
                        img = rgba_gt[i].cpu().numpy() * 255.0
                        out_file = os.path.join(
                            self.vis_dir, '%s%03d_Gt.png' % (prefix, i))
                        imageio.imwrite(out_file, img.astype(np.uint8))

                elif vis_type == 'pointcloud':
                    pcl_list = self.generator.generate_pointclouds(
                        data, cameras=cameras, lights=lights, **kwargs)
                    for i, pcl in enumerate(pcl_list):
                        if isinstance(pcl, trimesh.Trimesh):
                            pcl_out_file = os.path.join(
                                self.vis_dir, '%s%03d_pts.ply' % (prefix, i))
                            pcl.export(pcl_out_file, vertex_normal=True)
                            # self.tb_logger.add_mesh('train/vis/points', mesh.vertices, global_step=it)

                elif vis_type == 'mesh':
                    mesh = self.generator.generate_mesh(
                        data, with_colors=False, with_normals=False)
                    if isinstance(mesh, trimesh.Trimesh):
                        # self.tb_logger.add_mesh('train/vis/mesh', mesh.vertices, faces=mesh.faces, global_step=it)
                        mesh_out_file = os.path.join(
                            self.vis_dir, '%smesh.ply' % prefix)
                        mesh.export(mesh_out_file, vertex_normal=True)

            except Exception as e:
                logger_py.error(
                    "Exception occurred during visualization: {} ".format(e))

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def update_learning_rate(self, it):
        """Update learning rates for all modifiers"""
        self.scheduler.step()
        for param_group in self.optimizer.param_groups:
            v = param_group['lr']
            self.tb_logger.add_scalar('train/lr', v, it)

    def debug(self, data_dict, cameras, lights=None, it=0, mesh_gt=None, **kwargs):
        """
        output interactive plots for debugging
        # TODO(yifan): reused code from visualize
        """
        self._threads = getattr(self, '_threads', [])
        for t in self._threads:
            t.join()

        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        # use only one mini-batch
        data_dict = slice_dict(data_dict, [0, ])

        data = self.process_data_dict(data_dict, cameras, lights)
        try:
            mesh = self._query_mesh()
        except Exception as e:
            logger_py.error('Could not generate mesh: {}'.format(e))
            mesh = None

        # incoming data is channel fist
        mask_img_gt = data['mask_img'].detach().cpu().squeeze()
        H, W = mask_img_gt.shape

        set_debugging_mode_(True)
        self.model.train()
        self.model.debug(True)
        self.optimizer.zero_grad()
        loss = self.compute_loss(data['img'], data['mask_img'], data['input'],
                                 data['camera'], data['light'], it=it)
        loss.backward()

        # plot
        with torch.autograd.no_grad():
            dbg_tensor = get_debugging_tensor()

            # save figure
            if self.overwrite_visualization:
                ending = ''
            else:
                ending = '%010d_' % it

            # plot ground truth mesh if provided
            if mesh_gt is not None:
                assert(len(mesh_gt) == 1), \
                    "mesh_gt and gt_mask_img must have the same or broadcastable batchsize"
                mesh_gt = mesh_gt[0]
            try:
                # prepare data to create 2D and 3D figure
                n_pts = OrderedDict((k, dbg_tensor.pts_world_grad[k][0].shape[0])
                                    for k in dbg_tensor.pts_world_grad)

                for i, k in enumerate(dbg_tensor.pts_world_grad):
                    if dbg_tensor.pts_world[k][0].shape[0] != n_pts[k]:
                        logger_py.error('Found unequal pts[{0}] ({2}) and pts_grad[{0}] ({1}).'.format(
                            k, n_pts[k], dbg_tensor.pts_world[k][0].shape[0]))

                pts_list = [dbg_tensor.pts_world[k][0] for k in n_pts]
                grad_list = [dbg_tensor.pts_world_grad[k][0]
                             for k in n_pts]

                pts_world = torch.cat(pts_list, dim=0)
                pts_world_grad = torch.cat(grad_list, dim=0)

                try:
                    img_mask_grad = dbg_tensor.img_mask_grad[0].clone()
                except Exception:
                    img_mask_grad = None

                # convert world to ndc
                if len(cameras) > 1:
                    _cams = cameras.clone().to(device=pts_world.device)
                    _cams.R = _cams[0:0 + 1].R
                    _cams.T = _cams[0:0 + 1].T
                    _cams._N = 1
                else:
                    _cams = cameras.clone().to(device=pts_world.device)

                pts_ndc = _cams.transform_points_screen(pts_world.view(
                    1, -1, 3), ((W, H),), eps=1e-17).view(-1, 3)[..., :2]
                pts_grad_ndc = _cams.transform_points_screen(
                    (pts_world + pts_world_grad).view(1, -1, 3), ((W, H),), eps=1e-8).view(-1, 3)[..., :2]

                # create 2D plot
                pts_ndc_dict = {k: t for t, k in zip(torch.split(
                    pts_ndc, list(n_pts.values())), n_pts.keys())}
                grad_ndc_dict = {k: t for t, k in zip(torch.split(
                    pts_grad_ndc, list(n_pts.values())), n_pts.keys())}

                plotter_2d = Thread(target=plot_2D_quiver, name='%sproj.html' % ending,
                                    args=(pts_ndc_dict, grad_ndc_dict,
                                          mask_img_gt.clone()),
                                    kwargs=dict(img_mask_grad=img_mask_grad,
                                                save_html=os.path.join(
                                                    self.debug_dir, '%sproj.html' % ending)),
                                    )
                plotter_2d.start()
                self._threads.append(plotter_2d)

                # create 3D plot
                pts_world_dict = {k: t for t, k in zip(torch.split(
                    pts_world, list(n_pts.values())), n_pts.keys())}
                grad_world_dict = {k: t for t, k in zip(torch.split(
                    pts_world_grad, list(n_pts.values())), n_pts.keys())}
                plotter_3d = Thread(target=plot_3D_quiver, name='%sworld.html' % ending,
                                    args=(pts_world_dict, grad_world_dict),
                                    kwargs=dict(mesh_gt=mesh_gt, mesh=mesh,
                                                camera=_cams, n_debug_points=self.n_debug_points,
                                                save_html=os.path.join(self.debug_dir, '%sworld.html' % ending)),
                                    )
                plotter_3d.start()
                self._threads.append(plotter_3d)

            except Exception as e:
                logger_py.error('Could not plot gradient: {}'.format(repr(e)))

        # set debugging to false and remove hooks
        set_debugging_mode_(False)
        self.model.debug(False)
        self.iou_loss.debug(False)
        logger_py.info('Disabled debugging mode.')

        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def evaluate(self, val_dataloader, reduce=True, **kwargs):
        eval_dict = super().evaluate(val_dataloader, reduce=reduce,
                                     cameras=self.val_loader.dataset.get_cameras(), lights=self.val_loader.dataset.get_lights())
        if reduce:
            if self.cfg['model_selection_mode'] == 'maximize':
                model_selection_sign = 1
            elif self.cfg['model_selection_mode'] == 'minimize':
                model_selection_sign = -1
            if self._mesh_cache is None or model_selection_sign * (eval_dict[self.cfg['model_selection_metric']] - self._mesh_cache[0]) > 0:
                try:
                    mesh = self.generator.generate_mesh({}, with_colors=False,
                                                        with_normals=False)
                except Exception as e:
                    pass
                else:
                    logger_py.info('Updated cached mesh')
                    self._mesh_cache = (
                        eval_dict[self.cfg['model_selection_metric']], mesh)
                    self._pcl_cache.clear()
        return eval_dict

    def save_shape(self, save_path, it, **kwargs):
        # For combined model also save model.points and generate a mesh
        if isinstance(self.model, CombinedModel):
            mesh = self._query_mesh()
            mesh.export(os.path.splitext(save_path)[0] + '_mesh.ply')

            _, pcl = self._query_pcl(self.model.n_points_per_cloud)
            trimesh.Trimesh(pcl.points_packed().cpu().detach().numpy(), vertex_normals=pcl.normals_packed().cpu().detach().numpy(), process=False).export(os.path.splitext(save_path)[
                0] + '_pts.ply', vertex_normal=True)
