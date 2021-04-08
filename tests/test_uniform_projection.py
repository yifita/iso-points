import unittest
import config
import time
import imageio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from im2mesh.checkpoints import CheckpointIO
from DSS.models.levelset_sampling import UniformProjection
from DSS.core.cloud import PointClouds3D, PointCloudsFilters
from pytorch3d.renderer import (FoVPerspectiveCameras)
import os
import torch
import numpy as np
from DSS.utils import scaler_to_color, get_tensor_values
from DSS.models.levelset_sampling import UniformProjection
from DSS.utils.io import save_ply
from DSS.utils.point_processing import wlop, remove_outliers


class TestUniformProj(unittest.TestCase):
    in_dir = os.path.join(
        'tests', 'test_inputs')
    in_dir = os.path.join(
        'exp', 'armadillo_uni')
    # in_dir = os.path.join(
    #     'exp', 'dtuBird_idrSiren_iso_Mix_dirSamp_growingRGB2_warmUp_more_s3D_largest')

    out_dir = os.path.join('tests', 'outputs', os.path.splitext(
        os.path.basename(__file__))[0])
    os.makedirs(out_dir, exist_ok=True)

    # create and load model
    # load config
    # cfg = config.load_config(os.path.join(
    #     in_dir, 'fandisk_config.yaml'), 'configs/default.yaml')
    cfg = config.load_config(os.path.join(
        in_dir, 'config.yaml'), 'configs/default.yaml')
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    # cfg.training.point_file = 'shape_pts.ply'
    cfg.training.point_file = 'none.ply'
    cfg.model.model_kwargs.n_points_per_cloud = 6000

    # create model from config
    dataset = config.create_dataset(cfg.data, mode='test')

    # create model
    model = config.create_model(cfg, mode='train', device=device,
                                camera_model=dataset.get_cameras()).to(device=device)
    print(model)

    checkpoint_io = CheckpointIO(in_dir, model=model)
    # checkpoint_io.load('fandisk_model.pt')
    checkpoint_io.load('model_5000.pt')

    # 1. load dataset and a batch of cameras
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)

    # 2. create initial point cloud from generated mesh
    generator = config.create_generator(cfg, model, device=device)
    model.eval()

    # 3. create trainer
    cfg.training.reduction_method = 'none'
    cfg.training.lambda_eikonal = 0.0
    cfg.training.lambda_rgb = 100.0
    lr = 0.0001
    epoch_it = -1
    trainer = config.create_trainer(
        cfg, model, None, None, generator, data_loader, data_loader, device=device)

    points = model._points.points_padded()
    def test_step_wise(self):
        device = self.device
        out_dir = self.out_dir
        in_dir = self.in_dir
        model = self.model
        decoder = self.model.decoder
        trainer = self.trainer
        data_loader = self.data_loader

        # project
        projector = UniformProjection(
            max_points_per_pass=16000, proj_max_iters=10,
            proj_tolerance=1e-6, knn_k=8)

        from DSS.models.levelset_sampling import sample_uniform_iso_points
        iso_points = sample_uniform_iso_points(decoder, 5000, bounding_sphere_radius=1.0)
        save_ply(os.path.join(out_dir, 'uniform_iso_points.ply'), iso_points.points_packed().cpu())

        # # only projection
        # points = (torch.rand((1, 15000, 3), device=device) - 0.5) * 2.0
        # save_ply(os.path.join(out_dir, 'step_init.ply'), points[0].cpu().detach())
        # proj_results = projector.project_points(points, decoder, skip_resampling=True, skip_upsampling=True)
        # mask = proj_results['mask']
        # proj_points = proj_results['levelset_points'][mask].detach()
        # normals = model.get_normals_from_grad(proj_points, requires_grad=False)
        # save_ply(os.path.join(out_dir, "step_projection.ply"), proj_points.cpu().detach().view(-1, 3).numpy(),
        #          normals=normals.cpu().detach().view(-1, 3).numpy())

        # # uniform projection
        # wlop_result = wlop(PointClouds3D(proj_points.view(1,-1,3)), 0.5)
        # save_ply(os.path.join(out_dir, "step_resample_wlop.ply"),
        #          wlop_result.points_packed().cpu().detach().numpy())
        # proj_results = projector.project_points(wlop_result.points_padded(), decoder, skip_resampling=True, skip_upsampling=False)
        # mask = proj_results['mask']
        # proj_points = proj_results['levelset_points'][mask].detach()
        # normals = model.get_normals_from_grad(proj_points, requires_grad=False)
        # save_ply(os.path.join(out_dir, "step_resample.ply"), proj_points.cpu().detach().view(-1, 3).numpy(),
        #          normals=normals.cpu().detach().view(-1, 3).numpy())

        # # proj_results = projector.project_points(proj_points.view(1, -1, 3), decoder, proj_max_iters=0, sample_iters=3)
        # # mask = proj_results['mask']
        # # proj_points = proj_results['levelset_points'][mask].detach()
        # # normals = model.get_normals_from_grad(proj_points, requires_grad=False)
        # # save_ply(os.path.join(out_dir, "step_resample.ply"),
        # #          proj_points.cpu().detach().view(-1, 3).numpy(),
        # #          normals=normals.cpu().detach().view(-1, 3).numpy())
        # from DSS.utils.point_processing import upsample
        # upsampled, num_points = upsample(proj_points.view(1, -1 ,3), 8000)
        # save_ply(os.path.join(out_dir, 'step_upsample_0.ply'),
        #          upsampled.cpu().detach().view(-1, 3).numpy())
        # proj_results = projector.project_points(upsampled.view(1, -1, 3), decoder, max_proj_iters=5, skip_resampling=True, skip_upsampling=False)
        # mask = proj_results['mask']
        # proj_points = proj_results['levelset_points'][mask].detach()
        # normals = model.get_normals_from_grad(proj_points, requires_grad=False)
        # save_ply(os.path.join(out_dir, 'step_upsample.ply'),
        #          proj_points.cpu().detach().view(-1, 3).numpy(),
        #          normals=normals.cpu().detach().view(-1, 3).numpy())

    def test_uniform_proj_v2(self):
        """ Projection with pre filtering """
        device = self.device
        out_dir = self.out_dir
        in_dir = self.in_dir
        model = self.model
        decoder = self.model.decoder
        points = self.points
        trainer = self.trainer
        data_loader = self.data_loader

        # project
        projector = UniformProjection(
            max_points_per_pass=16000, proj_max_iters=10,
            proj_tolerance=1e-6, knn_k=8, total_iters=3, sample_iters=1)

        proj_points = points.clone()
        proj_normals = None
        cameras = data_loader.dataset.get_cameras()
        lights = data_loader.dataset.get_lights()
        t_acc = 0
        normals = model.get_normals_from_grad(
            model._points.points_padded(), requires_grad=False)
        colors = torch.from_numpy(scaler_to_color(model._points.points_padded(
        )[..., 0].view(-1).cpu().detach().numpy())).to(device=device, dtype=torch.float)
        model._points = PointClouds3D(
            model._points.points_padded(), normals, colors.view_as(normals))
        for it, batch in enumerate(data_loader):
            data = trainer.process_data_dict(
                batch, cameras=cameras, lights=lights)
            mask_img = data['mask_img']
            # first filter renderable points
            t0 = time.time()
            pcl_filter = PointCloudsFilters(device=device)
            fragments = model.renderer.rasterizer(
                model._points, pcl_filter, cameras=data['camera'])
            pcl_visible = pcl_filter.filter_with(
                model._points.clone(), ('visibility',))

            proj_results = projector.project_points(pcl_visible.offset(
                (0.1 * (torch.rand_like(pcl_visible.points_packed()) - 0.5))), decoder)
            t_acc += (time.time() - t0)

            rgba = model.renderer(model._points, cameras=data['camera'])
            if rgba is not None:
                save_ply(os.path.join(out_dir, 'raster_projected_%02d_1.ply' % it),
                         pcl_visible.points_packed().cpu().numpy(), normals=pcl_visible.normals_packed().cpu().numpy()
                         )
                imageio.imwrite(os.path.join(out_dir, 'raster_projected_%02d.png' %
                                             it), (rgba[0].cpu().detach().numpy() * 255).astype('uint8'))

            mask = proj_results['mask']
            proj_points = proj_results['levelset_points'][mask].detach()
            save_ply(os.path.join(out_dir, 'raster_projected_%02d_2.ply' % it),
                     proj_points.cpu().numpy(),
                     )
            pcl_removed_outlier = remove_outliers(
                proj_points.view(1, -1, 3), neighborhood_size=16)
            save_ply(os.path.join(out_dir, 'raster_projected_%02d_3.ply' % it),
                     pcl_removed_outlier.points_packed().cpu().numpy(),)

            iso_points_grad = model.get_normals_from_grad(
                proj_points.view(1, -1, 3), requires_grad=False, return_sdf=False)
            iso_raytraced_points, num_iso_raytraced_per_batch = model.sample_inmask_using_isopoints(
                proj_points.view(1, -1, 3), iso_points_grad, mask_img, cameras)

        print('test_uniform_proj_v2, time avg: %.4g' %
              (t_acc / len(data_loader)))

    def test_uniform_proj_v1(self):
        """ Project without prefiltering """
        device = self.device
        out_dir = self.out_dir
        in_dir = self.in_dir
        model = self.model
        decoder = self.model.decoder
        points = self.points
        trainer = self.trainer
        data_loader = self.data_loader

        # project
        projector = UniformProjection(
            max_points_per_pass=16000, proj_max_iters=10,
            proj_tolerance=1e-6, knn_k=8, total_iters=3, sample_iters=1)

        proj_points = points.clone()
        proj_normals = None
        cameras = data_loader.dataset.get_cameras()
        lights = data_loader.dataset.get_lights()
        t_acc = 0
        tproj_acc = 0
        tocc_acc = 0
        for it, batch in enumerate(data_loader):
            data = trainer.process_data_dict(
                batch, cameras=cameras, lights=lights)
            mask_img = data['mask_img']
            # first filter renderable points
            t0 = time.time()
            # pcl_filter = PointCloudsFilters(device=device)
            # fragments = model.renderer.rasterizer(model._points, pcl_filter, cameras=data['camera'])
            # pcl_visible = pcl_filter.filter_with(self.model._points.clone(), ('visibility',))
            pcl_all = model._points.clone()
            pcl_all.offset_(
                (0.1 * (torch.rand_like(pcl_all.points_packed()) - 0.5)))
            proj_results = projector.project_points(pcl_all, decoder)
            t1 = time.time()
            t_acc += (t1 - t0)
            tproj_acc += (t1 - t0)

            # then need to filter visibility using ray marching
            mask = proj_results['mask']
            proj_points = proj_results['levelset_points'][mask].detach()
            t0 = time.time()
            with torch.autograd.no_grad():
                sample_points = proj_points.view(1, -1, 3)
                p_screen_hat = cameras.transform_points(sample_points)
                iso_mask_gt = get_tensor_values(mask_img.float(),
                                                (-p_screen_hat[..., :2]).clamp(-1.0,
                                                                               1.0),
                                                squeeze_channel_dim=True).bool()
                if isinstance(cameras, FoVPerspectiveCameras):
                    iso_incamera = (p_screen_hat[..., 2] >= cameras.znear) & (p_screen_hat[..., 2] <= cameras.zfar) & (
                        p_screen_hat[..., :2].abs() <= 1.0).all(dim=-1)
                else:
                    iso_incamera = (p_screen_hat[..., 2] >= 0) & (
                        p_screen_hat[..., :2].abs() <= 1.0).all(dim=-1)

            iso_points_grad = model.get_normals_from_grad(
                proj_points.view(1, -1, 3), requires_grad=False, return_sdf=False)
            iso_raytraced_points, num_iso_raytraced_per_batch = model.sample_inmask_using_isopoints(
                proj_points.view(1, -1, 3), iso_points_grad, mask_img, cameras)
            t1 = time.time()
            t_acc += (t1 - t0)
            tocc_acc += (t1 - t0)

            save_ply(os.path.join(out_dir, 'all_projected_%02d_1.ply' %
                                  it), proj_points.cpu().numpy(),)
            save_ply(os.path.join(out_dir, 'all_projected_%02d_2.ply' %
                                  it), iso_raytraced_points.cpu().numpy(),)

        print('test_uniform_proj_v1, time avg: %.4g, proj: %.4g, occ: %.4g' % (
            t_acc / len(data_loader), tproj_acc / len(data_loader), tocc_acc / len(data_loader)))

    def test_uniform_proj_time(self):
        """ Project without prefiltering """
        device = self.device
        out_dir = self.out_dir
        in_dir = self.in_dir
        model = self.model
        decoder = self.model.decoder
        points = self.points
        trainer = self.trainer
        data_loader = self.data_loader

        # project
        projector = UniformProjection(
            max_points_per_pass=16000, proj_max_iters=10,
            proj_tolerance=1e-4, knn_k=8, total_iters=1, sample_iters=3)

        forward_time = 0
        project_time = 0
        pcl_all = model._points.clone()
        pcl_all.offset_(
            (0.1 * (torch.rand_like(pcl_all.points_packed()) - 0.5)))
        proj_results = projector.project_points(pcl_all, decoder)
        for it in range(11):
            pcl_all = model._points.clone()
            pcl_all.offset_(
                (0.1 * (torch.rand_like(pcl_all.points_packed()) - 0.5)))
            torch.cuda.synchronize()
            t0 = time.time()
            proj_results = projector.project_points(pcl_all, decoder)
            torch.cuda.synchronize()
            project_time += time.time()-t0

            # then need to filter visibility using ray marching
            mask = proj_results['mask'].view(1, -1)
            proj_points = proj_results['levelset_points'][mask].detach()

            save_ply(os.path.join(out_dir, 'uniform_proj_time_%02d.ply' %
                                    it), proj_points.cpu().numpy(),)

            proj_results = projector.project_points(pcl_all, decoder, skip_resampling=True)

            # then need to filter visibility using ray marching
            mask = proj_results['mask'].view(1, -1)
            proj_points = proj_results['levelset_points'][mask].detach()

            save_ply(os.path.join(out_dir, 'uniform_proj_time_%02d_skipresample.ply' %
                                    it), proj_points.cpu().numpy(),)

            torch.cuda.synchronize()
            t0 = time.time()
            sdf = model.decoder(proj_points).sdf
            loss = sdf.abs().mean()
            loss.backward()
            torch.cuda.synchronize()
            forward_time += time.time()-t0

        print('runtime %f, project time %f' % (forward_time/10.0, project_time/10))

if __name__ == '__main__':
    testcase = TestUniformProj()
    # testcase.test_uniform_proj_v1()
    # testcase.test_uniform_proj_v2()
    testcase.test_step_wise()
    # testcase.test_uniform_proj_time()
