import unittest
import os
import trimesh
import imageio
import torch
import numpy as np
import point_cloud_utils as pcu
from collections import defaultdict
from DSS.models.common import SDF
from DSS.models.combined_modeling import Model, Generator
from DSS.core.texture import LightingTexture
from DSS.core.renderer import SurfaceSplattingRenderer
from DSS.core.rasterizer import PointsRasterizationSettings
from DSS.core.rasterizer import SurfaceSplatting
from pytorch3d.renderer import NormWeightedCompositor
from pytorch3d.loss import point_mesh_face_distance, chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from DSS.utils import check_weights
from DSS.utils.dataset import MVRDataset
from DSS.utils.mathHelper import decompose_to_R_and_t
from DSS.misc.visualize import plot_3D_quiver
from DSS.training.losses import NormalLengthLoss
from DSS import set_debugging_mode_, get_debugging_tensor, get_debugging_mode
from DSS.utils import tolerating_collate


def sample_iso_points_from_meshes(generator):
    output = []
    n_points = generator.model.n_points_per_cloud
    from DSS.models.implicit_modeling import Generator
    Generator.generate_meshes(generator, {}, outputs=output)
    try:
        mesh = output.pop()
        points, _ = pcu.sample_mesh_poisson_disk(
            mesh.vertices, mesh.faces, mesh.vertex_normals.ravel().reshape(-1, 3), n_points, use_geodesic_distance=True)
        p_idx = np.random.permutation(points.shape[0])[:n_points]
        points = points[p_idx, ...]
        generator.model.points = torch.tensor(
            points, dtype=torch.float, device=generator.model.device).view(1, -1, 3)
    except Exception as e:
        print("Couldn't sample points from mesh: " + repr(e))

def process_data_dict(device, data, cameras):
    ''' Processes the data dictionary and returns respective tensors

    Args:
        data (dictionary): data dictionary
    '''
    # Get "ordinary" data
    img = data.get('img.rgb').to(device)
    assert(img.min() >= 0 and img.max() <=
           1), "Image must be a floating number between 0 and 1."
    mask_img = data.get('img.mask').to(device)

    camera_mat = data.get('camera_mat', None)

    # inputs for SVR
    inputs = data.get('inputs', torch.empty(0, 0)).to(device)

    mesh = data.get('shape.mesh', None)
    if mesh is not None:
        mesh = mesh.to(device=device)

    # set camera matrix to cameras
    if camera_mat is None:
        logger_py.warning(
            "Camera matrix is not provided! Using the identity matrix")

    cameras.R, cameras.T = decompose_to_R_and_t(camera_mat)
    cameras._N = cameras.R.shape[0]
    cameras.to(device)

    return (img, mask_img, inputs, mesh), cameras


class TestProjection(unittest.TestCase):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    # def test_projection(self):
    #     """

    #     """
    #     device = torch.device('cuda:0')
    #     data_dir = "data/synthetic/Torus"
    #     out_dir = os.path.join('tests', 'outputs',
    #                            os.path.splitext(
    #                                os.path.basename(__file__))[0],
    #                            'test_SDF')
    #     if not os.path.exists(out_dir):
    #         os.makedirs(out_dir)

    #     dataset = MVRDataset(data_dir)
    #     data_loader = torch.utils.data.DataLoader(
    #         dataset, batch_size=4, num_workers=0, shuffle=True,
    #         collate_fn=tolerating_collate,
    #     )

    #     raster_params = {
    #         "points_per_pixel": 5,
    #         "cutoff_threshold": 0.5,
    #         "depth_merging_threshold": 0.02,
    #         "Vrk_isotropic": True,
    #         "radii_backward_scaler": 10,
    #         "image_size": 512,
    #         "points_per_pixel": 5,
    #         "bin_size": None,
    #         "max_points_per_bin": None,
    #         "backward_rbf": False,
    #     }
    #     raster_setting = PointsRasterizationSettings(**raster_params)
    #     rasterizer = SurfaceSplatting(
    #         cameras=None, raster_settings=raster_setting
    #     )
    #     compositor = NormWeightedCompositor()
    #     renderer = SurfaceSplattingRenderer(
    #         rasterizer=rasterizer,
    #         compositor=compositor,
    #         backface_culling=True
    #     )

    #     lights = dataset.get_lights().to(device=device)
    #     shader = LightingTexture(specular=False, lights=lights)

    #     decoder_params = {
    #         'pos_encoding': False,
    #         "out_dim": 1,
    #         "c_dim": 0,
    #         "hidden_size": 512,
    #         'n_layers': 5,
    #         "dropout": [],
    #         "dropout_prob": 0.2,
    #         "norm_layers": [0, 1, 2, 3, 4],
    #         "latent_in": [],
    #         "xyz_in_all": False,
    #         "activation": None,
    #         "latent_dropout": False,
    #         "weight_norm": True,
    #     }
    #     decoder = SDF(**decoder_params)
    #     n_points_per_cloud = 8000
    #     model = Model(
    #         decoder, renderer, shader=shader, encoder=None, device=device,
    #         n_points_per_cloud=n_points_per_cloud, sample_iters=1,
    #     )
    #     generator = Generator(model, device=device, threshold=0.5, resolution=256,
    #                           upsampling_steps=3, with_colors=True)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #     eikonal_loss = NormalLengthLoss(reduction='mean')

    #     # create ground truth from the ground truth mesh
    #     mesh_gt = dataset.get_meshes().to(device)
    #     points_gt, normals_gt = sample_points_from_meshes(
    #         mesh_gt, n_points_per_cloud, return_normals=True)
    #     pcl = trimesh.Trimesh(vertices=points_gt.cpu()[0],
    #                           vertex_normals=normals_gt.cpu()[0],
    #                           process=False)

    #     pcl.export(os.path.join(out_dir, "GT_PCL.ply"),
    #                vertex_normal=True)

    #     # create camera instance
    #     cameras = dataset.get_cameras().to(device)

    #     # start training
    #     it = 0
    #     torch.autograd.set_detect_anomaly(True)
#         for batch in data_loader:
#             model.train()
#             optimizer.zero_grad()
#             loss = defaultdict(lambda: 0.0)

#             data, cameras = process_data_dict(device, batch, cameras)
#             # Process data dictionary
#             img, mask_img, _, _ = data

#             with torch.autograd.no_grad():
#                 returns = model(mask_img,
#                                 cameras=cameras, inputs=None, it=it, eval_mode=False)
#                 rgb_pred = returns[-2]
#                 mask_pred = returns[-1]

#             # debug
#             if it % 1 == 0:
#                 set_debugging_mode_(True)

#             point_clouds = model.get_point_clouds(
#                 project=True, with_normals=True, require_normals_grad=True)
#             points_pred = point_clouds[0].points_padded()
#             normals_pred = point_clouds[0].normals_padded()
#             if get_debugging_mode():
#                 model._debugger_logging(
#                     points_pred, points_pred.new_full(
#                         points_pred.shape[:-1], True, dtype=torch.bool),
#                     None, None, None, None)

#             # 1. eikonal term
#             normals = point_clouds.normals_packed()
#             loss_eikonal = eikonal_loss(normals) * 0.0001
#             loss['loss_eikonal'] = loss_eikonal
#             loss['loss'] += loss_eikonal

#             # 2. 3D loss
#             # loss_3D = point_mesh_face_distance(
#             #     mesh_gt.extend(len(point_clouds)), point_clouds) * 100
#             loss_3D, loss_3D_normal = chamfer_distance(
#                 points_pred, points_gt, x_normals=normals_pred, y_normals=normals_gt, point_reduction='sum')
#             loss['loss_3D'] = loss_3D
#             loss['loss'] += loss_3D

#             # 3. boundary loss
#             points = point_clouds.points_packed()
#             loss_bdry = torch.nn.functional.relu(
#                 points.norm(dim=-1) - 1).mean()
#             loss['loss_boundary'] = loss_bdry
#             loss['loss'] += loss_bdry

#             loss['loss'].backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
#             optimizer.step()
#             check_weights(model.state_dict())

#             print("iter {:05d} loss_3D: {:.3g} loss_eikonal: {:.3g} loss_boundary: {:.3g}".format(
#                 it, loss['loss_3D'], loss['loss_eikonal'], loss['loss_boundary']))

#             # outputs point clouds, render, gradients
#             if get_debugging_mode():
#                 rgba_pred = torch.cat([rgb_pred[0], mask_pred[0]], dim=-1)
#                 rgba_pred = (
#                     rgba_pred * 255.0).to(torch.uint8).detach().cpu().numpy()
#                 imageio.imwrite(os.path.join(
#                     out_dir, "{:05d}_RGBA.png".format(it)), rgba_pred)
#                 points = point_clouds[0].points_packed(
#                 ).cpu().detach().numpy()
#                 normals = point_clouds[0].normals_packed(
#                 ).cpu().detach().numpy()
#                 colors = point_clouds[0].features_packed(
#                 ).cpu().detach().numpy()
#                 pcl = trimesh.Trimesh(vertices=points,
#                                       vertex_normals=normals,
#                                       vertex_colors=colors[:, :3], process=False)

#                 pcl.export(os.path.join(out_dir, "{:05d}_PCL.ply".format(it)),
#                            vertex_normal=True)
#                 dbg_tensor = get_debugging_tensor()
#                 k_names = list(dbg_tensor.pts_world_grad.keys())
#                 grad_dict = {
#                     k: dbg_tensor.pts_world_grad[k][0] for k in k_names}
#                 pts_dict = {k: dbg_tensor.pts_world[k][0] for k in k_names}
#                 _cams = cameras.clone().to(device='cpu')
#                 _cams.R = _cams[:1].R
#                 _cams.T = _cams[:1].T
#                 _cams._N = 1
#                 if len(pts_dict) > 0:
#                     plot_3D_quiver(pts_dict, grad_dict, mesh_gt=mesh_gt.to('cpu'),
#                                    camera=_cams,
#                                    save_html=os.path.join(out_dir, "{:05d}_WORLD.html".format(it)))

#                 set_debugging_mode_(False)
#                 model.debug(False)

#             it += 1

    #         if (it >= 150):
    #             mesh_outputs = []
    #             generator.generate_mesh({}, outputs=mesh_outputs)
    #             mesh_outputs.pop().export(os.path.join(out_dir, "final.obj"), include_color=True)
    #             break

    def test_posEnc(self):
        """
        Test projection when positional encoding is True
        """
        device = torch.device('cuda:0')
        data_dir = "data/synthetic/Torus"
        out_dir = os.path.join('tests', 'outputs',
                               os.path.splitext(
                                   os.path.basename(__file__))[0],
                               'test_posEnc')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        dataset = MVRDataset(data_dir)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, num_workers=0, shuffle=True,
            collate_fn=tolerating_collate,
        )

        raster_params = {
            "points_per_pixel": 5,
            "cutoff_threshold": 0.5,
            "depth_merging_threshold": 0.02,
            "Vrk_isotropic": True,
            "radii_backward_scaler": 10,
            "image_size": 512,
            "points_per_pixel": 5,
            "bin_size": None,
            "max_points_per_bin": None,
            "backward_rbf": False,
        }
        raster_setting = PointsRasterizationSettings(**raster_params)
        rasterizer = SurfaceSplatting(
            cameras=None, raster_settings=raster_setting
        )
        compositor = NormWeightedCompositor()
        renderer = SurfaceSplattingRenderer(
            rasterizer=rasterizer,
            compositor=compositor,
            backface_culling=True
        )

        lights = dataset.get_lights().to(device=device)
        shader = LightingTexture(specular=False, lights=lights)

        decoder_params = {
            'pos_encoding': True,
            "out_dim": 1,
            "c_dim": 0,
            "hidden_size": 512,
            'n_layers': 5,
            "dropout": [],
            "dropout_prob": 0.2,
            "norm_layers": [0, 1, 2, 3, 4],
            "latent_in": [],
            "xyz_in_all": False,
            "activation": None,
            "latent_dropout": False,
            "weight_norm": True,
        }
        decoder = SDF(**decoder_params)
        n_points_per_cloud = 8000
        model = Model(
            decoder, renderer, shader=shader, encoder=None, device=device,
            uniform_projection=True,
            n_points_per_cloud=n_points_per_cloud, proj_max_iters=10,
            exact_gradient=True, approx_grad_step=0.05
        )
        generator = Generator(model, device=device, threshold=0.5, resolution=256,
                              upsampling_steps=3, with_colors=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        eikonal_loss = NormalLengthLoss(reduction='mean')

        # create ground truth from the ground truth mesh
        mesh_gt = dataset.get_meshes().to(device)
        points_gt, normals_gt = sample_points_from_meshes(
            mesh_gt, n_points_per_cloud, return_normals=True)
        pcl = trimesh.Trimesh(vertices=points_gt.cpu()[0],
                              vertex_normals=normals_gt.cpu()[0],
                              process=False)

        pcl.export(os.path.join(out_dir, "GT_PCL.ply"),
                   vertex_normal=True)

        # create camera instance
        cameras = dataset.get_cameras().to(device)

        # start training
        it = 0
        torch.autograd.set_detect_anomaly(True)
        for batch in data_loader:
            model.train()
            optimizer.zero_grad()
            loss = defaultdict(lambda: 0.0)

            # debug
            if it % 20 == 0:
                set_debugging_mode_(True)

            point_clouds = model.get_point_clouds(
                project=True, with_normals=True, require_normals_grad=True)
            points_pred = point_clouds.points_padded()
            normals_pred = point_clouds.normals_padded()
            if get_debugging_mode():
                model._debugger_logging(
                    points_pred, points_pred.new_full(
                        points_pred.shape[:-1], True, dtype=torch.bool),
                    None, None, None, None)

            # 1. eikonal term
            normals = point_clouds.normals_packed()
            loss_eikonal = eikonal_loss(normals) * 0.0001
            loss['loss_eikonal'] = loss_eikonal
            loss['loss'] += loss_eikonal

            # 2. 3D loss
            # loss_3D = point_mesh_face_distance(
            #     mesh_gt.extend(len(point_clouds)), point_clouds) * 100
            loss_3D, loss_3D_normal = chamfer_distance(
                points_pred, points_gt, point_reduction='sum')
            loss_3D, loss_3D_normal = chamfer_distance(
                points_pred, points_gt, x_normals=normals_pred, y_normals=normals_gt, point_reduction='sum')
            loss['loss_3D'] = loss_3D + loss_3D_normal
            loss['loss'] += loss_3D

            # 3. boundary loss
            points = point_clouds.points_packed()
            loss_bdry = torch.nn.functional.relu(
                points.norm(dim=-1) - 1).mean()
            loss['loss_boundary'] = loss_bdry
            loss['loss'] += loss_bdry

            loss['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            print("iter {:05d} loss_3D: {:.3g} loss_eikonal: {:.3g} loss_boundary: {:.3g}".format(
                it, loss['loss_3D'], loss['loss_eikonal'], loss['loss_boundary']))

            # outputs point clouds, render, gradients
            if get_debugging_mode():
                points = point_clouds.points_packed(
                ).cpu().detach().numpy()
                normals = point_clouds.normals_packed(
                ).cpu().detach().numpy()
                colors = point_clouds.features_packed(
                ).cpu().detach().numpy()
                pcl = trimesh.Trimesh(vertices=points,
                                        vertex_normals=normals,
                                        vertex_colors=colors[:, :3], process=False)

                pcl.export(os.path.join(out_dir, "{:05d}_PCL.ply".format(it)),
                            vertex_normal=True)
                dbg_tensor = get_debugging_tensor()
                k_names = list(dbg_tensor.pts_world_grad.keys())
                grad_dict = {
                    k: dbg_tensor.pts_world_grad[k][0] for k in k_names}
                pts_dict = {k: dbg_tensor.pts_world[k][0] for k in k_names}
                _cams = cameras.clone().to(device='cpu')
                _cams.R = _cams[:1].R
                _cams.T = _cams[:1].T
                _cams._N = 1
                if len(pts_dict) > 0:
                    plot_3D_quiver(pts_dict, grad_dict, mesh_gt=mesh_gt.to('cpu'),
                                    camera=_cams,
                                    save_html=os.path.join(out_dir, "{:05d}_WORLD.html".format(it)))

                img_list = []
                generator.generate_images({}, outputs=img_list)
                for i, img in enumerate(img_list):
                    out_file = os.path.join(
                        out_dir, '%05d_%02d.png' % (it, i))
                    imageio.imwrite(out_file, img)

                set_debugging_mode_(False)
                model.debug(False)

            optimizer.step()
            check_weights(model.state_dict())

            sample_iso_points_from_meshes(generator)

            # additionally outputs previous iso points + normals *after* network update
            point_clouds = model.get_point_clouds(
                project=False, with_normals=True, require_normals_grad=False)
            points = point_clouds.points_packed(
            ).cpu().detach().numpy()
            normals = point_clouds.normals_packed(
            ).cpu().detach().numpy()
            colors = point_clouds.features_packed(
            ).cpu().detach().numpy()
            pcl = trimesh.Trimesh(vertices=points,
                                    vertex_normals=normals,
                                    vertex_colors=colors[:, :3], process=False)

            pcl.export(os.path.join(out_dir, "{:05d}_PCL_postUpdate.ply".format(it)),
                        vertex_normal=True)

            it += 1

            if (it > 200):
                mesh_outputs = []
                generator.generate_mesh({}, outputs=mesh_outputs)
                mesh_outputs.pop().export(os.path.join(out_dir, "final.obj"), include_color=True)
                break
