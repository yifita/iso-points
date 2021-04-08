import unittest
import numpy as np
import glob
import os
import imageio
import trimesh
import torch
from DSS.utils.mathHelper import decompose_to_R_and_t
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (PerspectiveCameras,
                                MeshRenderer,
                                MeshRasterizer,
                                SoftPhongShader,
                                RasterizationSettings,
                                TexturesVertex)


class TestDVRData(unittest.TestCase):
    """
    parse DVR camera data
    """
    def test_cameras(self):
        """
        DVR cameras
        """
        device = torch.device('cuda:0')
        input_dir = '/home/ywang/Documents/points/neural_splatter/differentiable_volumetric_rendering_upstream/data/DTU/scan106/scan106'
        out_dir = os.path.join('tests', 'outputs', 'test_dvr_data')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        dvr_camera_file = os.path.join(input_dir, 'cameras.npz')
        dvr_camera_dict = np.load(dvr_camera_file)
        n_views = len(glob.glob(os.path.join(input_dir, 'image', '*.png')))

        focal_lengths = dvr_camera_dict['camera_mat_0'][(0,1),(0,1)].reshape(1,2)
        principal_point = dvr_camera_dict['camera_mat_0'][(0,1),(2,2)].reshape(1,2)
        cameras = PerspectiveCameras(focal_length=focal_lengths, principal_point=principal_point).to(device)
        # Define the settings for rasterization and shading.
        # Refer to raster_points.py for explanations of these parameters.
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=5,
            # this setting controls whether naive or coarse-to-fine rasterization is used
            bin_size=None,
            max_faces_per_bin=None  # this setting is for coarse rasterization
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=None, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device)
        )
        mesh = trimesh.load_mesh('/home/ywang/Documents/points/neural_splatter/differentiable_volumetric_rendering_upstream/out/multi_view_reconstruction/birds/ours_depth_mvs/vis/000_0000477500.ply')
        textures = TexturesVertex(verts_features=torch.ones(
            1, mesh.vertices.shape[0], 3)).to(device=device)
        meshes = Meshes(verts=[torch.tensor(mesh.vertices).float()], faces=[torch.tensor(mesh.faces)],
                        textures=textures).to(device=device)
        for i in range(n_views):
            transform_mat = torch.from_numpy(dvr_camera_dict['scale_mat_%d' % i].T @ dvr_camera_dict['world_mat_%d' % i].T).to(device).unsqueeze(0).float()
            cameras.R, cameras.T = decompose_to_R_and_t(transform_mat)
            cameras._N = cameras.R.shape[0]
            imgs = renderer(meshes, cameras=cameras, zfar=1e4, znear=1.0)
            import pdb; pdb.set_trace()
            imageio.imwrite(os.path.join(out_dir, '%06d.png' % i), (imgs[0].detach().cpu().numpy()*255).astype('uint8'))