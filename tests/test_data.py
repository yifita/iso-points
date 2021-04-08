"""
Test camera matrics for generated data by reprojecting point cloud/object
"""
import unittest
from pytorch3d.renderer import (
    look_at_view_transform,
    RasterizationSettings,
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.ops import (
    packed_to_padded,
    eyes,
    sample_points_from_meshes,
    padded_to_packed
    )
from pytorch3d.io import load_ply, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes, Textures, padded_to_list
from itertools import chain
from glob import glob
import numpy as np
import imageio
import argparse
import os
import torch
import sys
import trimesh
sys.path.append(".")
from DSS.core.camera import CameraSampler
from DSS.core.texture import LightingTexture
from DSS.utils.dataset import MVRDataset
from DSS.utils.mathHelper import decompose_to_R_and_t
from common import get_tri_color_lights
from im2mesh.common import (transform_to_camera_space,
                            sample_patch_points,
                            arange_pixels,
                            transform_to_world,
                            get_tensor_values)


class TestMVRData(unittest.TestCase):
    def test_dataset(self):
        # 1. rerender input point clouds / meshes using the saved camera_mat
        #    compare mask image with saved mask image
        # 2. backproject masked points to space with dense depth map,
        #    fuse all views and save
        batch_size = 1
        device = torch.device('cuda:0')

        data_dir = 'data/synthetic/cube_mesh'
        output_dir = os.path.join('tests', 'outputs', 'test_data')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # dataset
        dataset = MVRDataset(data_dir=data_dir, load_dense_depth=True, mode="train")
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=0, shuffle=False
        )
        meshes = load_objs_as_meshes([os.path.join(data_dir, 'mesh.obj')]).to(device)
        cams = dataset.get_cameras().to(device)
        image_size = imageio.imread(dataset.image_files[0]).shape[0]

        # initialize rasterizer, we check mask pngs only, so no need to create lights and shaders etc
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=5,
            bin_size = None,  # this setting controls whether naive or coarse-to-fine rasterization is used
            max_faces_per_bin = None  # this setting is for coarse rasterization
        )
        rasterizer = MeshRasterizer(cameras=None, raster_settings=raster_settings)

        # render with loaded cameras positions and training tranformation functions
        pixel_world_all = []
        for idx, data in enumerate(data_loader):
            # get datas
            img = data.get('img.rgb').to(device)
            assert(img.min() >= 0 and img.max() <= 1), "Image must be a floating number between 0 and 1."
            mask_gt = data.get('img.mask').to(device).permute(0, 2, 3, 1)

            camera_mat = data['camera_mat'].to(device)

            cams.R, cams.T = decompose_to_R_and_t(camera_mat)
            cams._N = cams.R.shape[0]
            cams.to(device)
            self.assertTrue(torch.equal(cams.get_world_to_view_transform().get_matrix(), camera_mat))

            # transform to view and rerender with non-rotated camera
            verts_padded = transform_to_camera_space(meshes.verts_padded(), cams)
            meshes_in_view = meshes.offset_verts(
                -meshes.verts_packed() +
                padded_to_packed(verts_padded, meshes.mesh_to_verts_packed_first_idx(), meshes.verts_packed().shape[0]))

            fragments = rasterizer(meshes_in_view, cameras=dataset.get_cameras().to(device))

            # compare mask
            mask = fragments.pix_to_face[..., :1] >= 0
            imageio.imwrite(os.path.join(output_dir, "mask_%06d.png" % idx), mask[0, ...].cpu().to(dtype=torch.uint8)*255)
            # allow 5 pixels difference
            self.assertTrue(torch.sum(mask_gt != mask)<5)

            # check dense maps
            # backproject points to the world pixel range (-1, 1)
            pixels = arange_pixels((image_size, image_size), batch_size)[1].to(device)

            depth_img = data.get('img.depth').to(device)
            # get the depth and mask at the sampled pixel position
            depth_gt = get_tensor_values(depth_img, pixels, squeeze_channel_dim=True)
            mask_gt  = get_tensor_values(mask.permute(0, 3, 1, 2).float(), pixels, squeeze_channel_dim=True).bool()
            # get pixels and depth inside the masked area
            pixels_packed = pixels[mask_gt]
            depth_gt_packed = depth_gt[mask_gt]
            first_idx = torch.zeros((pixels.shape[0],), device=device, dtype=torch.long)
            num_pts_in_mask = mask_gt.sum(dim=1)
            first_idx[1:] = num_pts_in_mask.cumsum(dim=0)[:-1]
            pixels_padded = packed_to_padded(pixels_packed, first_idx, num_pts_in_mask.max().item())
            depth_gt_padded = packed_to_padded(depth_gt_packed, first_idx, num_pts_in_mask.max().item())
            # backproject to world coordinates
            # contains nan and infinite values due to depth_gt_padded containing 0.0
            pixel_world_padded = transform_to_world(pixels_padded, depth_gt_padded[..., None], cams)
            # transform back to list, containing no padded values
            split_size = num_pts_in_mask[..., None].repeat(1, 2)
            split_size[:, 1] = 3
            pixel_world_list = padded_to_list(pixel_world_padded, split_size)
            pixel_world_all.extend(pixel_world_list)

            idx += 1
            if idx >= 10:
                break

        pixel_world_all = torch.cat(pixel_world_all, dim=0)
        mesh = trimesh.Trimesh(vertices=pixel_world_all.cpu(), faces=None,process=False)
        mesh.export(os.path.join(output_dir, 'pixel_to_world.ply'))