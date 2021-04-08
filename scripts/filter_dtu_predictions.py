import torch
import os
import argparse
from tqdm import tqdm, trange
import numpy as np
from im2mesh import config
from DSS.utils.io import read_ply, save_ply
from DSS.utils.dataset import DTUDataset
from skimage.morphology import binary_dilation, disk
import open3d


if __name__ == '__main__':
    # Adjust this to your paths; the input path should point to the
    # DTU dataset including the mvs data which can be downloaded here
    # http://roboimagedata.compute.dtu.dk/
    INPUT_PATH = '/home/mnt/points/data/DTU_MVS/'
    INPUT_PATH = os.path.join(INPUT_PATH, 'Points')
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("The input path is not pointing to the DTU Dataset. " + \
            "Please download the DTU Dataset and adjust your input path.")

    methods = ['furu', 'tola', 'camp', 'stl']
    # Shortcuts
    out_dir = '/home/mnt/points/data/DTU_MVS/Points'
    generation_dir = os.path.join(out_dir, '..', 'Points_in_Mask')

    if not os.path.isdir(generation_dir):
        os.makedirs(generation_dir)

    parser = argparse.ArgumentParser(
        description='Filter the DTU baseline predictions with the object masks.'
    )
    parser.add_argument('scan_ids', type=int, nargs='+', help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    args = parser.parse_args()
    scan_ids = args.scan_ids

    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")


    def filter_points(p, data):
        n_images = len(data)

        p = torch.from_numpy(p)
        n_p = p.shape[0]
        inside_mask = np.ones((n_p,), dtype=np.bool)
        inside_img = np.zeros((n_p,), dtype=np.bool)
        for i in trange(n_images):
            # get data
            maski_in = data.object_masks[i][0].astype('float32')

            # Apply binary dilation to account for errors in the mask
            maski = torch.from_numpy(binary_dilation(maski_in, disk(12))).float()

            #h, w = maski.shape
            h, w = maski.shape
            w_mat = torch.from_numpy(data.data_dict['world_mat_%d' % i])
            c_mat = torch.from_numpy(data.data_dict['camera_mat_%d' % i])
            s_mat = torch.from_numpy(data.data_dict['scale_mat_%d' % i])

            # project points into image
            phom = torch.cat([p, torch.ones(n_p, 1)], dim=-1).transpose(1, 0)
            proj = c_mat @ w_mat @ phom
            proj = (proj[:2] / proj[-2].unsqueeze(0)).transpose(1, 0)

            # check which points are inside image; by our definition,
            # the coordinates have to be in [-1, 1]
            mask_p_inside = ((proj[:, 0] >= -1) &
                (proj[:, 1] >= -1) &
                (proj[:, 0] <= 1) &
                (proj[:, 1] <= 1)
            )
            inside_img |= mask_p_inside.cpu().numpy()

            # get image coordinates
            proj[:, 0] = (proj[:, 0] + 1) * (w - 1) / 2.
            proj[:, 1] = (proj[:, 1] + 1) * (h - 1) / 2.
            proj = proj.long()

            # fill occupancy values
            proj = proj[mask_p_inside]
            occ = torch.ones(n_p)
            occ[mask_p_inside] = maski[proj[:, 1], proj[:, 0]]
            inside_mask &= (occ.cpu().numpy() >= 0.5)

        occ_out = np.zeros((n_p,))
        occ_out[inside_img & inside_mask] = 1.

        return occ_out

    # Dataset
    for scan_id in scan_ids:
        dataset = DTUDataset('data/DTU/scan%d' % scan_id)

        for method in methods:
            out_dir = os.path.join(generation_dir, method)

            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            in_dir = os.path.join(INPUT_PATH, method)
            if method != 'stl':
                scan_path = os.path.join(in_dir, '%s%03d_l3.ply' % (method, scan_id))
            else:
                scan_path = os.path.join(in_dir, '%s%03d_total.ply' % (method, scan_id))

            print(scan_path)

            out_file = os.path.join(out_dir, 'scan%d.ply' % scan_id)
            if not os.path.exists(out_file):
                pcl = open3d.io.read_point_cloud(scan_path)
                p = np.asarray(pcl.points).astype(np.float32)
                occ = filter_points(p, dataset) > 0.5
                pcl.points = open3d.utility.Vector3dVector(p[occ])
                if len(pcl.colors) != 0:
                    c = np.asarray(pcl.colors)
                    pcl.colors = open3d.utility.Vector3dVector(c[occ])
                if len(pcl.normals) != 0:
                    n = np.asarray(pcl.normals)
                    pcl.normals = open3d.utility.Vector3dVector(n[occ])
                open3d.io.write_point_cloud(out_file, pcl)
