import os
import argparse
import csv
import numpy as np
from glob import glob
from collections import OrderedDict, defaultdict
import config
import trimesh
from tqdm import tqdm

import torch
import pytorch3d
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance, chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
import point_cloud_utils as pcu
from DSS import set_deterministic_
from DSS.utils.io import read_ply

"""
Given an experiment folder, evaluate the meshes `vis/*_mesh.ply` and `generation/mesh00.ply`,
write the results in `val/all.csv`
"""

set_deterministic_()


def get_filenames(source, extension):
    # If extension is a list
    if source is None:
        return []
    # Seamlessy load single file, list of files and files from directories.
    source_fns = []
    if isinstance(source, str):
        if os.path.isdir(source):
            if not isinstance(extension, str):
                for fmt in extension:
                    source_fns += get_filenames(source, fmt)
            else:
                source_fns = sorted(
                    glob("{}/**/*{}".format(source, extension), recursive=True))
        elif os.path.isfile(source):
            source_fns = [source]
    elif len(source) and isinstance(source[0], str):
        for s in source:
            source_fns.extend(get_filenames(s, extension=extension))
    return source_fns


def eval_one_dir(exp_dir, n_pts=50000):
    """
    Function for one directory
    """
    device = torch.device('cuda:0')
    cfg = config.load_config(os.path.join(exp_dir, 'config.yaml'))
    dataset = config.create_dataset(cfg.data, mode='val')
    meshes_gt = dataset.get_meshes().to(device)
    val_gt_pts_file = os.path.join(cfg.data.data_dir, 'val%d.ply' % n_pts)
    if os.path.isfile(val_gt_pts_file):
        points, normals = np.split(read_ply(val_gt_pts_file), 2, axis=1)
        pcl_gt = Pointclouds(torch.from_numpy(points[None, ...]).float(),
                             torch.from_numpy(normals[None, ...]).float()).to(device)
    else:
        pcl_gt = dataset.get_pointclouds(n_pts).to(device)
        trimesh.Trimesh(pcl_gt.points_packed().cpu().numpy(),
                        vertex_normals=pcl_gt.normals_packed().cpu().numpy(), process=False).export(
            val_gt_pts_file, vertex_normal=True
        )

    # load vis directories
    vis_dir = os.path.join(exp_dir, 'vis')
    vis_files = sorted(get_filenames(vis_dir, '_mesh.ply'))
    iters = [int(os.path.basename(v).split('_')[0]) for v in vis_files]
    best_dict = defaultdict(lambda: float('inf'))
    vis_eval_csv = os.path.join(vis_dir, "evaluation_n%d.csv" % n_pts)
    if not os.path.isfile(vis_eval_csv):
        with open(os.path.join(vis_dir, "evaluation_n%d.csv" % n_pts), "w") as f:
            fieldnames = ['mtime', 'it', 'chamfer_p', 'chamfer_n', 'pf_dist']
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    restval="-", extrasaction="ignore")
            writer.writeheader()
            mtime0 = None
            for it, vis_file in zip(iters, vis_files):
                eval_dict = OrderedDict()
                mtime = os.path.getmtime(vis_file)
                if mtime0 is None:
                    mtime0 = mtime
                eval_dict['it'] = it
                eval_dict['mtime'] = mtime - mtime0
                val_pts_file = os.path.join(vis_dir, os.path.basename(
                    vis_file).replace('_mesh', '_val%d' % n_pts))
                if os.path.isfile(val_pts_file):
                    points, normals = np.split(
                        read_ply(val_pts_file), 2, axis=1)
                    points = torch.from_numpy(points).float().to(
                        device=device).view(1, -1, 3)
                    normals = torch.from_numpy(normals).float().to(
                        device=device).view(1, -1, 3)
                else:
                    mesh = trimesh.load(vis_file, process=False)
                    # points, normals = pcu.sample_mesh_poisson_disk(
                    #     mesh.vertices, mesh.faces,
                    #     mesh.vertex_normals.ravel().reshape(-1, 3), n_pts, use_geodesic_distance=True)
                    # p_idx = np.random.permutation(points.shape[0])[:n_pts]
                    # points = points[p_idx, ...]
                    # normals = normals[p_idx, ...]
                    # points = torch.from_numpy(points).float().to(
                    #     device=device).view(1, -1, 3)
                    # normals = torch.from_numpy(normals).float().to(
                    #     device=device).view(1, -1, 3)
                    meshes = Meshes(torch.from_numpy(mesh.vertices[None, ...]).float(),
                                    torch.from_numpy(mesh.faces[None, ...]).float()).to(device)
                    points, normals = sample_points_from_meshes(
                        meshes, n_pts, return_normals=True)
                    trimesh.Trimesh(points.cpu().numpy()[0], vertex_normals=normals.cpu().numpy()[0], process=False).export(
                        val_pts_file, vertex_normal=True
                    )
                pcl = Pointclouds(points, normals)
                chamfer_p, chamfer_n = chamfer_distance(
                    points, pcl_gt.points_padded(),
                    x_normals=normals, y_normals=pcl_gt.normals_padded(),
                )
                eval_dict['chamfer_p'] = chamfer_p.item()
                eval_dict['chamfer_n'] = chamfer_n.item()
                pf_dist = point_mesh_face_distance(meshes_gt, pcl)
                eval_dict['pf_dist'] = pf_dist.item()
                writer.writerow(eval_dict)
                for k, v in eval_dict.items():
                    if v < best_dict[k]:
                        best_dict[k] = v
                        print('best {} so far ({}): {:.4g}'.format(k, vis_file, v))

    # generation dictories
    gen_dir = os.path.join(exp_dir, 'generation')
    if not os.path.isdir(gen_dir):
        return

    final_file = os.path.join(gen_dir, 'mesh.ply')
    val_pts_file = final_file[:-4] + '_val%d' % n_pts + '.ply'
    if not os.path.isfile(final_file):
        return

    gen_file_csv = os.path.join(gen_dir, "evaluation_n%d.csv" % n_pts)
    if not os.path.isfile(gen_file_csv):
        with open(os.path.join(gen_dir, "evaluation_n%d.csv" % n_pts), "w") as f:
            fieldnames = ['chamfer_p', 'chamfer_n', 'pf_dist']
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    restval="-", extrasaction="ignore")
            writer.writeheader()
            eval_dict = OrderedDict()
            mesh = trimesh.load(final_file)
            # points, normals = pcu.sample_mesh_poisson_disk(
            #     mesh.vertices, mesh.faces,
            #     mesh.vertex_normals.ravel().reshape(-1, 3), n_pts, use_geodesic_distance=True)
            # p_idx = np.random.permutation(points.shape[0])[:n_pts]
            # points = points[p_idx, ...]
            # normals = normals[p_idx, ...]
            # points = torch.from_numpy(points).float().to(
            #     device=device).view(1, -1, 3)
            # normals = torch.from_numpy(normals).float().to(
            #     device=device).view(1, -1, 3)
            meshes = Meshes(torch.from_numpy(mesh.vertices[None, ...]).float(),
                            torch.from_numpy(mesh.faces[None, ...]).float()).to(device)
            points, normals = sample_points_from_meshes(
                meshes, n_pts, return_normals=True)
            trimesh.Trimesh(points.cpu().numpy()[0], vertex_normals=normals.cpu().numpy()[0], process=False).export(
                val_pts_file, vertex_normal=True)
            pcl = Pointclouds(points, normals)
            chamfer_p, chamfer_n = chamfer_distance(
                points, pcl_gt.points_padded(),
                x_normals=normals, y_normals=pcl_gt.normals_padded(),
            )
            eval_dict['chamfer_p'] = chamfer_p.item()
            eval_dict['chamfer_n'] = chamfer_n.item()
            pf_dist = point_mesh_face_distance(meshes_gt, pcl)
            eval_dict['pf_dist'] = pf_dist.item()
            writer.writerow(eval_dict)
            for k, v in eval_dict.items():
                if v < best_dict[k]:
                    best_dict[k] = v
                    print('best {} so far ({}): {:.4g}'.format(k, final_file, v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", type=str, nargs='+', required=True,
                        help="Experiment directories")
    parser.add_argument("--n_pts", type=int, default=50000,
                        help="number of points used for evaluation")
    args = parser.parse_args()
    for exp in args.dirs:
        eval_one_dir(exp, args.n_pts)
