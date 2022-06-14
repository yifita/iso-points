import torch
from numbers import Number
# import torch.distributions as dist
import os
import shutil
import argparse
import trimesh
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
import numpy as np
import config
from DSS.misc.checkpoints import CheckpointIO
import imageio
import plotly.graph_objs as go
from DSS import logger_py
from DSS.utils import get_surface_high_res_mesh


parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--img_size', type=int, nargs='*', help="overwrite original image size")
parser.add_argument('--resolution', type=int, default=512,
                    help='Overrites the default resolution in config')
parser.add_argument('--mesh-only', action='store_true')
parser.add_argument('--render-only', action='store_true')


args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')


is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shortcuts
out_dir = os.path.join(cfg['training']['out_dir'], cfg['name'])
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
if not os.path.exists(generation_dir):
    os.makedirs(generation_dir)

batch_size = 1
vis_n_outputs = cfg['generation']['vis_n_outputs']
mesh_extension = cfg['generation']['mesh_extension']

# Dataset
dataset = config.create_dataset(cfg.data, mode='test')
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, num_workers=1, shuffle=False)
img_size = args.img_size or dataset.resolution
if isinstance(img_size, Number):
    img_size = (img_size, img_size)

# Model
model = config.create_model(cfg, mode='test', device=device, camera_model=dataset.get_cameras()).to(device=device)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.create_generator(cfg, model, device=device)

torch.manual_seed(0)

# Generate
with torch.autograd.no_grad():
    model.eval()
    # Generate meshes
    if not args.render_only:
        logger_py.info('Generating mesh...')
        mesh = get_surface_high_res_mesh(lambda x: model.decode(x).sdf.squeeze(), resolution=args.resolution)
        if cfg.data.type == 'DTU':
            mesh.apply_transform(dataset.get_scale_mat())
        mesh_out_file = os.path.join(generation_dir, 'mesh.%s' % mesh_extension)
        mesh.export(mesh_out_file)

    # Generate cuts
    logger_py.info('Generating cross section plots')
    img = generator.generate_iso_contour(imgs_per_cut=5)
    out_file = os.path.join(generation_dir, 'iso')
    img.write_html(out_file + '.html')

    if not args.mesh_only:
        # generate images
        for i, batch in enumerate(test_loader):
            img_mask = batch['img.mask']
            cam_mat = batch['camera_mat']
            cameras = dataset.get_cameras(cam_mat)
            lights = dataset.get_lights(**batch.get('lights', {}))
            rgbas = generator.raytrace_images(img_size, img_mask, cameras=cameras, lights=lights)
            for rgba in rgbas:
                imageio.imwrite(os.path.join(generation_dir, '%05d.png' % i), rgba)
            torch.cuda.empty_cache()
