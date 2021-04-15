import argparse
import git
import time
import numpy as np
import os
import logging
import config
import torch
import torch.optim as optim
from DSS.models import ImplicitModel, PointModel
from DSS.utils import tolerating_collate
from common import create_animation
from DSS.misc.checkpoints import CheckpointIO
from DSS import logger_py, set_deterministic_

set_deterministic_()

# Arguments
parser = argparse.ArgumentParser(
    description='Train implicit representations without 3D supervision.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')

is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands
out_dir = os.path.join(cfg['training']['out_dir'], cfg['name'])
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after
lr = cfg['training']['learning_rate']
batch_size = cfg['training']['batch_size']
batch_size_val = cfg['training']['batch_size_val']
n_workers = cfg['training']['n_workers']
model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')


# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Begin logging also to the log file
fileHandler = logging.FileHandler(os.path.join(out_dir, cfg.training.logfile))
fileHandler.setLevel(logging.DEBUG)
logger_py.addHandler(fileHandler)

repo = git.Repo(search_parent_directories=False)
sha = repo.head.object.hexsha
logger_py.debug('Git commit: %s' % sha)

# Data
train_dataset = config.create_dataset(cfg.data, mode='train')
val_dataset = config.create_dataset(cfg.data, mode='val')
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size_val, num_workers=int(n_workers // 2),
    shuffle=False, collate_fn=tolerating_collate,
)
# data_viz = next(iter(val_loader))
model = config.create_model(cfg, camera_model=train_dataset.get_cameras(), device=device)

# Create rendering objects from loaded data
cameras = train_dataset.get_cameras()
lights = train_dataset.get_lights()


# Optimizer
if cfg.model.type == 'point':
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
else:
    if cfg.renderer.is_neural_texture:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

# Loads checkpoints
checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load(cfg.training.resume_from)
except FileExistsError:
    load_dict = dict()

epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)

# Save config to log directory
config.save_config(os.path.join(out_dir, 'config.yaml'), cfg)

# Update Metrics from loaded
model_selection_metric = cfg['training']['model_selection_metric']
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

logger_py.info('Current best validation metric (%s): %.8f'
               % (model_selection_metric, metric_val_best))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']
debug_every = cfg['training']['debug_every']
reweight_every = cfg['training']['reweight_every']

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, cfg['training']['scheduler_milestones'],
    gamma=cfg['training']['scheduler_gamma'], last_epoch=epoch_it)

# Set mesh extraction to low resolution for fast visuliation
# during training
cfg['generation']['resolution'] = cfg['training']['visualize_resolution']
cfg['generation']['img_size'] = tuple(x//4 for x in train_dataset.resolution)
generator = config.create_generator(cfg, model, device=device)
trainer = config.create_trainer(
    cfg, model, optimizer, scheduler, generator, None, val_loader, device=device)

# Print model
nparameters = sum(p.numel() for p in model.parameters())
logger_py.info('Total number of parameters: %d' % nparameters)


# Start training loop
t0 = time.time()
t0b = time.time()
sample_weights = np.ones(len(train_dataset)).astype('float32')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            num_workers=n_workers, drop_last=True,
                                            collate_fn=tolerating_collate)
while True:
    epoch_it += 1
    for batch in train_loader:
        it += 1

        loss = trainer.train_step(batch, cameras=cameras, lights=lights, it=it)

        # Visualize output
        if it > 0 and visualize_every > 0 and (it % visualize_every) == 0:
            logger_py.info('Visualizing')
            trainer.visualize(batch, it=it, vis_type='image', cameras=cameras, lights=lights)
            if isinstance(trainer.model, PointModel):
                trainer.visualize(
                    batch, it=it, vis_type='pointcloud', cameras=cameras, lights=lights)
            if isinstance(trainer.model, ImplicitModel):
                trainer.visualize(
                    batch, it=it, vis_type='mesh', cameras=cameras, lights=lights)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            logger_py.info('[Epoch %02d] it=%03d, loss=%.4f, time=%.4f'
                           % (epoch_it, it, loss, time.time() - t0b))
            t0b = time.time()

        # Debug visualization
        if it > 0 and debug_every > 0 and (it % debug_every) == 0:
            logger_py.info('Visualizing gradients')
            trainer.debug(batch, cameras=cameras, lights=lights, it=it,
                          mesh_gt=train_dataset.get_meshes())
        # Save checkpoint
        if it > 0 and (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            logger_py.info('Saving checkpoint')
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            trainer.save_shape(os.path.join(out_dir, 'shape'), it)

        # Backup if necessary
        if it > 0 and (backup_every > 0 and (it % backup_every) == 0):
            logger_py.info('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            trainer.save_shape(os.path.join(out_dir, 'shape_%d' % it), it)

        # Run validation and adjust sampling rate
        if it > 0 and validate_every > 0 and (it % validate_every) == 0:
            mesh_gt = train_dataset.get_meshes()
            if mesh_gt is not None:
                eval_dict = trainer.evaluate_mesh(val_loader, it, cameras=cameras, lights=lights)
                metric_val = eval_dict['chamfer']
            else:
                eval_dict = trainer.evaluate(val_loader, cameras=cameras, lights=lights)
                metric_val = eval_dict[model_selection_metric]

            logger_py.info('Validation metric (%s): %.4g' % (model_selection_metric, metric_val))

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                logger_py.info('New best model (loss %.4g)' % metric_val_best)
                checkpoint_io.backup_model_best('model_best.pt')
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)
                trainer.save_shape(os.path.join(out_dir, 'shape_best'), it)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            logger_py.info('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            # create animation
            try:
                os.makedirs(trainer.vis_dir, exist_ok=True)
                create_animation(trainer.vis_dir, show_max=20)
            except Exception as e:
                logger_py.warning(
                    "Couldn't create animated sequence: {}".format(e))
            for t in trainer._threads:
                t.join()
            exit(3)

    # Make scheduler step after full epoch
    trainer.update_learning_rate(it)