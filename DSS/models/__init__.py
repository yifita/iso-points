import torch
from collections import namedtuple

__all__ = ['BaseGenerator', 'ImplicitModel', 'OccupancyModel',
           'PointModel', 'CombinedModel', 'ModelReturns']


class BaseGenerator(object):
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def generate_meshes(self, *args, **kwargs):
        return []

    def generate_pointclouds(self, *args, **kwargs):
        return []

    def generate_images(self, *args, **kwargs):
        return []


ModelReturns = namedtuple(
    'ModelReturns', 'pointclouds mask_pred sdf_freespace sdf_occupancy img_pred mask_img_pred')

from .implicit_modeling import Model as ImplicitModel
from .point_modeling import Model as PointModel
from .combined_modeling import Model as CombinedModel
