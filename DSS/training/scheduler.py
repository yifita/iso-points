"""
change trainer settings according to iterations
"""
from typing import List
import bisect
from .. import logger_py
from ..models.levelset_sampling import LevelSetProjection


class TrainerScheduler(object):
    """ Increase n_points_per_cloud and Reduce n_training_points """

    def __init__(self, init_n_points_dss: int = 0, init_n_rays: int = 0,
                 init_proj_tolerance: float = 0,
                 init_sdf_alpha: float=1.0,
                 init_lambda_occupied: float=5.0, init_lambda_freespace: float=5.0,
                 steps_n_points_dss: int = -1, steps_n_rays: int= -1,
                 steps_proj_tolerance: int = -1, steps_lambda_rgb: int = -1,
                 steps_sdf_alpha:int = -1,
                 steps_lambda_sdf: int = -1,
                 init_lambda_rgb: float = 1.0, warm_up_iters: int = None,
                 gamma_n_points_dss: float = 2.0, gamma_n_rays: float = 0.5,
                 gamma_lambda_rgb: float = 1.0,
                 gamma_sdf_alpha: float = 1.0,
                 gamma_lambda_sdf: float = 1.0,
                 limit_n_points_dss: int = 1e5, limit_n_rays: int = 0,
                 limit_lambda_rgb: float = 1.0,
                 limit_sdf_alpha: float=100,
                 limit_lambda_freespace: float=1.0, limit_lambda_occupied=1.0,
                 gamma_proj_tolerance: float = 0.1, limit_proj_tolerance: float = 5e-5,
                 ):
        """ steps_n_points_dss: list  """

        self.init_n_points_dss = init_n_points_dss
        self.init_n_rays = init_n_rays
        self.init_proj_tolerance = init_proj_tolerance
        self.init_sdf_alpha = init_sdf_alpha
        self.init_lambda_rgb = init_lambda_rgb
        self.init_lambda_freespace = init_lambda_freespace
        self.init_lambda_occupied = init_lambda_occupied

        self.steps_n_points_dss = steps_n_points_dss
        self.steps_n_rays = steps_n_rays
        self.steps_proj_tolerance = steps_proj_tolerance
        self.steps_lambda_rgb = steps_lambda_rgb
        self.steps_sdf_alpha = steps_sdf_alpha
        self.steps_lambda_sdf = steps_lambda_sdf

        self.gamma_n_points_dss = gamma_n_points_dss
        self.gamma_n_rays = gamma_n_rays
        self.gamma_proj_tolerance = gamma_proj_tolerance
        self.gamma_lambda_rgb = gamma_lambda_rgb
        self.gamma_sdf_alpha = gamma_sdf_alpha
        self.gamma_lambda_sdf = gamma_lambda_sdf

        self.limit_n_points_dss = limit_n_points_dss
        self.limit_n_rays = limit_n_rays
        self.limit_proj_tolerance = limit_proj_tolerance
        self.limit_sdf_alpha = limit_sdf_alpha
        self.limit_lambda_freespace = limit_lambda_freespace
        self.limit_lambda_occupied = limit_lambda_occupied
        self.limit_lambda_rgb = limit_lambda_rgb

        self.warm_up_iters = warm_up_iters

    def step(self, trainer, it):
        if it < 0:
            return

        if self.steps_n_points_dss > 0 and hasattr(trainer.model, 'n_points_per_cloud'):
            i = it // self.steps_n_points_dss
            gamma = self.gamma_n_points_dss ** i
            old_n_points_per_cloud = trainer.model.n_points_per_cloud
            trainer.model.n_points_per_cloud = min(
                int(self.init_n_points_dss * gamma), self.limit_n_points_dss)
            if old_n_points_per_cloud != trainer.model.n_points_per_cloud:
                logger_py.info('Updated n_points_per_cloud: {} -> {}'.format(
                    old_n_points_per_cloud, trainer.model.n_points_per_cloud))

        if self.steps_n_rays > 0:
            # reduce n_rays gradually
            i = it // self.steps_n_rays
            gamma = self.gamma_n_rays ** i
            old_n_rays = trainer.n_training_points
            if self.gamma_n_rays < 1:
                trainer.n_training_points = max(
                    int(self.init_n_rays * gamma), self.limit_n_rays)
            else:
                trainer.n_training_points = min(
                    int(self.init_n_rays * gamma), self.limit_n_rays)
            if old_n_rays != trainer.n_training_points:
                logger_py.info('Updated n_training_points: {} -> {}'.format(
                    old_n_rays, trainer.n_training_points))

        # adjust projection tolerance and proj_max_iters
        if self.steps_proj_tolerance > 0 and it % self.steps_proj_tolerance == 0:
            if hasattr(trainer.model, 'projection') and isinstance(trainer.model.projection, LevelSetProjection):
                old_proj_tol = trainer.model.projection.proj_tolerance
                i = it // self.steps_proj_tolerance
                gamma = self.gamma_proj_tolerance ** i
                trainer.model.projection.proj_tolerance = max(self.init_proj_tolerance*gamma, self.limit_proj_tolerance)
                if old_proj_tol != trainer.model.projection.proj_tolerance:
                    logger_py.info('Updated projection.proj_tolerance: {} -> {}'.format(
                        old_proj_tol, trainer.model.projection.proj_tolerance))
                    trainer.model.projection.proj_max_iters = min(trainer.model.projection.proj_max_iters *2, 50)
                if hasattr(trainer.model, 'sphere_tracing') and isinstance(trainer.model.sphere_tracing, LevelSetProjection):
                    old_proj_tol = trainer.model.sphere_tracing.proj_tolerance
                    trainer.model.sphere_tracing.proj_tolerance = max(self.init_proj_tolerance*gamma, self.limit_proj_tolerance)
                    if old_proj_tol != trainer.model.sphere_tracing.proj_tolerance:
                        logger_py.info('Updated sphere_tracing.proj_tolerance: {} -> {}'.format(
                            old_proj_tol, trainer.model.sphere_tracing.proj_tolerance))
                        trainer.model.sphere_tracing.proj_max_iters = min(trainer.model.sphere_tracing.proj_max_iters *2, 50)


        # increase lambda_rgb slowly
        if self.steps_lambda_rgb > 0:
            # also change the init_lambda_rgb
            old_lambda_rgb = trainer.lambda_rgb
            trainer.lambda_rgb = self.init_lambda_rgb * self.gamma_lambda_rgb ** (
                it // self.steps_lambda_rgb)
            trainer.lambda_rgb = min(trainer.lambda_rgb, self.limit_lambda_rgb)
            if old_lambda_rgb != trainer.lambda_rgb:
                logger_py.info('Updated lambda_rgb: {} -> {}'.format(
                    old_lambda_rgb, trainer.lambda_rgb))

        # update init_lambda_occupied and init_lambda_freespace
        if self.steps_lambda_sdf > 0:
            old_lambda = trainer.lambda_freespace
            scale = self.gamma_lambda_sdf ** (it // self.steps_lambda_sdf)
            trainer.lambda_freespace = self.init_lambda_freespace * scale
            if self.gamma_lambda_sdf < 1.0 and self.limit_lambda_freespace < self.init_lambda_freespace:
                trainer.lambda_freespace = max(trainer.lambda_freespace, self.limit_lambda_freespace)
            else:
                trainer.lambda_freespace = min(trainer.lambda_freespace, self.limit_lambda_freespace)

            if old_lambda != trainer.lambda_freespace:
                logger_py.info('Updated lambda_freespace: {} -> {}'.format(
                    old_lambda, trainer.lambda_freespace))

            old_lambda = trainer.lambda_occupied
            scale = self.gamma_lambda_sdf ** (it // self.steps_lambda_sdf)
            trainer.lambda_occupied = self.init_lambda_occupied * scale
            if self.gamma_lambda_sdf < 1.0 and self.limit_lambda_occupied < self.init_lambda_occupied:
                trainer.lambda_occupied = max(trainer.lambda_occupied, self.limit_lambda_occupied)
            else:
                trainer.lambda_occupied = min(trainer.lambda_occupied, self.limit_lambda_occupied)

            if old_lambda != trainer.lambda_occupied:
                logger_py.info('Updated lambda_occupied: {} -> {}'.format(
                    old_lambda, trainer.lambda_occupied))

        if self.steps_sdf_alpha > 0:
            # change sdf loss weight gradually
            i = it // self.steps_sdf_alpha
            gamma = self.gamma_sdf_alpha ** i
            old_alpha = trainer.sdf_alpha
            if self.gamma_sdf_alpha < 1:
                trainer.sdf_alpha = max(
                    int(self.init_sdf_alpha * gamma), self.limit_sdf_alpha)
            else:
                trainer.sdf_alpha = min(
                    int(self.init_sdf_alpha * gamma), self.limit_sdf_alpha)
            if old_alpha != trainer.sdf_alpha:
                logger_py.info('Updated sdf_alpha: {} -> {}'.format(
                    old_alpha, trainer.sdf_alpha))
