import os
from typing import Literal, Self

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.ml.inr import HSiren
from quantem.core.ml.optimizer_mixin import (
    OptimizerParams,
    SchedulerParams,
)
from quantem.tomography.dataset_models import (
    DatasetConstraintsType,
    TomographyINRDataset,
    TomographyPixDataset,
    TomographySplatDataset,
)
from quantem.tomography.logger_tomography import LoggerTomography, LoggerTomographySplat
from quantem.tomography.object_models import (
    ObjConstraintsType,
    ObjectGaussianSplat,
    ObjectINR,
    ObjectPixelated,
)
from quantem.tomography.tomography import Tomography, TomographyConventional
from quantem.tomography.tomography_splat import TomographySplat


class TomographyLiteINR(Tomography):
    """
    A lite version of the Tomography class.
    """

    @classmethod
    def from_dataset(
        cls,
        tilt_series: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        device: str = "cuda",
        log_dir: os.PathLike | str | None = None,
        log_images_every: int = 10,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        dset_model = TomographyINRDataset.from_data(
            tilt_stack=tilt_series,
            tilt_angles=tilt_angles,
        )

        # Define the object model
        model = HSiren(alpha=1, winner_initialization=72)
        obj_model = ObjectINR.from_model(
            model=model,
            shape=(
                max(dset_model.tilt_stack.shape),
                max(dset_model.tilt_stack.shape),
                max(dset_model.tilt_stack.shape),
            ),
            device=device,
            rng=rng,
        )

        # TODO: Implement pretrain

        if log_dir is not None:
            logger = LoggerTomography(
                log_dir=str(log_dir),
                run_prefix="tomography_lite_inr",
                run_suffix="",
                log_images_every=log_images_every,
            )
        else:
            logger = None

        tomography = cls.from_models(
            dset=dset_model,
            obj_model=obj_model,
            device=device,
            rng=rng,
            logger=logger,
        )

        return tomography

    def reconstruct(  # type:ignore[reportIncompatibleMethodOverride] ## easier than overloads
        self,
        num_iter: int = 10,
        reset: bool = False,
        obj_lr: float = 1e-4,
        pose_lr: float = 1e-2,
        batch_size: int = 1024,
        num_workers: int = 32,
        learn_pose: bool = True,
        warmup_routine: bool = True,
        scheduler_type: Literal[
            "exp", "cyclic", "plateau", "cosine_annealing", "linear", "full_warmup", "none"
        ] = "none",
        scheduler_params: dict = {},
        new_optimizers: bool = False,
        obj_constraints: ObjConstraintsType | dict | None = None,
        dset_constraints: DatasetConstraintsType | dict | None = None,
        show_metrics: bool = False,
    ):
        if self.num_epochs == 0:
            opt_params = {
                "object": OptimizerParams.Adam(lr=obj_lr),
            }

            all_scheduler_params = {
                "object": SchedulerParams.parse_dict(
                    {
                        "name": scheduler_type,
                        **scheduler_params,
                    }
                ),
            }

            if learn_pose:
                opt_params["pose"] = OptimizerParams.Adam(lr=pose_lr)
                all_scheduler_params["pose"] = SchedulerParams.parse_dict(
                    {
                        "name": scheduler_type,
                        **scheduler_params,
                    }
                )
        else:
            opt_params = None
            all_scheduler_params = None
            obj_constraints = None
            dset_constraints = None

        num_samples_per_ray = int(max(self.dset.tilt_stack.shape))
        return super().reconstruct(
            num_iter=num_iter,
            batch_size=batch_size,
            num_workers=num_workers,
            reset=reset,
            num_samples_per_ray=num_samples_per_ray,
            optimizer_params=opt_params,
            scheduler_params=all_scheduler_params,
            obj_constraints=obj_constraints,
            dset_constraints=dset_constraints,
            show_metrics=show_metrics,
        )


class TomographyLiteConv(TomographyConventional):
    @classmethod
    def from_dataset(
        cls,
        tilt_series: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        device: str = "cuda",
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        dset_model = TomographyPixDataset.from_data(
            tilt_stack=tilt_series,
            tilt_angles=tilt_angles,
        )

        obj_model = ObjectPixelated.from_uniform(
            shape=(
                max(tilt_series.shape),
                max(tilt_series.shape),
                max(tilt_series.shape),
            ),
            device=device,
            rng=rng,
        )

        tomography = cls.from_models(
            dset=dset_model,
            obj_model=obj_model,
            device=device,
            rng=rng,
        )

        return tomography


class TomographyLiteSplat(TomographySplat):
    """
    One-call Gaussian-splat tomography: build the dataset and the atoms, then ``reconstruct``.
    """

    @classmethod
    def from_dataset(  # type: ignore[override]
        cls,
        tilt_series: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        sampling: float,
        init: Literal["backproject"] | NDArray | torch.Tensor | str | os.PathLike = "backproject",
        sigma_init: float = 0.38,
        amplitude_init: float | None = None,
        learn_tilt_offset: bool = False,
        voxel_size: float | None = None,
        fov_z_A: float | None = None,
        device: str = "cuda",
        log_dir: os.PathLike | str | None = None,
        log_images_every: int = 10,
        rng: np.random.Generator | int | None = None,
        **seed_kwargs,
    ) -> Self:
        dset = TomographySplatDataset.from_data(
            tilt_stack=tilt_series,
            tilt_angles=tilt_angles,
            sampling=sampling,
            learn_tilt_offset=learn_tilt_offset,
        )
        dset.to("cpu")
        H, W = dset.image_shape
        fov = (
            W * sampling,
            H * sampling,
            float(fov_z_A) if fov_z_A is not None else min(W, H) * sampling,
        )
        if isinstance(init, str) and init == "backproject":
            obj = ObjectGaussianSplat.from_dataset(
                dset,
                sigma=sigma_init,
                amplitude=amplitude_init,
                voxel_size=voxel_size,
                fov_z_A=fov_z_A,
                **seed_kwargs,
            )
        else:
            if amplitude_init is None:
                stack = dset.tilt_stack
                n = (
                    len(np.loadtxt(init, skiprows=2, usecols=(1,)))
                    if isinstance(init, (str, os.PathLike))
                    else len(init)
                )
                amplitude_init = float(stack.sum(dim=(1, 2)).mean()) * sampling**2 / max(n, 1)
            if isinstance(init, (str, os.PathLike)):
                obj = ObjectGaussianSplat.from_xyz(
                    str(init), amplitude_init, sigma_init, fov_A=fov, num_tilts=dset.num_tilts
                )
            else:
                obj = ObjectGaussianSplat.from_positions(
                    init, amplitude_init, sigma_init, fov_A=fov, num_tilts=dset.num_tilts
                )
        logger = None
        if log_dir is not None:
            logger = LoggerTomographySplat(
                log_dir=str(log_dir), run_prefix="splat", log_images_every=log_images_every
            )
        return cls.from_models(dset=dset, obj_model=obj, logger=logger, device=device, rng=rng)

    def reconstruct(  # type: ignore[override]
        self,
        num_iter: int = 100,
        position_lr: float = 2e-2,
        amplitude_lr: float = 1e-2,
        sigma_lr: float = 5e-3,
        background_lr: float = 1e-2,
        pose_lr: float = 5e-2,
        learn_pose: bool = True,
        batch_size: int | None = None,
        scheduler_type: Literal["none", "cosine_annealing", "exponential"] = "none",
        scheduler_params: dict = {},
        **kwargs,
    ) -> Self:
        if not self.obj_model.has_optimizer():
            lrs = {
                "positions": position_lr,
                "amplitudes": amplitude_lr,
                "sigmas": sigma_lr,
                "background": background_lr,
            }
            obj = {
                k: OptimizerParams.Adam(lr=lrs[k])
                for k in self.obj_model.get_optimization_parameters()
            }
            kwargs["optimizer_params"] = {"object": obj, "pose": OptimizerParams.Adam(lr=pose_lr)}
            if scheduler_type != "none":
                sp = {"name": scheduler_type, **scheduler_params}
                kwargs["scheduler_params"] = {"object": sp, "pose": sp}
        return super().reconstruct(
            num_iter=num_iter, batch_size=batch_size, learn_pose=learn_pose, **kwargs
        )
