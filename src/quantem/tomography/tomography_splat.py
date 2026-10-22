"""Gaussian-splat tomography: explicit atoms (isotropic 3D Gaussians) fitted to a tilt series
jointly with the poses, optionally regularised by a learned atomic prior.

The loop renders whole tilt images with :func:`quantem.core.ml.splat.render_projections`
(``ObjectGaussianSplat.forward``), compares them with the measured stack (``TomographySplatDataset``),
adds the object and dataset soft constraints (including the prior term
``lam * mean ||x - stopgrad(x_hat)||^2``) and steps one optimizer per model. No DDP: a reconstruction
runs on one device (CPU or a single GPU).
"""

from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Self

import numpy as np
import torch
from tqdm.auto import tqdm

from quantem.core.ml.loss_functions import get_loss_module
from quantem.core.ml.optimizer_mixin import OptimizerParams
from quantem.tomography.dataset_models import (
    DatasetConstraintParams,
    DatasetConstraintsType,
    TomographySplatDataset,
)
from quantem.tomography.logger_tomography import LoggerTomography, LoggerTomographySplat
from quantem.tomography.object_models import (
    ObjConstraintParams,
    ObjConstraintsType,
    ObjectGaussianSplat,
)
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.tomography_context import ReconstructionContext
from quantem.tomography.tomography_opt import TomographyOpt

__all__ = ["AtomPrior", "PriorSchedule", "DensifyParams", "TomographySplat"]


class AtomPrior(Protocol):
    """Anything that maps current positions to a clean-structure estimate.

    ``clean_estimate(positions_A, t=...)`` takes ``(N, 3)`` positions in Angstrom (specimen frame)
    and returns ``(N, 3)`` positions in the same frame; ``t`` in ``[0, 1]`` is the flow time
    (1 = clean). Implementations: :class:`quantem.core.ml.splat.atom_prior.AtomPrior` (learned),
    any callable object with that method (e.g. an analytic lattice snap).
    """

    def clean_estimate(
        self, positions_A: torch.Tensor, t: float | None = None
    ) -> torch.Tensor: ...


ScheduleValue = float | tuple[float, float] | Callable[[int, int], float]


def _schedule_value(spec: ScheduleValue, it: int, num_iter: int, anneal_frac: float) -> float:
    """Constant, linear ``(start, end)`` over the first ``anneal_frac`` of the run, or callable."""
    if callable(spec):
        return float(spec(it, num_iter))
    if isinstance(spec, tuple):
        a, b = spec
        n = max(1, int(round(anneal_frac * num_iter)))
        f = min(1.0, it / n) if n > 0 else 1.0
        return float(a + (b - a) * f)
    return float(spec)


@dataclass
class PriorSchedule:
    """When and how strongly the prior is applied.

    Attributes
    ----------
    every : int
        Refresh the prior target every ``every`` iterations (epochs).
    start : int
        First iteration at which the prior is applied.
    t : ScheduleValue or None
        Flow time passed to the prior; a ``(t0, t1)`` tuple anneals linearly over the first
        ``anneal_frac`` of the run. Ignored when ``sigma`` is given and the prior exposes
        ``t_of_sigma``.
    sigma : ScheduleValue or None
        Noise level in Angstrom (converted with ``prior.t_of_sigma``); ``(s0, s1)`` anneals.
    lam : ScheduleValue
        Prior weight. With ``lam_mode="relative"`` it multiplies a scale fixed at the first
        application so that ``||grad L_prior|| = lam * ||grad L_data||`` on the positions.
    lam_mode : {"relative", "absolute"}
    anneal_frac : float
        Fraction of ``num_iter`` over which tuple schedules anneal (then held).
    """

    every: int = 10
    start: int = 0
    t: ScheduleValue | None = 0.5
    sigma: ScheduleValue | None = None
    lam: ScheduleValue = 0.3
    lam_mode: Literal["relative", "absolute"] = "relative"
    anneal_frac: float = 0.6

    def at(
        self, it: int, num_iter: int, prior: AtomPrior | None = None
    ) -> tuple[float | None, float]:
        """``(t, lam_raw)`` for iteration ``it`` (``lam_raw`` before the relative scaling)."""
        lam = _schedule_value(self.lam, it, num_iter, self.anneal_frac)
        if self.sigma is not None and prior is not None and hasattr(prior, "t_of_sigma"):
            s = _schedule_value(self.sigma, it, num_iter, self.anneal_frac)
            return float(prior.t_of_sigma(s)), lam  # type: ignore[attr-defined]
        if self.t is None:
            return None, lam
        return _schedule_value(self.t, it, num_iter, self.anneal_frac), lam


@dataclass
class DensifyParams:
    """Prune / merge / add atoms every ``every`` iterations (0 disables)."""

    every: int = 0
    start: int = 50
    stop: int = 10**9
    prune_amplitude_frac: float = 0.1  # prune atoms with A < frac * median(A)
    merge_distance: float | None = None  # Angstrom
    add_from_residual: bool = False
    add_threshold_fraction: float = 0.999
    add_min_distance: float = 2.0  # Angstrom from existing atoms
    add_min_rel_peak: float = 0.3  # residual back-projection peak / single-atom peak needed to add
    prune_outside_box: bool = True  # drop atoms that left the field-of-view box
    add_blur_sigmas: tuple[float, float] = (1.0, 2.0)  # voxels
    max_add: int | None = None
    voxel_size: float | None = None  # Angstrom; default 2 * sampling


class TomographySplat(TomographyOpt, TomographyBase):
    """
    Reconstruction driver for :class:`ObjectGaussianSplat` + :class:`TomographySplatDataset`.
    """

    @classmethod
    def from_models(
        cls,
        dset: TomographySplatDataset,
        obj_model: ObjectGaussianSplat,
        logger: LoggerTomography | None = None,
        device: str = "cuda",
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        if not isinstance(dset, TomographySplatDataset):
            raise TypeError(f"dset must be a TomographySplatDataset, got {type(dset)}")
        if not isinstance(obj_model, ObjectGaussianSplat):
            raise TypeError(f"obj_model must be an ObjectGaussianSplat, got {type(obj_model)}")
        self = cls(
            dset=dset,
            obj_model=obj_model,
            logger=logger,
            device=device,
            rng=rng,
            verbose=verbose,
            _token=cls._token,
        )
        # No DDP for the splat path: single device, honour "cpu".
        self.world_size, self.global_rank, self.local_rank = 1, 0, 0
        self.device = torch.device(device)
        self.obj_model.to(self.device)
        self.dset.to(self.device)
        self._num_atoms_history: list[int] = []
        self._prior_history: list[dict[str, float]] = []
        self._snapshots: list[dict] = []
        self._lam_scale: float | None = None
        return self

    # region --- properties ---
    @property
    def obj_model(self) -> ObjectGaussianSplat:  # type: ignore[override]
        return self._obj_model  # type: ignore[return-value]

    @obj_model.setter
    def obj_model(self, obj_model: ObjectGaussianSplat):
        self._obj_model = obj_model

    @property
    def dset(self) -> TomographySplatDataset:  # type: ignore[override]
        return self._dset  # type: ignore[return-value]

    @dset.setter
    def dset(self, new_dset: TomographySplatDataset):
        self._dset = new_dset

    @property
    def num_atoms_history(self) -> np.ndarray:
        return np.array(self._num_atoms_history)

    @property
    def prior_history(self) -> list[dict[str, float]]:
        return self._prior_history

    @property
    def snapshots(self) -> list[dict]:
        return self._snapshots

    # endregion

    # region --- setup helpers ---
    def _default_optimizer_params(self, learn_pose: bool) -> dict:
        obj = {
            k: OptimizerParams.Adam(lr=self.obj_model.DEFAULT_LRS.get(k, 1e-2))
            for k in self.obj_model.get_optimization_parameters()
        }
        params: dict = {"object": obj}
        if learn_pose:
            params["pose"] = OptimizerParams.Adam(lr=self.dset.DEFAULT_LRS.get("pose", 5e-2))
        return params

    def _positions_grad_norm_data(
        self, loss_func, image_shape, sampling, window_sigmas, tilt_chunk
    ) -> float:
        """||d L_data / d positions|| over all tilts (for the relative prior weight)."""
        target, rot, shifts = self.dset.batch(None)
        pred = self.obj_model.forward(
            rot, shifts, image_shape, sampling, None, window_sigmas, tilt_chunk
        )
        data_loss = loss_func(pred, target)
        (g,) = torch.autograd.grad(data_loss, self.obj_model._positions, allow_unused=True)
        return 0.0 if g is None else float(g.norm())

    def _snapshot(self, it: int) -> dict:
        shifts, z1, z3, dtilt = (v.detach().cpu().numpy() for v in self.dset.forward(None))
        d = self.obj_model.sites_dict()
        d.update({"iteration": it, "shifts": shifts, "z1": z1, "z3": z3, "dtilt": dtilt})
        return d

    @torch.no_grad()
    def _densify(
        self, it: int, params: DensifyParams, image_shape, sampling, window_sigmas, tilt_chunk
    ) -> dict:
        from quantem.core.ml.splat.densify import prune_mask
        from quantem.core.ml.splat.renderer import backproject_images, render_projections
        from quantem.tomography.atom_trace import gaussian_blur3d, seed_peaks

        obj = self.obj_model
        out = {"pruned": 0, "merged": 0, "added": 0}
        if params.prune_amplitude_frac > 0 and obj.num_atoms > 1:
            amp = obj.amplitudes.detach()
            keep = prune_mask(
                amp,
                obj._positions.detach(),
                float(params.prune_amplitude_frac * amp.median()),
                obj.box_half if params.prune_outside_box else None,
            )
            out["pruned"] = obj.prune(keep)
        if params.merge_distance:
            out["merged"] = obj.merge(float(params.merge_distance))
        if params.add_from_residual:
            target, rot, shifts = self.dset.batch(None)
            pred = obj.forward(rot, shifts, image_shape, sampling, None, window_sigmas, tilt_chunk)
            resid = torch.clamp(target - pred, min=0.0)
            v = float(params.voxel_size) if params.voxel_size else 2.0 * sampling
            D, H, W = obj.shape
            vol_shape = (
                max(2, int(round(D * obj._view_sampling / v))),
                max(2, int(round(H * obj._view_sampling / v))),
                max(2, int(round(W * obj._view_sampling / v))),
            )
            vol = backproject_images(resid, rot, shifts, sampling, vol_shape, v)
            # reference: back-projected peak of one median atom at the origin, same blur as the seeds
            one = render_projections(
                torch.zeros(1, 3, device=obj.device, dtype=obj.dtype),
                obj.amplitudes.detach().median().reshape(1),
                obj.sigmas.detach().median().reshape(1),
                rot,
                shifts,
                sampling,
                image_shape,
                obj.window_radius(sampling, window_sigmas),
            )
            ref_vol = backproject_images(one, rot, shifts, sampling, vol_shape, v).cpu()
            bs = params.add_blur_sigmas
            if isinstance(bs, (tuple, list)):
                ref_resp = gaussian_blur3d(ref_vol, float(bs[0])) - gaussian_blur3d(
                    ref_vol, float(bs[1])
                )
            else:
                ref_resp = gaussian_blur3d(ref_vol, float(bs))
            thr = (
                params.add_min_rel_peak * float(ref_resp.max())
                if params.add_min_rel_peak > 0
                else None
            )
            peaks, _ = seed_peaks(
                vol.cpu(),
                blur_sigmas=params.add_blur_sigmas,
                threshold=thr,
                threshold_fraction=params.add_threshold_fraction,
                min_distance=params.add_min_distance / v,
                max_peaks=params.max_add,
            )
            if peaks.shape[0]:
                pos = torch.stack(
                    (
                        (peaks[:, 2].double() - (vol_shape[2] - 1) / 2) * v,
                        (peaks[:, 1].double() - (vol_shape[1] - 1) / 2) * v,
                        (peaks[:, 0].double() - (vol_shape[0] - 1) / 2) * v,
                    ),
                    dim=1,
                ).float()
                if obj.num_atoms:
                    d = torch.cdist(pos, obj._positions.detach().cpu()).min(dim=1).values
                    pos = pos[d > params.add_min_distance]
                out["added"] = obj.add_atoms(pos)
        return out

    # endregion

    def reconstruct(
        self,
        num_iter: int = 100,
        batch_size: int | None = None,
        learn_pose: bool = True,
        reset: bool = False,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        obj_constraints: dict | ObjConstraintsType | None = None,
        dset_constraints: dict | DatasetConstraintsType | None = None,
        loss_type: Literal["l2", "l1", "smooth_l1", "charbonnier"] = "l2",
        loss_func_kwargs: dict = {},
        grad_clip: float | None = None,
        window_sigmas: float = 4.0,
        tilt_chunk: int | None = None,
        prior: AtomPrior | None = None,
        prior_schedule: PriorSchedule | None = None,
        densify: DensifyParams | None = None,
        neighbor_every: int = 20,
        store_snapshots_every: int | None = None,
        show_metrics: bool = False,
        schedule_total: int | None = None,
    ) -> Self:
        """
        Fit the atoms (and poses) to the tilt series.

        Parameters
        ----------
        num_iter : int
            Iterations (one iteration = one pass over all tilts).
        batch_size : int or None
            Tilts rendered per optimizer step; ``None`` renders all tilts in one step.
        learn_pose : bool
            Optimize the dataset pose parameters (shifts, z1/z3, dtilt as enabled on the dataset).
        prior, prior_schedule
            A clean-structure estimator and its schedule; the target is refreshed every
            ``prior_schedule.every`` iterations and enters the loss through the object model.
        densify : DensifyParams or None
            Prune / merge / add atoms on a schedule.
        schedule_total : int or None
            Total planned iterations for the prior schedule when ``reconstruct`` is called in
            chunks (defaults to ``num_iter``); schedule iteration counts are global (``num_epochs``).
        """
        if reset:
            self.obj_model.reset()
        if optimizer_params is None and not self.obj_model.has_optimizer():
            optimizer_params = self._default_optimizer_params(learn_pose)
        new_scheduler = False
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
            self.set_optimizers()
            new_scheduler = True
        if not learn_pose:
            self.remove_optimizer("pose")
        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            new_scheduler = True
        if new_scheduler:
            self.set_schedulers(self.scheduler_params, num_iter=num_iter)
        if obj_constraints is not None:
            if isinstance(obj_constraints, dict):
                obj_constraints = ObjConstraintParams.parse_dict(obj_constraints)
            self.obj_model.constraints = obj_constraints
        if dset_constraints is not None:
            if isinstance(dset_constraints, dict):
                dset_constraints = DatasetConstraintParams.parse_dict(dset_constraints)
            self.dset.constraints = dset_constraints
        if prior is not None and prior_schedule is None:
            prior_schedule = PriorSchedule()

        loss_func = get_loss_module(name=loss_type, dtype=self.obj_model.dtype, **loss_func_kwargs)
        image_shape = self.dset.image_shape
        sampling = self.dset.sampling
        T = self.dset.num_tilts
        bs = T if batch_size is None else max(1, min(int(batch_size), T))
        gen = torch.Generator(device="cpu")
        if getattr(self, "_rng_seed", None) is not None:
            gen.manual_seed(int(self._rng_seed) % 2**32)
        else:
            gen.seed()
        if self.obj_model.constraints.repulsion > 0:
            self.obj_model.refresh_neighbors()

        pbar = tqdm(range(num_iter), disable=not self.verbose)
        for a0 in pbar:
            it = self.num_epochs
            total_loss = torch.zeros((), device=self.device)
            data_loss_sum = torch.zeros((), device=self.device)
            soft_loss_sum = torch.zeros((), device=self.device)
            perm = torch.randperm(T, generator=gen) if bs < T else torch.arange(T)
            n_batches = 0
            for b0 in range(0, T, bs):
                idx = perm[b0 : b0 + bs].to(self.device)
                tilt_idx = None if bs == T else idx
                self.zero_grad_all()
                target, rot, shifts = self.dset.batch(tilt_idx)
                pred = self.obj_model.forward(
                    rot, shifts, image_shape, sampling, tilt_idx, window_sigmas, tilt_chunk
                )
                data_loss = loss_func(pred, target)
                soft = self.obj_model.apply_soft_constraints(ReconstructionContext(pred=pred))
                soft = soft + self.dset.apply_soft_constraints()
                loss = data_loss + soft
                loss.backward()
                if grad_clip is not None:
                    params = list(self.obj_model.params) + [p for p in self.dset.parameters()]
                    torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))
                self.step_optimizers()
                self.obj_model.apply_hard_constraints()
                total_loss += loss.detach()
                data_loss_sum += data_loss.detach()
                soft_loss_sum += soft.detach()
                n_batches += 1

            total = float(total_loss) / n_batches
            data_mean = float(data_loss_sum) / n_batches
            soft_mean = float(soft_loss_sum) / n_batches
            self.step_schedulers(loss=total)
            self._epoch_losses.append(total)
            self._consistency_losses.append(data_mean)
            self.obj_model._soft_constraint_losses.append(soft_mean)
            self.append_learning_rates(self.get_current_lrs())
            self._num_atoms_history.append(self.obj_model.num_atoms)
            self.obj_model.accumulate_constraint_losses()

            # --- neighbour cache for the repulsion term ---
            if (
                self.obj_model.constraints.repulsion > 0
                and neighbor_every > 0
                and (it + 1) % neighbor_every == 0
            ):
                self.obj_model.refresh_neighbors()

            # --- densification ---
            if (
                densify is not None
                and densify.every > 0
                and densify.start <= it < densify.stop
                and (it + 1) % densify.every == 0
            ):
                changes = self._densify(
                    it, densify, image_shape, sampling, window_sigmas, tilt_chunk
                )
                if any(changes.values()):
                    self.set_optimizers() if self.obj_model.optimizer is None else None
                    if self.obj_model.constraints.repulsion > 0:
                        self.obj_model.refresh_neighbors()

            # --- prior target refresh ---
            if (
                prior is not None
                and prior_schedule is not None
                and it >= prior_schedule.start
                and ((it - prior_schedule.start) % max(1, prior_schedule.every) == 0)
            ):
                t, lam_raw = prior_schedule.at(it, schedule_total or num_iter, prior)
                x = self.obj_model._positions.detach()
                x_hat = prior.clean_estimate(x, t=t).to(x)
                pull = (x_hat - x).norm(dim=-1)
                if prior_schedule.lam_mode == "relative":
                    if self._lam_scale is None:
                        g_data = self._positions_grad_norm_data(
                            loss_func, image_shape, sampling, window_sigmas, tilt_chunk
                        )
                        g_prior_unit = 2.0 * float(pull.norm()) / max(x.shape[0], 1)
                        self._lam_scale = g_data / g_prior_unit if g_prior_unit > 0 else 1.0
                    lam = lam_raw * self._lam_scale
                else:
                    lam = lam_raw
                self.obj_model.set_prior_target(x_hat, lam)
                self._prior_history.append(
                    {
                        "iteration": it,
                        "t": -1.0 if t is None else float(t),
                        "lam": float(lam),
                        "pull_mean": float(pull.mean()),
                        "pull_max": float(pull.max()),
                        "pull_frac_gt_0p5": float((pull > 0.5).float().mean()),
                    }
                )

            if store_snapshots_every and (it + 1) % store_snapshots_every == 0:
                self._snapshots.append(self._snapshot(it))

            desc = f"Splat | loss {total:.4e} data {data_mean:.4e} soft {soft_mean:.3e} atoms {self.obj_model.num_atoms}"
            pbar.set_description(desc)
            if self.logger is not None:
                if isinstance(self.logger, LoggerTomographySplat):
                    self.logger.log_iter_splat(self, it, total, data_mean, soft_mean)
                    if (
                        self.logger.log_images_every > 0
                        and (it + 1) % self.logger.log_images_every == 0
                    ):
                        with torch.no_grad():
                            target, rot, shifts = self.dset.batch(None)
                            pred = self.obj_model.forward(
                                rot, shifts, image_shape, sampling, None, window_sigmas, tilt_chunk
                            )
                        self.logger.log_iter_images_splat(
                            pred.detach(), target.detach(), self.dset, self.obj_model, it
                        )
                self.logger.flush()
        if show_metrics:
            self.plot_losses()
        return self

    def plot_losses(self):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
        axs[0].semilogy(self.epoch_losses, label="total")
        axs[0].semilogy(self.consistency_losses, label="data")
        axs[0].set_xlabel("iteration")
        axs[0].set_ylabel("loss")
        axs[0].legend()
        axs[0].set_title("Loss")
        axs[1].plot(self.num_atoms_history)
        axs[1].set_xlabel("iteration")
        axs[1].set_ylabel("atoms")
        axs[1].set_title("Atom count")
        fig.tight_layout()
        return fig

    # region --- export ---
    def render_all(
        self, window_sigmas: float = 4.0, tilt_chunk: int | None = None
    ) -> torch.Tensor:
        """Rendered stack ``(T, H, W)`` at the current atoms and poses (no grad)."""
        with torch.no_grad():
            target, rot, shifts = self.dset.batch(None)
            return self.obj_model.forward(
                rot,
                shifts,
                self.dset.image_shape,
                self.dset.sampling,
                None,
                window_sigmas,
                tilt_chunk,
            )

    def save_atoms(self, path: str) -> None:
        """``np.savez`` of positions / amplitudes / sigmas / background / poses / histories."""
        snap = self._snapshot(self.num_epochs)
        np.savez(
            path,
            **{k: v for k, v in snap.items() if v is not None},
            epoch_losses=self.epoch_losses,
            consistency_losses=self.consistency_losses,
            num_atoms=self.num_atoms_history,
            sampling=self.dset.sampling,
            intensity_scale=self.dset.intensity_scale,
            tilt_angles=self.dset.tilt_angles.detach().cpu().numpy(),
        )

    # endregion
