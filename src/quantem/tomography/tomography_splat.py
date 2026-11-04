"""Gaussian-splat tomography: explicit atoms (isotropic 3D Gaussians) fitted to a tilt series
jointly with the poses, optionally regularised by a learned atomic prior.

The loop renders whole tilt images with :func:`quantem.core.ml.splat.render_projections`
(``ObjectGaussianSplat.forward``), compares them with the measured stack (``TomographySplatDataset``),
adds the object and dataset soft constraints (including the prior term
``lam * mean ||x - stopgrad(x_hat)||^2``) and steps one optimizer per model. No DDP: a reconstruction
runs on one device (CPU or a single GPU).
"""

import math
import time
import warnings
from dataclasses import dataclass, replace
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
    lam_mode : {"relative", "relative_live", "absolute"}
    anneal_frac : float
        Fraction of ``num_iter`` over which tuple schedules anneal (then held).
    lam_fade_start : float or None
        If given, a tuple ``lam = (a, b)`` is held at ``a`` until this fraction of the run and then
        anneals linearly to ``b`` at the end of the run (a late fade, e.g. ``(lam, 0)`` with 2/3 fades
        the prior out over the last third), instead of annealing over the first ``anneal_frac``
        like every other tuple schedule. ``None`` (the default) keeps the old behaviour.
    sigma_mode : {"clock", "pull"}
        "clock" (default): ``sigma`` follows the iteration clock. "pull": sigma_t is set at every call from the
        previous call's measured pull, ``clamp(sigma_pull_scale * pull_rms_coord, min(sigma), max(sigma))``
        (first call ``max(sigma)``); see ``sigma_at``.
    sigma_pull_scale : float
        The factor ``c`` of the "pull" mode (the pull underestimates the true error of a conservative denoiser).
    """

    every: int = 10
    start: int = 0
    t: ScheduleValue | None = 0.5
    sigma: ScheduleValue | None = None
    lam: ScheduleValue = 0.3
    lam_mode: Literal["relative", "relative_live", "absolute"] = "relative"
    anneal_frac: float = 0.6
    stop: int = (
        10**9
    )  # no prior from this iteration on (data-only fine-tune, "three-stage" schedules)
    mode: Literal["loss", "proximal", "energy"] = (
        "loss"  # "proximal": move the atoms a fraction ``alpha`` of the
    )
    # way to x̂ at every refresh (plug-and-play / RED style) instead of adding a loss term; ``lam`` unused.
    # "energy" (round 5 step 2, 2026-10-29): the prior is an energy (``prior.energy_and_forces(x)``, e.g.
    # :class:`quantem.tomography.energy_prior.EnergyPrior`); ``lam * E(x)`` enters the loss as the constant
    # gradient ``-lam * F(x)`` on the positions, refreshed every ``every`` iterations and held in between
    # (as x̂ is held in "loss" mode). ``lam_mode`` as in "loss" mode: "relative" fixes lam once so that
    # ``||lam * dE/dx|| = lam_rel * ||grad L_data||`` at the first application.
    alpha: float = 0.3
    # noise added to the current positions before each prior call, as a multiple of sigma_t (the
    # training corruption; 1.0 = the behaviour since chapter 01). 0.0 treats the positions as the
    # noisy state x_t themselves (round 4 step 6, 2026-10-28: the fresh noise sets a ~0.03 A floor).
    noise_scale: float = 1.0
    # round 5 step 1 (2026-10-29): late fade of a tuple ``lam`` over [lam_fade_start * num_iter, num_iter]
    lam_fade_start: float | None = None
    # round 6 step 4 (2026-11-02): where sigma_t comes from. "clock" (the default): the tuple ``sigma`` anneals over
    # ``anneal_frac`` of the run like every other tuple schedule. "pull": sigma_t is measured, not scheduled —
    # ``sigma_t = clamp(sigma_pull_scale * e, min(sigma), max(sigma))`` with ``e`` the per-coordinate RMS of the
    # prior's own pull at the PREVIOUS call (``rms over atoms of ||x_hat - x|| / sqrt(3)``, ``pull_rms_coord`` in the
    # history); the first call (no previous pull) uses ``max(sigma)``. Needs a tuple ``sigma``; lam / fade / stop untouched.
    sigma_mode: Literal["clock", "pull"] = "clock"
    sigma_pull_scale: float = 1.0

    def sigma_at(
        self, it: int, num_iter: int, last_pull_rms_coord: float | None = None
    ) -> float | None:
        """The noise level sigma_t (Angstrom) for iteration ``it``; ``None`` when the schedule is given as ``t``."""
        if self.sigma is None:
            return None
        if self.sigma_mode == "clock":
            return _schedule_value(self.sigma, it, num_iter, self.anneal_frac)
        if self.sigma_mode != "pull":
            raise ValueError(f"sigma_mode must be 'clock' or 'pull', got {self.sigma_mode!r}")
        if not isinstance(self.sigma, tuple):
            raise ValueError("sigma_mode='pull' needs sigma=(s_max, s_min): the clamp range of the measured sigma_t")
        lo, hi = min(self.sigma), max(self.sigma)
        if last_pull_rms_coord is None:
            return float(hi)
        return float(min(max(self.sigma_pull_scale * float(last_pull_rms_coord), lo), hi))

    def at(
        self,
        it: int,
        num_iter: int,
        prior: AtomPrior | None = None,
        last_pull_rms_coord: float | None = None,
    ) -> tuple[float | None, float]:
        """``(t, lam_raw)`` for iteration ``it`` (``lam_raw`` before the relative scaling); ``last_pull_rms_coord``
        (the previous call's ``pull_rms_coord``) is used only by ``sigma_mode="pull"``."""
        if self.lam_fade_start is not None and isinstance(self.lam, tuple):
            a, b = self.lam
            n0 = int(round(self.lam_fade_start * num_iter))
            f = 0.0 if it <= n0 else min(1.0, (it - n0) / max(1, num_iter - n0))
            lam = float(a + (b - a) * f)
        else:
            lam = _schedule_value(self.lam, it, num_iter, self.anneal_frac)
        if self.sigma is not None and prior is not None and hasattr(prior, "t_of_sigma"):
            s = self.sigma_at(it, num_iter, last_pull_rms_coord)
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
    split_from_residual: bool = False  # split over-bright atoms toward rejected residual peaks
    split_amplitude_frac: float = 1.5  # an atom is "fat" when A > frac * median(A)
    split_min_offset: float = 0.7  # Angstrom; a residual peak closer than this is the same site
    max_add_frac: float | None = (
        None  # add at most this fraction of the current atom count per pass
    )
    prune_reference: Literal["median", "max_median"] = (
        "median"  # "max_median": the running maximum
    )
    # of the median amplitude over passes, so a flood of weak spurious atoms cannot lower the bar
    add_noise_region: Literal["support", "volume"] = "support"  # where the residual noise is
    # measured: inside the particle (voxels where the back-projected model exceeds 10 % of its
    # maximum, where shot noise is highest and where candidates compete with real atoms) or everywhere
    add_noise_sigmas: float = 0.0  # also require a residual peak > this many robust sigmas of the
    # residual's own filtered noise (MAD over the volume); 0 disables. Needed at low dose, where the
    # single-atom-relative threshold alone admits shot-noise peaks.
    # --- chapter 06 candidates (defaults = the chapter 03 behaviour) ---
    add_signal: Literal["residual", "whitened"] = "whitened"  # "whitened": back-project the residual
    # divided by its expected shot-noise standard deviation (pixel_variance), so the volume's noise is
    # uniform and sites wrong in many views stand out ("multi-view inconsistency"); needs the dose
    add_chi2_min: float = 0.0  # accept a candidate only where the back-projected normalised squared
    # residual (chi^2 per view, 1 = consistent) exceeds this; 0 disables
    remove_whitened_sigmas: float = 0.0  # remove an atom where the filtered whitened residual volume
    # is below -k sigma (the model is too bright there in many views); 0 disables
    remove_snr_min: float = 2.0  # remove atoms whose data support (densify.atom_support, chi^2 the
    # atom explains) is below snr_min^2, i.e. detection SNR below snr_min; 0 disables
    max_remove_frac: float | None = 0.2  # cap of the support / whitened removals per pass
    merge_amplitude: Literal["sum", "max"] = "sum"  # merged atom carries the sum or the larger amplitude
    # --- chapter 06 round 3: gradient-clone add rule (3DGS densification; Kerbl et al. 2023 sec. 5.2, and the
    # Gaussian AET preprint arXiv:2512.15034, which duplicates Gaussians with a large |dL/dr|). 0 disables. ---
    clone_grad_k: float = 0.0  # clone atoms whose mean positional-gradient norm over the densify interval exceeds
    # k x the population median (relative, so the bar does not depend on the dose or the loss scale)
    clone_offset: float = 1.0  # Angstrom; the clone sits this far from the original along the accumulated descent
    # direction -sum(dL/dr); both halves carry A/2 (mass conserving: the model is additive, not alpha-blended)
    add_from_residual_when_cloning: bool = True  # False = the clone rule replaces the residual back-projection add
    model_error_frac: float = 0.0  # round 3: add (frac * prediction)^2 to the shot-noise variance used by the whitened add
    # signal and the data-support removal, a systematic model-error floor that caps the per-pixel SNR at 1 / frac (at full
    # dose the shot noise is so small that every misfit looks significant); 0 = off (v1 / v2)
    variance_source: Literal["model", "max", "max_smooth"] = "model"  # round 3: the expected counts used for the shot-noise variance:
    # "model" = the prediction (v1 / v2); "max" = max(prediction, measured image), so pixels the model does not yet explain
    # keep their measured variance instead of the empty-pixel floor (at 1e8 that floor is ~1e-4 of a particle pixel's
    # variance and the whitened add then floods vacuum during the fill-in; chapter 06 round 3, S3vm 1e8); "max_smooth" =
    # the same with the image smoothed by a Gaussian of ``variance_blur_px`` pixels (no shot-noise bias at low dose)
    variance_blur_px: float = 1.5
    # --- round 5 step 3 (chapter 07, 2026-10-29): the per-atom energy as the remove and add-vetting signal. Everything
    # below is inert while ``energy_prior`` is None (every earlier run is reproduced unchanged). ---
    energy_prior: object | None = None  # anything with ``per_atom_energy(x) -> (N,)`` in eV at positions ``x`` (N, 3) in
    # Angstrom, e.g. :class:`quantem.tomography.energy_prior.EnergyPrior` (MACE); evaluated once per pass after the
    # prune and merge steps at the current fitted positions, and once more on the residual add candidates
    energy_prune_above: float = -2.4  # eV; an atom whose per-atom energy is above this is "unbound" (less bound than a
    # CN-5 Au corner atom, -2.5 eV; step 2 §F.5: 99.5 % of the extra atoms of the S4bm 1e8 refit lie above it, 0.2 % of
    # the matched atoms)
    energy_prune_require_low_amp: bool = True  # AND with "low data support" = amplitude below
    # ``energy_prune_amp_frac`` x the prune reference of this pass (the median amplitude, or its running maximum with
    # prune_reference="max_median"): a splat the data do not hold decays between the uniformity resets, a held atom
    # stays at the median. False = remove on the energy alone.
    energy_prune_amp_frac: float = 0.9
    energy_prune_max_frac: float | None = 0.2  # cap of the energy removals per pass (highest energies go first)
    energy_vet_add: bool = True  # reject residual add candidates whose per-atom energy, evaluated in place with the
    # candidates inserted among the current atoms (one forward per ``energy_vet_chunk`` candidates), is above the threshold
    energy_vet_chunk: int = 4000
    energy_start: int = 0  # first iteration at which the energy rule (prune and vet) applies; passes before it run as plain
    # v1. A blind start fills its particle in during the first ~300 iterations, and on an incomplete particle the true atoms
    # at the growing front read as unbound (the step-3 probe: every candidate the vet rejected at iteration 9-19 sat on a true
    # site), so the rule is meant for the complete structure (the driver starts it at 500 by default).
    # --- chapter 10 step 4 (2026-11-03): the candidate classifiers as the add / remove signals. Everything below is
    # inert at its default (add_method="residual", prune_method="amplitude": every earlier run is reproduced). ---
    add_method: Literal["residual", "classifier"] = "residual"  # "classifier": from ``cand_start`` on, the residual add is
    # replaced by the add classifier: residual peaks above ``cand_residual_frac`` x the usual add threshold become extra
    # candidates of :func:`~quantem.tomography.candidate_sites.generate_candidate_sites` (lattice continuation of the atoms),
    # ``cand_add_model.score_candidates`` scores every candidate and those with p > ``cand_p_add`` are added as weak atoms
    # (``cand_add_amp_frac`` x the median amplitude, median width; at most ``max_add_frac`` of the atoms, highest p first),
    # without the energy vet. Before ``cand_start`` the residual add runs as usual (a half-built particle implies every site).
    prune_method: Literal["amplitude", "classifier"] = "amplitude"  # "classifier": from ``cand_start`` on, an extra removal
    # step after the merge (before the energy prune): atoms with ``cand_rm_model.score_atoms`` p_real < ``cand_p_rm`` AND
    # amplitude < ``cand_rm_amp_frac`` x the current median are removed, at most ``cand_rm_max_frac`` per pass, lowest p first.
    # "amplitude" = no extra step (the amplitude prune runs in both cases).
    cand_add_model: object | None = None  # :class:`~quantem.tomography.candidate_classifier.CandidateClassifier` kind "add"
    # (or anything with ``score_candidates(atoms, candidates) -> p`` and ``.d1``)
    cand_rm_model: object | None = None  # kind "remove" (``score_atoms(atoms) -> p_real``)
    cand_start: int = 500  # first iteration of the classifier add / remove
    cand_p_add: float = 0.5
    cand_p_rm: float = 0.5
    cand_add_amp_frac: float = 0.3  # amplitude of a classifier-added atom / the median amplitude
    cand_confirm_iters: int = 100  # a classifier-added atom is judged by the data this many iterations after its addition:
    cand_confirm_amp_frac: float = 0.6  # removed ("rejected by the data") if its amplitude is then below this x the median,
    # else confirmed (an ordinary atom from then on). Until judged it is exempt from the uniformity reset, the amplitude
    # prune, the classifier removal and the energy prune (the out-of-box prune and the merge still apply).
    cand_rm_amp_frac: float = 0.9
    cand_rm_max_frac: float | None = 0.2
    cand_residual_frac: float = 0.5  # residual peaks above this x the usual add threshold become candidates
    cand_add_stop: int = 10**9  # no classifier adds from this iteration on (so every add is judged before the run ends;
    # the driver sets num_iter - cand_confirm_iters); the residual add does not resume after it
    cand_block_radius: float = 1.0  # Angstrom; a candidate this close to a site the data rejected (an add judged too weak,
    # or one that left the box before it was judged) is not proposed again (otherwise the add model re-adds the same
    # rejected site every pass); 0 disables

    @classmethod
    def v2(cls, **kwargs) -> "DensifyParams":
        """Bookkeeping v2 (chapter 06, 2026-10-27; the defaults since gate 06): the chapter 03 rules
        plus the shot-noise-whitened residual as the add signal and removal by data support below
        SNR 2 (at most 20 % of the atoms per pass). Needs ``dose_e_per_A2`` on the dataset (without
        it the loop falls back to v1 with a warning). ``kwargs`` override any field."""
        base = {"add_signal": "whitened", "remove_snr_min": 2.0, "max_remove_frac": 0.2}
        base.update(kwargs)
        return cls(**base)

    @classmethod
    def v1(cls, **kwargs) -> "DensifyParams":
        """The chapter 03 bookkeeping: raw residual as the add signal, no data-support removal."""
        base = {"add_signal": "residual", "remove_snr_min": 0.0, "max_remove_frac": None}
        base.update(kwargs)
        return cls(**base)


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
        self._energy_term_history: list[dict[str, float]] = []  # round 5 step 4: the energy term beside the prior
        self._lam_scale_energy: float | None = None
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
    def energy_term_history(self) -> list[dict[str, float]]:
        """Round 5 step 4: one record per refresh of the energy term given to :meth:`reconstruct` as
        ``energy_prior`` / ``energy_schedule`` (the second prior term, beside ``prior``)."""
        return getattr(self, "_energy_term_history", [])

    @property
    def energy_history(self) -> list[dict]:
        """Round 5 step 3: one record per energy prune pass ("prune") and per add-vetting call ("vet")."""
        return list(getattr(self, "_energy_history", []))

    @property
    def cand_history(self) -> list[dict]:
        """Chapter 10 step 4: one record per densify pass with the candidate classifiers on (``add_method`` or
        ``prune_method`` "classifier", from ``cand_start``)."""
        return list(getattr(self, "_cand_history", []))

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

    def _apply_energy_prior(
        self,
        prior,
        prior_schedule: "PriorSchedule",
        it: int,
        lam_raw: float,
        x: torch.Tensor,
        grad_args,
        second_term: bool = False,
    ) -> None:
        """``mode="energy"``: one ``energy_and_forces`` call, ``lam * dE/dx`` stored as a constant gradient.
        ``second_term`` (round 5 step 4): the energy runs beside the quadratic prior target, which is kept; its
        relative-lam scale and history are its own (``_lam_scale_energy``, ``energy_term_history``)."""
        if not hasattr(prior, "energy_and_forces"):
            raise TypeError(
                f"PriorSchedule(mode='energy') needs a prior with energy_and_forces(x); got {type(prior).__name__}"
            )
        e_total, e_atom, forces = prior.energy_and_forces(x)
        forces = forces.to(x)
        g_prior_unit = float(forces.norm())  # ||dE/dx|| at lam = 1
        scale_attr = "_lam_scale_energy" if second_term else "_lam_scale"
        if prior_schedule.lam_mode in ("relative", "relative_live"):
            if getattr(self, scale_attr, None) is None or prior_schedule.lam_mode == "relative_live":
                g_data = self._positions_grad_norm_data(*grad_args)
                setattr(self, scale_attr, g_data / g_prior_unit if g_prior_unit > 0 else 1.0)
            lam = lam_raw * getattr(self, scale_attr)
        else:
            lam = lam_raw
        self.obj_model.set_prior_gradient(-lam * forces, lam * float(e_total), keep_target=second_term)
        fnorm = forces.norm(dim=-1)
        if second_term and not hasattr(self, "_energy_term_history"):
            self._energy_term_history = []
        (self._energy_term_history if second_term else self._prior_history).append(
            {
                "iteration": it,
                "t": -1.0,
                "lam": float(lam),
                "energy": float(e_total),
                "energy_per_atom": float(e_total) / max(x.shape[0], 1),
                "fmax": float(fnorm.max()) if fnorm.numel() else 0.0,
                "force_rms": float(fnorm.pow(2).mean().sqrt()) if fnorm.numel() else 0.0,
                "grad_norm_prior": float(lam * g_prior_unit),
            }
        )
        self._last_energy_per_atom = e_atom.detach()

    def _snapshot(self, it: int) -> dict:
        shifts, z1, z3, dtilt = (v.detach().cpu().numpy() for v in self.dset.forward(None))
        d = self.obj_model.sites_dict()
        d.update({"iteration": it, "shifts": shifts, "z1": z1, "z3": z3, "dtilt": dtilt})
        return d

    @torch.no_grad()
    def _densify(
        self, it: int, params: DensifyParams, image_shape, sampling, window_sigmas, tilt_chunk
    ) -> dict:
        from quantem.core.ml.splat.densify import atom_support, pixel_variance, prune_mask
        from quantem.core.ml.splat.renderer import backproject_images, render_projections
        from quantem.tomography.atom_trace import gaussian_blur3d, seed_peaks

        obj = self.obj_model
        out = {"pruned": 0, "merged": 0, "added": 0, "split": 0, "removed_support": 0, "removed_whitened": 0}
        clone_sel = self._clone_candidates(params) if params.clone_grad_k > 0 else None
        if clone_sel is not None and not params.add_from_residual_when_cloning:
            params = replace(params, add_from_residual=False, split_from_residual=False)
        whiten = (
            params.add_signal == "whitened"
            or params.add_chi2_min > 0
            or params.remove_whitened_sigmas > 0
            or params.remove_snr_min > 0
        )
        if whiten and self.dset.dose_e_per_A2 is None:
            if not getattr(self, "_warned_no_dose", False):
                warnings.warn(
                    "whitened / support-based densification needs the dataset dose (dose_e_per_A2); "
                    "falling back to the chapter 03 rules (raw residual, no data-support removal)",
                    stacklevel=2,
                )
                self._warned_no_dose = True
            params = replace(
                params, add_signal="residual", add_chi2_min=0.0, remove_whitened_sigmas=0.0, remove_snr_min=0.0
            )
            whiten = False
        need_render = params.add_from_residual or params.remove_snr_min > 0 or params.remove_whitened_sigmas > 0
        target = rot = shifts = pred = None

        def render():
            target, rot, shifts = self.dset.batch(None)
            pred = obj.forward(rot, shifts, image_shape, sampling, None, window_sigmas, tilt_chunk)
            return target, rot, shifts, pred

        def variance(pred):
            return pixel_variance(
                pred, self.dset.intensity_scale, self.dset.dose_e_per_A2, sampling,
                target=target if params.variance_source in ("max", "max_smooth") else None, model_error_frac=params.model_error_frac,
                target_blur_px=params.variance_blur_px if params.variance_source == "max_smooth" else 0.0,
            )  # fmt: skip

        def bs_filter(vol):
            bs = params.add_blur_sigmas
            if isinstance(bs, (tuple, list)):
                return gaussian_blur3d(vol, float(bs[0])) - gaussian_blur3d(vol, float(bs[1]))
            return gaussian_blur3d(vol, float(bs))

        def robust_sigma(sel):
            med = sel.median()
            return 1.4826 * float((sel - med).abs().median())

        # chapter 10 step 4: the candidate classifiers (inert unless add_method / prune_method is "classifier")
        cls_replace = params.add_method == "classifier" and it >= params.cand_start  # the residual add is replaced
        cls_add = cls_replace and it < params.cand_add_stop
        cls_rm = params.prune_method == "classifier" and it >= params.cand_start
        crec, cand_extra = None, None
        if cls_replace or cls_rm or getattr(obj, "_atom_tags", None) is not None:
            crec = {"iteration": it, "n_atoms_start": obj.num_atoms}
            t_cls = time.perf_counter()
            self._cand_confirm(it, params, crec)  # judge the classifier-added atoms that are due (before any prune)
            out["confirmed"] = crec["n_confirmed"]
            out["rejected_data"] = crec["n_rejected_data"]
            n_tagged = self._n_unconfirmed()
            t_cls = time.perf_counter() - t_cls
        if params.prune_amplitude_frac > 0 and obj.num_atoms > 1:
            amp = obj.amplitudes.detach()
            ref = float(amp.median())
            if params.prune_reference == "max_median":
                ref = max(ref, float(getattr(self, "_prune_amp_ref", 0.0)))
                self._prune_amp_ref = ref
            keep = prune_mask(
                amp,
                obj._positions.detach(),
                float(params.prune_amplitude_frac * ref),
                obj.box_half if params.prune_outside_box else None,
            )
            tags = getattr(obj, "_atom_tags", None)
            if crec is not None and tags is not None:
                # chapter 10: an unjudged classifier add starts at cand_add_amp_frac x the median, which can be at or below
                # the prune bar (prune_reference="max_median" remembers the early, brighter median): the amplitude prune
                # leaves it to the data confirmation; leaving the box still removes it (and blocks the site)
                unj = (tags >= 0).to(keep.device)
                in_box = prune_mask(amp, obj._positions.detach(), -math.inf,
                                    obj.box_half if params.prune_outside_box else None)  # fmt: skip
                crec["n_prune_exempt_unjudged"] = int((unj & ~keep & in_box).sum())
                keep = keep | (unj & in_box)
                lost = ~keep & unj
                crec["n_pruned_unjudged"] = int(lost.sum())
                self._remember_rejected(obj._positions.detach()[lost])
            out["pruned"] = obj.prune(keep)
        if need_render:
            target, rot, shifts, pred = render()
        if params.remove_snr_min > 0 and obj.num_atoms > 1:
            # data support: the chi^2 each atom explains; below snr_min^2 the data do not need it
            sup = atom_support(
                obj._positions.detach(), obj.amplitudes.detach(), obj.sigmas.detach(), rot, shifts,
                sampling, image_shape, obj.window_radius(sampling, window_sigmas),
                target - pred, variance(pred), tilt_chunk,
            )  # fmt: skip
            self._last_support = sup.detach().cpu()
            keep = sup >= params.remove_snr_min**2
            keep = self._cap_removals(keep, sup, params.max_remove_frac)
            out["removed_support"] = obj.prune(keep)
            if out["removed_support"]:
                target, rot, shifts, pred = render()
        if params.merge_distance:
            out["merged"] = obj.merge(float(params.merge_distance), params.merge_amplitude)
            if out["merged"] and need_render:
                target, rot, shifts, pred = render()
        if cls_rm:
            t0 = time.perf_counter()
            out["removed_classifier"] = self._classifier_remove(it, params, crec) if obj.num_atoms > 1 else 0
            t_cls += time.perf_counter() - t0
            if out["removed_classifier"] and need_render:
                target, rot, shifts, pred = render()
        energy_on = params.energy_prior is not None and it >= params.energy_start
        if energy_on:
            # round 5 step 3: remove unbound atoms (per-atom energy above the threshold, AND low amplitude unless disabled)
            tags = getattr(obj, "_atom_tags", None)
            exempt = None if tags is None else tags >= 0  # chapter 10: unconfirmed classifier adds are judged by the data
            out["removed_energy"] = self._energy_prune(it, params, exempt) if obj.num_atoms > 1 else 0
            out["rejected_energy"] = 0
            if out["removed_energy"] and need_render:
                target, rot, shifts, pred = render()
        if params.add_from_residual or params.remove_whitened_sigmas > 0:
            v = float(params.voxel_size) if params.voxel_size else 2.0 * sampling
            D, H, W = obj.shape
            vol_shape = (
                max(2, int(round(D * obj._view_sampling / v))),
                max(2, int(round(H * obj._view_sampling / v))),
                max(2, int(round(W * obj._view_sampling / v))),
            )
            resid = target - pred  # raw residual: the chapter 03 add signal
            whiten_add = params.add_signal == "whitened"
            if whiten:  # whitened residual: the candidate-3 add signal, the whitened removal and the chi^2 gate
                sd = variance(pred).sqrt()
                resid_w = resid / sd
            if whiten_add:
                resid = resid_w
            bp_pred = None
            if params.add_noise_region == "support" and (
                params.add_noise_sigmas > 0 or params.remove_whitened_sigmas > 0
            ):
                bp_pred = backproject_images(pred.detach(), rot, shifts, sampling, vol_shape, v).cpu()

            def noise_of(resp):
                if bp_pred is not None:
                    sel = resp[bp_pred > 0.1 * float(bp_pred.max())]
                    if sel.numel() >= 64:
                        return robust_sigma(sel)
                return robust_sigma(resp.reshape(-1))

            if params.remove_whitened_sigmas > 0 and obj.num_atoms > 1:
                # the model is too bright in many views where the filtered whitened residual is strongly negative
                resp_s = bs_filter(backproject_images(resid_w, rot, shifts, sampling, vol_shape, v).cpu())
                noise = noise_of(resp_s)
                pos = obj._positions.detach().cpu()
                idx = [
                    (pos[:, 2 - k] / v + (vol_shape[k] - 1) / 2).round().long().clamp(0, vol_shape[k] - 1)
                    for k in range(3)
                ]
                at_atom = resp_s[idx[0], idx[1], idx[2]]
                keep = at_atom >= -params.remove_whitened_sigmas * noise
                keep = self._cap_removals(keep.to(obj.device), at_atom.to(obj.device), params.max_remove_frac)
                out["removed_whitened"] = obj.prune(keep)
                if out["removed_whitened"] and params.add_from_residual:
                    target, rot, shifts, pred = render()
                    resid = target - pred
                    sd = variance(pred).sqrt()
                    resid_w = resid / sd
                    if whiten_add:
                        resid = resid_w
            if params.add_from_residual:
                vol = backproject_images(torch.clamp(resid, min=0.0), rot, shifts, sampling, vol_shape, v)
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
                if whiten_add:
                    one = one / sd
                ref_resp = bs_filter(backproject_images(one, rot, shifts, sampling, vol_shape, v).cpu())
                thr = (
                    params.add_min_rel_peak * float(ref_resp.max())
                    if params.add_min_rel_peak > 0
                    else None
                )
                if params.add_noise_sigmas > 0:
                    noise = noise_of(bs_filter(vol.cpu()))
                    self._last_add_noise = noise
                    thr = max(thr or 0.0, params.add_noise_sigmas * noise)
                chi2_vol: list[torch.Tensor] = []  # the chi^2 gate's volume, computed once per pass when needed

                def residual_peaks(threshold, record: bool = True) -> torch.Tensor:
                    """Residual peaks above ``threshold`` (strongest first) after the chi^2 gate, (M, 3) Angstrom."""
                    peaks, _ = seed_peaks(
                        vol.cpu(),
                        blur_sigmas=params.add_blur_sigmas,
                        threshold=threshold,
                        threshold_fraction=params.add_threshold_fraction,
                        min_distance=params.add_min_distance / v,
                        max_peaks=params.max_add,
                    )
                    if peaks.shape[0] and params.add_chi2_min > 0:
                        # multi-view inconsistency gate: chi^2 per view at the candidate (1 = consistent model)
                        if not chi2_vol:
                            bs = params.add_blur_sigmas
                            chi2_vol.append(
                                gaussian_blur3d(
                                    backproject_images(resid_w * resid_w, rot, shifts, sampling, vol_shape, v).cpu(),
                                    float(bs[0] if isinstance(bs, (tuple, list)) else bs),
                                )
                            )
                        pk = [peaks[:, k].round().long().clamp(0, vol_shape[k] - 1) for k in range(3)]
                        ok = chi2_vol[0][pk[0], pk[1], pk[2]] >= params.add_chi2_min
                        if record:
                            self._last_chi2_rejected = int((~ok).sum())
                        peaks = peaks[ok]
                    return torch.stack(
                        (
                            (peaks[:, 2].double() - (vol_shape[2] - 1) / 2) * v,
                            (peaks[:, 1].double() - (vol_shape[1] - 1) / 2) * v,
                            (peaks[:, 0].double() - (vol_shape[0] - 1) / 2) * v,
                        ),
                        dim=1,
                    ).float()

                pos = residual_peaks(thr)
                if pos.shape[0]:
                    rejected = pos[:0]
                    rejected_nn = torch.zeros(0, dtype=torch.long)
                    if obj.num_atoms:
                        d, nn = torch.cdist(pos, obj._positions.detach().cpu()).min(dim=1)
                        accept = d > params.add_min_distance
                        rejected, rejected_nn = pos[~accept], nn[~accept]
                        pos = pos[accept]
                    if not cls_replace:  # chapter 10: from cand_start on the classifier add replaces the residual add
                        if params.max_add_frac is not None and obj.num_atoms:
                            cap = max(1, int(params.max_add_frac * obj.num_atoms))
                            pos = pos[:cap]  # seed_peaks returns peaks strongest first
                        if energy_on and params.energy_vet_add and pos.shape[0] and obj.num_atoms:
                            pos, out["rejected_energy"] = self._energy_vet_candidates(it, params, pos)
                        out["added"] = obj.add_atoms(pos)  # appends, so ``rejected_nn`` indices stay valid
                    if params.split_from_residual and rejected.shape[0]:
                        out["split"] = self._split_fat_atoms(rejected, rejected_nn, params)
                if cls_add:
                    # the weaker residual peaks (cand_residual_frac x the add threshold) away from the atoms (after the
                    # split) become extra candidates of the generator
                    t0 = time.perf_counter()
                    cand_extra = residual_peaks(
                        None if thr is None else params.cand_residual_frac * thr, record=False
                    )
                    if cand_extra.shape[0] and obj.num_atoms:
                        d = torch.cdist(cand_extra, obj._positions.detach().cpu()).min(dim=1).values
                        cand_extra = cand_extra[d > params.add_min_distance]
                    crec["residual_peaks_s"] = time.perf_counter() - t0
                    t_cls += crec["residual_peaks_s"]
        if cls_add:
            t0 = time.perf_counter()
            out["added_classifier"] = self._classifier_add(it, params, cand_extra, crec)
            t_cls += time.perf_counter() - t0
        if crec is not None:
            n_unc = self._n_unconfirmed()
            crec["n_lost_unjudged"] = n_tagged - (n_unc - crec.get("n_added", 0))  # amplitude prune (blocked) + merge
            crec.setdefault("n_pruned_unjudged", 0)
            crec["n_unconfirmed"] = n_unc
            crec["n_atoms_end"] = obj.num_atoms
            crec["classifier_s"] = t_cls  # confirmation + residual peaks for the candidates + generator + both models
            if not hasattr(self, "_cand_history"):
                self._cand_history: list[dict] = []
            self._cand_history.append(crec)
        if clone_sel is not None:
            out["cloned"] = self._clone_atoms(*clone_sel, params)
        self._reset_grad_accumulators()
        return out

    # --- per-atom energy as the remove / add-vetting signal (round 5 step 3, chapter 07) ---
    _ENERGY_HIST_EDGES = (-4.0, 2.0, 120)  # eV: min, max, bins of the per-pass energy histogram kept in the history

    def _energy_history_append(self, rec: dict) -> None:
        if not hasattr(self, "_energy_history"):
            self._energy_history: list[dict] = []
        self._energy_history.append(rec)

    @torch.no_grad()
    def _energy_prune(self, it: int, params: DensifyParams, exempt: torch.Tensor | None = None) -> int:
        """Remove the atoms whose per-atom energy at the current positions is above ``params.energy_prune_above`` (eV)
        and, unless ``energy_prune_require_low_amp`` is False, whose amplitude is below ``energy_prune_amp_frac`` x the
        CURRENT median amplitude (not the prune step's running-maximum reference: early in a blind start every amplitude
        decays from a too-bright init, and only an atom weaker than its peers lacks data support); at most
        ``energy_prune_max_frac`` of the atoms per pass, highest energies first. Logs the count, the energies, the
        relative amplitudes and the positions of the removed atoms and a histogram of all energies to
        ``self._energy_history``. ``exempt`` (N,) bool (chapter 10): atoms never removed in this pass (the unconfirmed
        classifier-added atoms). Returns the number removed."""
        obj = self.obj_model
        x = obj._positions.detach()
        amp = obj.amplitudes.detach()
        amp_ref = float(amp.median())
        t0 = time.perf_counter()
        e = params.energy_prior.per_atom_energy(x).detach().to(device=x.device, dtype=torch.float32).reshape(-1)  # type: ignore[attr-defined]
        mace_s = time.perf_counter() - t0
        if e.shape[0] != x.shape[0]:
            raise ValueError(f"per_atom_energy returned {e.shape[0]} values for {x.shape[0]} atoms")
        high = e > params.energy_prune_above
        low_amp = amp < params.energy_prune_amp_frac * amp_ref
        marked = (high & low_amp) if params.energy_prune_require_low_amp else high
        n_exempt = 0
        if exempt is not None:
            n_exempt = int((marked & exempt.to(marked.device)).sum())
            marked = marked & ~exempt.to(marked.device)
        keep = self._cap_removals(~marked, -e, params.energy_prune_max_frac)
        removed = ~keep
        lo, hi, nb = self._ENERGY_HIST_EDGES
        rel = amp / amp_ref
        rec = {
            "kind": "prune", "iteration": it, "n_atoms": int(x.shape[0]), "threshold_eV": float(params.energy_prune_above),
            "require_low_amp": bool(params.energy_prune_require_low_amp), "amp_frac": float(params.energy_prune_amp_frac),
            "amp_ref": float(amp_ref), "n_above": int(high.sum()), "n_above_low_amp": int((high & low_amp).sum()),
            "n_marked": int(marked.sum()), "n_removed": int(removed.sum()), "capped": bool(int(removed.sum()) < int(marked.sum())),
            "energies_removed_eV": [round(v, 4) for v in e[removed].tolist()],
            "amp_rel_removed": [round(v, 3) for v in rel[removed].tolist()],
            "amp_rel_above_kept": [round(v, 3) for v in rel[high & keep].tolist()],
            "positions_removed_A": [[round(c, 3) for c in row] for row in x[removed].tolist()],
            "hist_all_edges_eV": [lo, hi, nb],
            "hist_all": [int(v) for v in torch.histc(e.clamp(lo, hi), bins=nb, min=lo, max=hi).tolist()],
            "e_p50_eV": float(e.median()), "mace_s": mace_s,
        }  # fmt: skip
        if exempt is not None:
            rec["n_exempt_unconfirmed"] = n_exempt  # marked, but unconfirmed classifier adds (left to the data)
        self._energy_history_append(rec)
        return obj.prune(keep)

    # --- the candidate classifiers as the add / remove signals (chapter 10 step 4) ---
    def _n_unconfirmed(self) -> int:
        tags = getattr(self.obj_model, "_atom_tags", None)
        return 0 if tags is None else int((tags >= 0).sum())

    def _established_mask(self) -> torch.Tensor | None:
        """(N,) bool: atoms that are not unconfirmed classifier adds, or None when there are no unconfirmed adds."""
        tags = getattr(self.obj_model, "_atom_tags", None)
        if tags is None or not bool((tags >= 0).any()):
            return None
        return tags < 0

    def _established_median(self, values: torch.Tensor) -> torch.Tensor:
        """Median of ``values`` (N,) over the established atoms (all atoms when every atom is established), so the
        weak unconfirmed adds do not lower the reference they are judged against."""
        est = self._established_mask()
        if est is None or not bool(est.any()):
            return values.median()
        return values[est.to(values.device)].median()

    @staticmethod
    def _short(v: torch.Tensor, nd: int = 3, n_max: int = 200) -> list:
        """A rounded python list of at most ``n_max`` values (the history stays small)."""
        return [round(float(a), nd) for a in v.reshape(-1)[:n_max].tolist()]

    @torch.no_grad()
    def _cand_confirm(self, it: int, params: DensifyParams, crec: dict) -> None:
        """Judge the classifier-added atoms added ``cand_confirm_iters`` or more iterations ago: below
        ``cand_confirm_amp_frac`` x the established median amplitude they are removed (rejected by the data), else
        confirmed (their tag is cleared and they are ordinary atoms from then on)."""
        obj = self.obj_model
        tags = getattr(obj, "_atom_tags", None)
        crec.update({"n_confirmed": 0, "n_rejected_data": 0, "amp_rel_confirmed": [], "amp_rel_rejected": []})
        if tags is None:
            return
        due = (tags >= 0) & (it - tags >= int(params.cand_confirm_iters))
        if not bool(due.any()):
            return
        amp = obj.amplitudes.detach()
        rel = amp / self._established_median(amp)
        reject = due & (rel < params.cand_confirm_amp_frac)
        confirm = due & ~reject
        crec.update(
            {
                "n_confirmed": int(confirm.sum()),
                "n_rejected_data": int(reject.sum()),
                "amp_rel_confirmed": self._short(rel[confirm]),
                "amp_rel_rejected": self._short(rel[reject]),
                "added_at_judged": sorted({int(t) for t in tags[due].tolist()}),
                "positions_rejected_A": [[round(c, 3) for c in row] for row in obj._positions.detach()[reject][:200].tolist()],
            }
        )
        tags[confirm] = -1
        self._remember_rejected(obj._positions.detach()[reject])
        obj.prune(~reject)

    def _remember_rejected(self, pos: torch.Tensor) -> None:
        """Sites the data rejected (judged too weak, or pruned before judged), so the add model does not propose them
        again (``cand_block_radius``)."""
        if pos.shape[0] == 0:
            return
        prev = getattr(self, "_cand_rejected_sites", None)
        new = pos.detach().float().cpu()
        self._cand_rejected_sites = new if prev is None else torch.cat([prev, new])

    @torch.no_grad()
    def _classifier_add(self, it: int, params: DensifyParams, extra: torch.Tensor | None, crec: dict) -> int:
        """The candidate generator (lattice continuation of the current atoms + ``extra`` residual peaks) proposes, the
        add model scores, candidates with p > ``cand_p_add`` are added (highest p first, at most ``max_add_frac`` of the
        atoms) as weak atoms (``cand_add_amp_frac`` x the established median amplitude, median width) tagged with
        this iteration. No energy vet. Returns the number added."""
        from quantem.tomography.candidate_sites import generate_candidate_sites

        obj, model = self.obj_model, params.cand_add_model
        if model is None:
            raise ValueError("add_method='classifier' needs cand_add_model")
        n_extra = 0 if extra is None else int(extra.shape[0])
        crec.update({"n_residual_peaks": n_extra, "n_candidates": 0, "n_from_residual": 0, "n_accepted": 0, "n_added": 0})
        if obj.num_atoms < 2:
            return 0
        x = obj._positions.detach()
        sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
        t0 = time.perf_counter()
        cs = generate_candidate_sites(x, extra=None if not n_extra else extra.to(x), d1=float(model.d1))
        sync()
        t1 = time.perf_counter()
        cpos, from_res = cs.positions, cs.from_residual
        crec.update({"n_candidates": int(cpos.shape[0]), "n_from_residual": int(from_res.sum()),
                     "n_under": int(cs.n_under), "generator_s": t1 - t0, "n_blocked": 0})  # fmt: skip
        blocked = getattr(self, "_cand_rejected_sites", None)
        if blocked is not None and params.cand_block_radius > 0 and cpos.shape[0]:
            ok = torch.cdist(cpos.float(), blocked.to(cpos.device)).min(dim=1).values > params.cand_block_radius
            crec["n_blocked"] = int((~ok).sum())
            cpos, from_res = cpos[ok], from_res[ok]
        if cpos.shape[0] == 0:
            crec["add_score_s"] = 0.0
            return 0
        p = model.score_candidates(x, cpos).reshape(-1).to(x.device)
        sync()
        crec["add_score_s"] = time.perf_counter() - t1
        acc = torch.nonzero(p > params.cand_p_add).reshape(-1)
        acc = acc[torch.argsort(p[acc], descending=True)]
        crec["n_accepted"] = int(acc.numel())
        if params.max_add_frac is not None:
            acc = acc[: max(1, int(params.max_add_frac * obj.num_atoms))]
        crec["p_hist_edges"] = [0.0, 1.0, 10]
        crec["p_hist_candidates"] = [int(v) for v in torch.histc(p.float(), bins=10, min=0.0, max=1.0).tolist()]
        crec["p_added"] = self._short(p[acc])
        crec["from_residual_added"] = int(from_res.to(p.device)[acc].sum())
        if acc.numel() == 0:
            return 0
        amp = float(params.cand_add_amp_frac * self._established_median(obj.amplitudes.detach()))
        sig = float(self._established_median(obj.sigmas.detach()))
        n0 = obj.num_atoms
        if getattr(obj, "_atom_tags", None) is None:
            obj._atom_tags = torch.full((n0,), -1, dtype=torch.long, device=x.device)
        crec["positions_added_A"] = [[round(c, 3) for c in row] for row in cpos[acc][:200].tolist()]
        n = obj.add_atoms(cpos[acc], amplitude=amp, sigma=sig)
        obj._atom_tags[n0:] = it
        crec["n_added"] = n
        crec["amp_added"] = amp
        return n

    @torch.no_grad()
    def _classifier_remove(self, it: int, params: DensifyParams, crec: dict) -> int:
        """Remove the atoms the remove model calls spurious (p_real < ``cand_p_rm``) AND that the data hold weakly
        (amplitude < ``cand_rm_amp_frac`` x the established median), at most ``cand_rm_max_frac`` of the atoms per pass,
        lowest p first; unconfirmed classifier adds are left to the data. Returns the number removed."""
        obj, model = self.obj_model, params.cand_rm_model
        if model is None:
            raise ValueError("prune_method='classifier' needs cand_rm_model")
        x = obj._positions.detach()
        amp = obj.amplitudes.detach()
        sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
        t0 = time.perf_counter()
        p = model.score_atoms(x).reshape(-1).to(x.device)
        sync()
        rel = amp / self._established_median(amp)
        low_p = p < params.cand_p_rm
        low_amp = rel < params.cand_rm_amp_frac
        marked = low_p & low_amp
        est = self._established_mask()
        if est is not None:
            marked = marked & est.to(marked.device)
        keep = self._cap_removals(~marked, p, params.cand_rm_max_frac)
        removed = ~keep
        crec.update(
            {
                "rm_score_s": time.perf_counter() - t0,
                "n_scored": int(x.shape[0]),
                "n_low_p": int(low_p.sum()),
                "n_low_p_high_amp": int((low_p & ~low_amp).sum()),  # spurious by geometry, but the data hold them: kept
                "n_rm_marked": int(marked.sum()),
                "n_removed_classifier": int(removed.sum()),
                "rm_capped": bool(int(removed.sum()) < int(marked.sum())),
                "p_removed": self._short(p[removed]),
                "amp_rel_removed": self._short(rel[removed]),
                "positions_removed_A": [[round(c, 3) for c in row] for row in x[removed][:200].tolist()],
            }
        )
        return obj.prune(keep)

    @torch.no_grad()
    def _energy_vet_candidates(self, it: int, params: DensifyParams, cand: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Per-atom energy of every residual add candidate evaluated in place (the candidates inserted among the current
        atoms, ``energy_vet_chunk`` candidates per forward; candidates of one chunk see each other); candidates above
        ``energy_prune_above`` are rejected. Returns ``(accepted candidates, number rejected)``."""
        obj = self.obj_model
        x = obj._positions.detach()
        n = int(x.shape[0])
        c_dev = cand.to(x)
        t0 = time.perf_counter()
        es = []
        for i0 in range(0, int(c_dev.shape[0]), max(1, int(params.energy_vet_chunk))):
            chunk = c_dev[i0 : i0 + max(1, int(params.energy_vet_chunk))]
            e = params.energy_prior.per_atom_energy(torch.cat([x, chunk], dim=0))  # type: ignore[attr-defined]
            es.append(e.detach().to(device=x.device, dtype=torch.float32).reshape(-1)[n:])
        e = torch.cat(es)
        mace_s = time.perf_counter() - t0
        ok = (e <= params.energy_prune_above).to(cand.device)
        rec = {
            "kind": "vet", "iteration": it, "n_atoms": n, "threshold_eV": float(params.energy_prune_above),
            "n_candidates": int(cand.shape[0]), "n_rejected": int((~ok).sum()),
            "energies_rejected_eV": [round(v, 4) for v in e[~ok.to(e.device)].tolist()],
            "energies_accepted_eV": [round(v, 4) for v in e[ok.to(e.device)].tolist()],
            "positions_rejected_A": [[round(c, 3) for c in row] for row in cand[~ok].tolist()],
            "mace_s": mace_s,
        }  # fmt: skip
        self._energy_history_append(rec)
        return cand[ok], int((~ok).sum())

    # --- gradient-clone add rule (chapter 06 round 3) ---
    def _reset_grad_accumulators(self) -> None:
        self._grad_norm_acc = None
        self._grad_vec_acc = None
        self._grad_count = 0

    @torch.no_grad()
    def _accumulate_position_grad(self) -> None:
        """Add this step's per-atom positional gradient (norm and vector) to the densify-interval accumulators."""
        g = self.obj_model._positions.grad
        if g is None:
            return
        acc = getattr(self, "_grad_norm_acc", None)
        if acc is None or acc.shape[0] != g.shape[0]:  # first step, or the atom set changed outside a densify pass
            self._grad_norm_acc = torch.zeros(g.shape[0], device=g.device, dtype=g.dtype)
            self._grad_vec_acc = torch.zeros_like(g)
            self._grad_count = 0
        self._grad_norm_acc += g.norm(dim=-1)
        self._grad_vec_acc += g
        self._grad_count += 1

    @torch.no_grad()
    def _clone_candidates(self, params: DensifyParams):
        """Atoms whose mean positional-gradient norm over the interval exceeds ``clone_grad_k`` x the population
        median, strongest first and capped at ``max_add_frac`` of the atoms; returns ``(positions, descent
        directions)`` of the selected atoms (positions identify them after the prune / merge / add steps of the pass,
        which re-index the atoms), or ``None``."""
        obj = self.obj_model
        acc = getattr(self, "_grad_norm_acc", None)
        if acc is None or acc.shape[0] != obj.num_atoms or self._grad_count == 0 or obj.num_atoms < 2:
            return None
        score = acc / self._grad_count
        sel = torch.nonzero(score > params.clone_grad_k * score.median()).reshape(-1)
        if sel.numel() == 0:
            return None
        sel = sel[torch.argsort(score[sel], descending=True)]
        if params.max_add_frac is not None:
            sel = sel[: max(1, int(params.max_add_frac * obj.num_atoms))]
        d = -self._grad_vec_acc[sel]
        n = d.norm(dim=-1, keepdim=True)
        flat = n.reshape(-1) <= 1e-30  # a vanishing summed gradient: any direction, take +x
        d = torch.where(n > 1e-30, d / n.clamp_min(1e-30), torch.zeros_like(d))
        d[flat, 0] = 1.0
        return obj._positions.detach()[sel].clone(), d

    @torch.no_grad()
    def _clone_atoms(self, sel_pos: torch.Tensor, direction: torch.Tensor, params: DensifyParams) -> int:
        """Clone each selected atom that survived the pass unchanged (same position): the clone sits
        ``clone_offset`` along the descent direction, original and clone carry half the amplitude each and the
        original keeps its width, position and optimizer moments (the clone starts from zero moments)."""
        obj = self.obj_model
        if obj.num_atoms == 0:
            return 0
        d, nn = torch.cdist(sel_pos, obj._positions.detach()).min(dim=1)
        ok = d < 1e-5  # pruned / merged / split atoms are gone or moved: skip them
        if not bool(ok.any()):
            return 0
        idx = nn[ok]
        amp = obj.amplitudes.detach()[idx] / 2.0
        new_pos = obj._positions.detach()[idx] + params.clone_offset * direction[ok]
        obj._rebuild(None, new_pos, amp, obj.sigmas.detach()[idx])
        obj._log_amplitudes[idx] -= math.log(2.0)
        return int(idx.numel())

    @staticmethod
    def _cap_removals(keep: torch.Tensor, score: torch.Tensor, max_frac: float | None) -> torch.Tensor:
        """Keep at most ``max_frac`` of the atoms marked for removal per pass (the lowest scores go first)."""
        if max_frac is None:
            return keep
        n_rm = int((~keep).sum())
        cap = int(max_frac * keep.numel())
        if n_rm <= cap:
            return keep
        keep = keep.clone()
        order = torch.argsort(score)  # lowest score first
        marked = order[~keep[order]]
        keep[marked[cap:]] = True
        return keep

    def _split_fat_atoms(
        self, candidates: torch.Tensor, nearest: torch.Tensor, params: DensifyParams
    ) -> int:
        """Replace each over-bright atom that has a rejected residual peak nearby by two half-amplitude
        atoms: at the peak and at its mirror image through the atom (or at the two nearest peaks when
        there are two). This is the "two atoms collapsed onto one Gaussian between two sites" case,
        which the add step cannot fix because both true sites lie within ``add_min_distance`` of the
        fat atom. Returns the number of atoms split."""
        obj = self.obj_model
        amp = obj.amplitudes.detach().cpu()
        pos = obj._positions.detach().cpu()
        fat = amp > params.split_amplitude_frac * amp.median()
        min_sep = float(params.merge_distance or 0.0)
        new_pos, new_amp, drop = [], [], []
        for i in torch.unique(nearest).tolist():
            if not bool(fat[i]):
                continue
            c = candidates[nearest == i]
            off = (c - pos[i]).norm(dim=1)
            c = c[off >= params.split_min_offset]
            if c.shape[0] == 0:
                continue
            order = torch.argsort((c - pos[i]).norm(dim=1))
            q1 = c[order[0]]
            q2 = c[order[1]] if c.shape[0] >= 2 else 2.0 * pos[i] - q1
            if float((q1 - q2).norm()) < min_sep:
                continue  # the merge step would undo it
            new_pos += [q1, q2]
            new_amp += [amp[i] / 2.0, amp[i] / 2.0]
            drop.append(i)
        if not drop:
            return 0
        keep = torch.ones(obj.num_atoms, dtype=torch.bool)
        keep[drop] = False
        keep_idx = torch.nonzero(keep).reshape(-1).to(obj.device)
        sig = obj.sigmas.detach().median().reshape(1).expand(len(new_pos))
        obj._rebuild(
            keep_idx,
            torch.stack(new_pos).to(obj._positions),
            torch.stack(new_amp).to(obj._positions),
            sig,
        )
        return len(drop)

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
        uniform_every: int = 0,
        uniform_window: tuple[float, float] | None = None,
        species_reset_every: int = 0,
        pose_l1: tuple[float, float] = (0.0, 0.0),
        energy_prior=None,
        energy_schedule: PriorSchedule | None = None,
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
        uniform_every, uniform_window
            Every ``uniform_every`` iterations reset the amplitudes and widths to their medians (only the atoms
            within ``uniform_window`` x median when given).
        species_reset_every : int
            Every ``species_reset_every`` iterations a HARD reset of every atom's amplitude and width to the
            medians (single species), independent of the windowed reset; 0 disables.
        pose_l1 : (float, float)
            L1 penalty on the learned pose corrections, added to the loss while poses are learned:
            ``w_shift * mean_k |shift_k|_1`` (pixels) ``+ w_angle * mean_k (|z1_k| + |z3_k| + |dtilt_k|)`` (degrees).
        energy_prior, energy_schedule
            Round 5 step 4: a second prior term, an energy (``energy_and_forces(x)``, e.g.
            :class:`~quantem.tomography.energy_prior.EnergyPrior`) with its own :class:`PriorSchedule`
            (``mode`` is taken as "energy"), applied beside ``prior`` (typically the learned denoiser in
            "loss" mode): ``lam * E(x)`` enters the loss as a constant gradient refreshed every
            ``energy_schedule.every`` iterations, and the quadratic target of ``prior`` is kept. ``None``
            (the default) = no second term. Records in :attr:`energy_term_history`.
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

        track_grad = densify is not None and densify.every > 0 and densify.clone_grad_k > 0
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
                if learn_pose and (pose_l1[0] > 0 or pose_l1[1] > 0):
                    sh, z1, z3, dt = self.dset.forward(None)
                    soft = soft + float(pose_l1[0]) * sh.abs().sum(dim=-1).mean()
                    soft = soft + float(pose_l1[1]) * (z1.abs() + z3.abs() + dt.abs()).mean()
                loss = data_loss + soft
                loss.backward()
                if track_grad:
                    self._accumulate_position_grad()
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
                if not hasattr(self, "_densify_history"):
                    self._densify_history = []
                self._densify_history.append({"iteration": it, "n_atoms": self.obj_model.num_atoms, **changes})
                if any(changes.values()):
                    self.set_optimizers() if self.obj_model.optimizer is None else None
                    if self.obj_model.constraints.repulsion > 0:
                        self.obj_model.refresh_neighbors()

            # --- prior target refresh ---
            if (
                prior is not None
                and prior_schedule is not None
                and prior_schedule.start <= it < prior_schedule.stop
                and ((it - prior_schedule.start) % max(1, prior_schedule.every) == 0)
            ):
                # round 6 step 4: sigma_mode="pull" reads the previous call's per-coordinate RMS pull (None at the first call)
                last_pull_rms = getattr(self, "_last_pull_rms_coord", None)
                t, lam_raw = prior_schedule.at(it, schedule_total or num_iter, prior, last_pull_rms)
                sigma_t = (
                    prior_schedule.sigma_at(it, schedule_total or num_iter, last_pull_rms)
                    if prior_schedule.sigma is not None and hasattr(prior, "t_of_sigma")
                    else None
                )
                x = self.obj_model._positions.detach()
                if prior_schedule.mode == "energy":
                    self._apply_energy_prior(
                        prior, prior_schedule, it, lam_raw, x,
                        (loss_func, image_shape, sampling, window_sigmas, tilt_chunk),
                    )
                else:
                    x_hat = prior.clean_estimate(
                        x, t=t, **({"noise_scale": prior_schedule.noise_scale} if hasattr(prior, "t_of_sigma") else {})
                    ).to(x)
                    pull = (x_hat - x).norm(dim=-1)
                    if prior_schedule.mode == "proximal":
                        with torch.no_grad():
                            self.obj_model._positions.add_(prior_schedule.alpha * (x_hat - x))
                        self.obj_model.clear_prior_target(keep_gradient=energy_prior is not None)
                        lam = 0.0
                    elif prior_schedule.lam_mode in ("relative", "relative_live"):
                        # "relative": the data/prior gradient-norm ratio is measured once, at the first
                        # application; "relative_live": re-measured at every application, so lam_rel keeps
                        # its meaning when the prior starts late on an already-converged fit
                        if self._lam_scale is None or prior_schedule.lam_mode == "relative_live":
                            g_data = self._positions_grad_norm_data(
                                loss_func, image_shape, sampling, window_sigmas, tilt_chunk
                            )
                            g_prior_unit = 2.0 * float(pull.norm()) / max(x.shape[0], 1)
                            self._lam_scale = g_data / g_prior_unit if g_prior_unit > 0 else 1.0
                        lam = lam_raw * self._lam_scale
                    else:
                        lam = lam_raw
                    if prior_schedule.mode != "proximal":
                        self.obj_model.set_prior_target(x_hat, lam)
                    # per-coordinate RMS of the pull (rms over atoms of ||x_hat - x|| / sqrt(3)): the error measure of
                    # sigma_mode="pull" at the next call, recorded in every mode (round 6 step 4)
                    pull_rms_coord = float(pull.pow(2).mean().sqrt() / math.sqrt(3.0))
                    self._last_pull_rms_coord = pull_rms_coord
                    self._prior_history.append(
                        {
                            "iteration": it,
                            "t": -1.0 if t is None else float(t),
                            "sigma": -1.0 if sigma_t is None else float(sigma_t),
                            "lam": float(lam),
                            "pull_mean": float(pull.mean()),
                            "pull_max": float(pull.max()),
                            "pull_rms_coord": pull_rms_coord,
                            "pull_frac_gt_0p5": float((pull > 0.5).float().mean()),
                        }
                    )

            if prior is not None and prior_schedule is not None and it == prior_schedule.stop:
                self.obj_model.clear_prior_target(keep_gradient=energy_prior is not None)

            # --- the energy term beside the prior (round 5 step 4) ---
            if (
                energy_prior is not None
                and energy_schedule is not None
                and energy_schedule.start <= it < energy_schedule.stop
                and ((it - energy_schedule.start) % max(1, energy_schedule.every) == 0)
            ):
                _, lam_raw_e = energy_schedule.at(it, schedule_total or num_iter, None)
                self._apply_energy_prior(
                    energy_prior, energy_schedule, it, lam_raw_e, self.obj_model._positions.detach(),
                    (loss_func, image_shape, sampling, window_sigmas, tilt_chunk), second_term=True,
                )
            if energy_prior is not None and energy_schedule is not None and it == energy_schedule.stop:
                self.obj_model.set_prior_gradient(None, keep_target=True)
            if (uniform_every and (it + 1) % uniform_every == 0) or (
                species_reset_every and (it + 1) % species_reset_every == 0
            ):
                with torch.no_grad():  # single species: one amplitude and one width for every atom
                    la, ls = self.obj_model._log_amplitudes, self.obj_model._log_sigmas
                    hard = bool(species_reset_every) and (it + 1) % species_reset_every == 0
                    est = self._established_mask()
                    if est is None:
                        ma, ms = float(la.median()), float(ls.median())
                        if uniform_window is None or hard:
                            la.fill_(ma)
                            ls.fill_(ms)
                        else:  # leave the outliers alone so spurious atoms can still decay and be pruned
                            lo, hi = math.log(uniform_window[0]), math.log(uniform_window[1])
                            sel = (la - ma >= lo) & (la - ma <= hi)
                            la[sel] = ma
                            ls[sel] = ms
                    else:  # chapter 10: unconfirmed classifier adds keep their own amplitude and width until judged
                        est = est.to(la.device)
                        ma, ms = float(la[est].median()), float(ls[est].median())
                        sel = est.clone()
                        if uniform_window is not None and not hard:
                            lo, hi = math.log(uniform_window[0]), math.log(uniform_window[1])
                            sel &= (la - ma >= lo) & (la - ma <= hi)
                        la[sel] = ma
                        ls[sel] = ms
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
