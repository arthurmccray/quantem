"""Multi-tilt 4D-STEM dataset model for joint ptycho-tomography.

``PtychoTomoDatasetRaster`` aggregates a tilt series of per-tilt ``PtychographyDatasetRaster``
instances into the flat ``(tilt x scan position)`` index space the ptychography reconstruction
loop expects, by subclassing the same base and concatenating into the *inherited* storage
(targets, scan-position / descan parameters, amplitudes). Mixed-tilt shuffled batches, the
DataLoader protocol, multi-GPU samplers, and ``error_estimate`` then work unchanged.

Per-tilt geometry (tilt angle + pose-correction slots) lives here, mirroring the tomography
module's design. Phase 1 of pose refinement (2026-08) makes the per-tilt beam-frame SHIFTS a
learned quantity (``set_learn_pose_shifts``); the angle slots stay frozen. ``forward`` returns a ``forward`` returns a :class:`PtychoTomoPatchData` payload carrying
beam-frame patch coordinates, per-element beam->specimen rotation matrices and (when active)
per-element beam-frame shifts, which the rotation-aware object models consume. Only implicit (coordinate-queried) object models are
supported — there is no integer patch-index path through a rotated volume.
"""

import warnings
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from quantem.core import config
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.ml.optimizer_mixin import OptimizerMixin, OptimizerParams
from quantem.diffractive_imaging.dataset_models import (
    DatasetConstraints,
    PtychographyDatasetRaster,
)
from quantem.ptycho_tomography.geometry import PtychoTomoPatchData, rot_beam_to_spec


class PtychoTomoDatasetRaster(DatasetConstraints):
    """Tilt series of raster 4D-STEM scans presented as one flat ptychography dataset.

    Construction wraps each tilt in a :class:`PtychographyDatasetRaster` (kept in
    ``self.tilt_datasets`` as preprocessing containers); the concatenated diffraction stack backs
    the inherited storage, so ``num_gpts = sum(per-tilt scan points)`` and all per-position
    parameters/targets are flat-indexed. ``_tilt_offsets`` maps flat indices to tilts.

    Scope: identical scan geometry across tilts, known tilt angles, no descan / scan-position
    learning. Pose refinement: the per-tilt beam-frame shift ``_pose_shifts`` is learnable
    (``set_learn_pose_shifts``, gauge-fixed by pinning the reference tilt — see the pose
    refinement region); ``_pose_z1`` / ``_pose_dtheta`` / ``_pose_z3`` exist but stay frozen. Per-tilt CoM rotation is
    forced to zero — solving it per tilt would scramble the cross-tilt geometry.
    """

    # registered buffer / parameter types (mirrors the base's _patch_indices declaration)
    _tilt_offsets: torch.Tensor
    _tilt_angles_deg: torch.Tensor
    _slab_window_flag: torch.Tensor
    # transient per-batch stash (slab-window mode): consumed once by
    # PtychoTomography.forward_operator, never serialized non-None
    _last_window_dz_A: torch.Tensor | None = None
    _scan_center_px: torch.Tensor
    _pose_z1: nn.Parameter
    _pose_dtheta: nn.Parameter
    _pose_z3: nn.Parameter
    _pose_shifts: nn.Parameter
    _pose_shifts_init: torch.Tensor
    _pose_shifts_accum: torch.Tensor

    def __init__(
        self,
        tilt_datasets: list[PtychographyDatasetRaster],
        tilt_angles_deg: "np.ndarray | torch.Tensor | list[float]",
        detector_mask: torch.Tensor | np.ndarray | None = None,
        verbose: int | bool = 1,
        _token: object | None = None,
    ):
        if len(tilt_datasets) < 1:
            raise ValueError("tilt_datasets must contain at least one dataset")
        angles = np.atleast_1d(np.asarray(tilt_angles_deg, dtype=float))
        if len(angles) != len(tilt_datasets):
            raise ValueError(
                f"got {len(angles)} tilt angles for {len(tilt_datasets)} tilt datasets"
            )
        ref = tilt_datasets[0]
        for i, ds in enumerate(tilt_datasets):
            if not isinstance(ds, PtychographyDatasetRaster):
                raise TypeError(
                    f"tilt_datasets[{i}] must be a PtychographyDatasetRaster, got {type(ds)}"
                )
            same = (
                np.array_equal(ds.roi_shape, ref.roi_shape)
                and np.array_equal(ds.gpts, ref.gpts)
                and np.allclose(ds.scan_sampling, ref.scan_sampling)
                and np.allclose(ds.detector_sampling, ref.detector_sampling)
            )
            if not same:
                raise ValueError(
                    f"tilt_datasets[{i}] geometry (roi/gpts/sampling) differs from tilt 0; "
                    "v1 requires identical scan geometry across tilts"
                )

        arrays = [self._tilt_array_3d(ds) for ds in tilt_datasets]
        concat = Dataset3d.from_array(
            array=np.concatenate(arrays, axis=0),
            name="ptycho_tomo_tilt_series",
            origin=ref.dset.origin,
            sampling=ref.dset.sampling,
            units=ref.dset.units,
        )
        self.tilt_datasets = tilt_datasets  # plain list: not registered as submodules
        super().__init__(
            dset=concat,
            detector_mask=detector_mask,
            verbose=verbose,
            learn_descan=False,
            learn_scan_positions=False,
            _token=_token,
        )
        # implicit-object only: drop the (num_gpts, *roi) int32 patch-index buffer (the rotated
        # forward never gathers integer patches; for a 5-tilt series this saves >1 GB)
        self._patch_indices = torch.empty(0, dtype=torch.int32)

        real_dtype = getattr(torch, config.get("dtype_real"))
        counts = [int(ds.num_gpts) for ds in tilt_datasets]
        self.register_buffer(
            "_tilt_offsets", torch.tensor(np.concatenate([[0], np.cumsum(counts)])).long()
        )
        self.register_buffer("_tilt_angles_deg", torch.as_tensor(angles, dtype=real_dtype))
        # Per-tilt pose-correction slots (tomography convention: z1 / x-tilt offset / z3 Euler
        # angles + beam-frame shifts). All frozen at construction; ``set_learn_pose_shifts``
        # flips the shift slot (an optimizer group + requires_grad, not an interface change).
        num_tilts = len(tilt_datasets)
        self._pose_z1 = nn.Parameter(torch.zeros(num_tilts, dtype=real_dtype), requires_grad=False)
        self._pose_dtheta = nn.Parameter(
            torch.zeros(num_tilts, dtype=real_dtype), requires_grad=False
        )
        self._pose_z3 = nn.Parameter(torch.zeros(num_tilts, dtype=real_dtype), requires_grad=False)
        self._pose_shifts = nn.Parameter(
            torch.zeros(num_tilts, 2, dtype=real_dtype), requires_grad=False
        )
        # Pose-refinement state (phase 1: shifts). ``_pose_shifts_init`` is the baseline
        # ``reset()`` returns to and the value the reference tilt is pinned at; the reference
        # tilt (argmin |tilt|) is the translation gauge -- a constant added to every tilt's shift
        # is exactly an object translation, so its row never moves. ``_pose_shifts_accum``
        # accumulates the (rank-averaged) gradient over M batches for one pose step.
        self.register_buffer("_pose_shifts_init", torch.zeros(num_tilts, 2, dtype=real_dtype))
        self.register_buffer("_pose_shifts_accum", torch.zeros(num_tilts, 2, dtype=real_dtype))
        self._learn_pose_shifts: bool = False
        self._pose_shifts_active: bool = False  # gather shifts_A in forward (learning or nonzero)
        self._reference_tilt_idx: int = int(np.argmin(np.abs(angles)))
        self._pose_steps_per_iter: int = 0  # 0 = per-batch steps (M = 1)
        self._pose_accum_steps: int = 1  # M: batches accumulated per pose step
        self._pose_accum_count: int = 0
        self._pose_step_count: int = 0
        self._pose_lr_hold: int = 0
        self._pose_lr_decay: float = 1.0
        self._pose_lr_floor: float = 0.0
        self._pose_lr_warmup: int = 0
        self._pose_lr_base: float | None = None  # LR at the first pose step (warm-up target)
        # diagnostics of the last pose step: rms / max |grad| over the non-reference rows (the
        # gradient that Adam saw, i.e. after accumulation), for the per-iteration history
        self._pose_last_grad_rms: float = 0.0
        self._pose_last_grad_max: float = 0.0
        # Beam-frame coordinate origin in scan-grid pixels (set at preprocess): the center of
        # the scan grid, anchored to the specimen-box center. Coordinates are emitted in Å
        # relative to this point (see PtychoTomoPatchData).
        self.register_buffer("_scan_center_px", torch.full((2,), torch.nan, dtype=real_dtype))

    @staticmethod
    def _tilt_array_3d(ds: PtychographyDatasetRaster) -> np.ndarray:
        arr = ds.dset.array if ds.dset.array is not None else ds.dset.tensor.cpu().numpy()
        return np.asarray(arr)

    # region --- factories ---
    @classmethod
    def from_datasets(
        cls,
        tilt_datasets: list[PtychographyDatasetRaster],
        tilt_angles_deg: "np.ndarray | torch.Tensor | list[float]",
        detector_mask: torch.Tensor | np.ndarray | None = None,
        verbose: int | bool = 1,
    ) -> "PtychoTomoDatasetRaster":
        return cls(
            tilt_datasets=tilt_datasets,
            tilt_angles_deg=tilt_angles_deg,
            detector_mask=detector_mask,
            verbose=verbose,
            _token=cls._token,
        )

    @classmethod
    def from_dataset4dstem_list(
        cls,
        dsets: list[Dataset4dstem],
        tilt_angles_deg: "np.ndarray | torch.Tensor | list[float]",
        detector_mask: torch.Tensor | np.ndarray | None = None,
        verbose: int | bool = 1,
    ) -> "PtychoTomoDatasetRaster":
        """Wrap a list of per-tilt ``Dataset4dstem`` (ordered like ``tilt_angles_deg``)."""
        tilt_datasets = [
            PtychographyDatasetRaster.from_dataset4dstem(
                d, verbose=False, learn_descan=False, learn_scan_positions=False
            )
            for d in dsets
        ]
        return cls.from_datasets(
            tilt_datasets=tilt_datasets,
            tilt_angles_deg=tilt_angles_deg,
            detector_mask=detector_mask,
            verbose=verbose,
        )

    # endregion --- factories ---

    # region --- tilt geometry ---
    @property
    def num_tilts(self) -> int:
        return len(self.tilt_datasets)

    @property
    def tilt_angles_deg(self) -> torch.Tensor:
        return self._tilt_angles_deg

    @property
    def tilt_offsets(self) -> torch.Tensor:
        """Flat-index offsets of each tilt, shape ``(num_tilts + 1,)``."""
        return self._tilt_offsets

    def tilt_index_of(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """Map flat scan indices to tilt indices, shape-preserving."""
        return torch.bucketize(batch_indices, self._tilt_offsets[1:], right=True)

    def rotations(self) -> torch.Tensor:
        """Current beam->specimen rotation matrices per tilt, ``(num_tilts, 3, 3)``."""
        return rot_beam_to_spec(
            self._pose_z1,
            self._tilt_angles_deg + self._pose_dtheta,
            self._pose_z3,
            device=self._tilt_angles_deg.device,
            dtype=self._tilt_angles_deg.dtype,
        )

    # endregion --- tilt geometry ---

    # region --- pose refinement (phase 1: per-tilt beam-frame shifts) ---
    # Convention: ``_pose_shifts[t] = (dy, dx)`` in Å, beam frame, SUBTRACTED from the patch
    # coordinates before rotation (object_models.ObjectPtychoTomoBase.forward); a specimen
    # physically displaced by +d in the beam frame is recovered as shift = +d. Gauge: the
    # reference tilt's row is pinned to its baseline (gradient masked after the DDP all-reduce,
    # value re-copied after every optimizer step).
    @property
    def learn_pose_shifts(self) -> bool:
        return self._learn_pose_shifts

    def set_learn_pose_shifts(self, enabled: bool) -> None:
        """Make the per-tilt shifts learnable (adds the ``pose_shifts`` optimizer group).

        The three angle slots are re-frozen here unconditionally: phase 1 learns shifts only.
        Disabling with an optimizer attached removes it (a learned table stays applied).
        """
        enabled = bool(enabled)
        self._learn_pose_shifts = enabled
        self._pose_shifts.requires_grad_(enabled)
        for p in (self._pose_z1, self._pose_dtheta, self._pose_z3):
            p.requires_grad_(False)
        self._refresh_pose_shifts_active()
        if not enabled and self.has_optimizer():
            self.remove_optimizer()

    def _refresh_pose_shifts_active(self) -> None:
        # one device sync per call (never per forward): the gather is only skipped when nothing
        # could change the payload, which keeps the no-pose path bit-identical
        self._pose_shifts_active = self._learn_pose_shifts or bool(
            (self._pose_shifts.detach() != 0).any().item()
        )

    @property
    def reference_tilt_idx(self) -> int:
        """Index of the gauge tilt whose shift never moves (default ``argmin |tilt|``)."""
        return self._reference_tilt_idx

    @reference_tilt_idx.setter
    def reference_tilt_idx(self, idx: int) -> None:
        idx = int(idx)
        if not 0 <= idx < self.num_tilts:
            raise ValueError(f"reference_tilt_idx {idx} out of range for {self.num_tilts} tilts")
        self._reference_tilt_idx = idx
        if bool((self._pose_shifts_init[idx] != 0).any().item()):
            warnings.warn(
                f"reference tilt {idx} has a nonzero shift baseline "
                f"{self._pose_shifts_init[idx].tolist()} Å; it is pinned there (gauge) and will "
                "not be learned",
                stacklevel=2,
            )

    def set_pose_shift_init(self, shifts_A: "np.ndarray | torch.Tensor | list[Any]") -> None:
        """Set the per-tilt shift baseline ``(num_tilts, 2)`` in Å and start the live table there.

        This is both the "start the search here" knob (an external initialisation) and the
        deliberate-perturbation knob (recovery tests). The reference tilt's row is forced to
        zero (gauge). ``reset()`` returns to this baseline, not to zero.
        """
        t = torch.as_tensor(
            np.asarray(shifts_A, dtype=float),
            dtype=self._pose_shifts.dtype,
            device=self._pose_shifts.device,
        )
        if t.shape != (self.num_tilts, 2):
            raise ValueError(
                f"shifts_A must have shape ({self.num_tilts}, 2), got {tuple(t.shape)}"
            )
        ref = self._reference_tilt_idx
        if bool((t[ref] != 0).any().item()):
            warnings.warn(
                f"reference tilt {ref} shift init {t[ref].tolist()} Å ignored (gauge: zeroed)",
                stacklevel=2,
            )
            t = t.clone()
            t[ref] = 0.0
        with torch.no_grad():
            self._pose_shifts_init.copy_(t)
            self._pose_shifts.copy_(t)
            self._pose_shifts_accum.zero_()
        self._pose_accum_count = 0
        self._refresh_pose_shifts_active()

    @property
    def pose_shifts_A(self) -> torch.Tensor:
        """Detached copy of the live per-tilt shift table, ``(num_tilts, 2)`` Å."""
        return self._pose_shifts.detach().clone()

    @property
    def pose_shifts_init_A(self) -> torch.Tensor:
        return self._pose_shifts_init.detach().clone()

    def set_pose_accum(self, steps_per_iter: int, batches_per_epoch: int | None = None) -> None:
        """Accumulate the shift gradient over ``M`` batches per pose step.

        ``steps_per_iter == 0`` steps every batch (``M = 1``). ``steps_per_iter >= 1`` takes that
        many pose steps per epoch: ``M = ceil(batches_per_epoch / steps_per_iter)``, and the
        epoch-end ``flush_pose_accum()`` takes the last (possibly shorter) step. The dataset
        cannot know the loader length, so ``batches_per_epoch`` is REQUIRED in that mode — a
        silent fallback here ran a whole campaign at the wrong setting once.
        """
        steps = int(steps_per_iter)
        if steps < 0:
            raise ValueError("steps_per_iter must be >= 0")
        if steps == 0:
            m = 1
        else:
            if batches_per_epoch is None:
                raise ValueError(
                    f"set_pose_accum(steps_per_iter={steps}) needs batches_per_epoch (the "
                    "per-rank training loader length); pass steps_per_iter=0 for per-batch steps"
                )
            b = int(batches_per_epoch)
            if b < 1:
                raise ValueError("batches_per_epoch must be >= 1")
            m = max(1, -(-b // steps))
        self._pose_steps_per_iter = steps
        self._pose_accum_steps = m
        self._pose_accum_count = 0
        with torch.no_grad():
            self._pose_shifts_accum.zero_()

    @property
    def pose_accum_steps(self) -> int:
        """Batches accumulated per pose step (``M``)."""
        return self._pose_accum_steps

    @property
    def pose_step_count(self) -> int:
        """Pose steps taken since the last ``reset()``."""
        return self._pose_step_count

    @property
    def pose_last_grad_stats(self) -> tuple[float, float]:
        """``(rms, max)`` of the shift gradient used by the most recent pose step (Å⁻¹ units of
        the loss). Adam's step is ~lr only while this is large against its ``eps`` (1e-8)."""
        return self._pose_last_grad_rms, self._pose_last_grad_max

    def set_pose_lr_schedule(
        self,
        hold_steps: int = 0,
        decay: float = 1.0,
        floor: float = 0.0,
        warmup_steps: int = 0,
    ) -> None:
        """Per-pose-step LR schedule: ramp linearly from ``lr / warmup_steps`` to the optimizer's
        LR over the first ``warmup_steps`` steps (so the first Adam steps cannot outrun a small
        gradient — the unfreeze kick of 2026-08-27), hold ``hold_steps`` steps, then multiply by
        ``decay`` after every step down to ``floor``. Dataset-owned because the model schedulers
        step per epoch and Adam's step is ~lr regardless of the gradient, so the LR is a travel
        budget in Å."""
        if not 0.0 < float(decay) <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if int(warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0")
        self._pose_lr_hold = int(hold_steps)
        self._pose_lr_decay = float(decay)
        self._pose_lr_floor = float(floor)
        self._pose_lr_warmup = int(warmup_steps)
        self._pose_lr_base = None

    def _pose_lr_before_step(self) -> None:
        """Warm-up: LR for pose step k (1-based) = base * min(1, k / warmup)."""
        if self._optimizer is None or self._pose_lr_warmup <= 0:
            return
        if self._pose_lr_base is None:
            self._pose_lr_base = float(self._optimizer.param_groups[0]["lr"])
        k = self._pose_step_count + 1
        if k <= self._pose_lr_warmup:
            lr = self._pose_lr_base * k / float(self._pose_lr_warmup)
            for pg in self._optimizer.param_groups:
                pg["lr"] = lr

    def _pose_lr_step(self) -> None:
        if self._optimizer is None or self._pose_lr_decay >= 1.0:
            return
        if self._pose_step_count <= self._pose_lr_hold + self._pose_lr_warmup:
            return
        for pg in self._optimizer.param_groups:
            pg["lr"] = max(self._pose_lr_floor, float(pg["lr"]) * self._pose_lr_decay)

    def step_optimizer(self) -> None:
        """Optimizer step with optional gradient accumulation and the reference-tilt gauge.

        Called once per batch AFTER the DDP all-reduce, so every rank accumulates the identical
        rank-averaged gradient and takes the identical step (no extra collective needed).
        """
        if self._optimizer is None:
            return
        if not self._learn_pose_shifts or self._pose_accum_steps <= 1:
            self._pose_step(mean_of=None)
            return
        g = self._pose_shifts.grad
        if g is not None:
            with torch.no_grad():
                self._pose_shifts_accum.add_(g)
        self._pose_accum_count += 1
        if self._pose_accum_count >= self._pose_accum_steps:
            self._pose_step(mean_of=self._pose_accum_count)

    def _pose_step(self, mean_of: int | None) -> None:
        if self._optimizer is None:
            return
        if mean_of is not None:
            self._pose_shifts.grad = self._pose_shifts_accum / float(max(1, mean_of))
        ref = self._reference_tilt_idx
        if self._learn_pose_shifts and self._pose_shifts.grad is not None:
            self._pose_shifts.grad[ref] = 0.0  # gauge: after the all-reduce, before the step
            g = self._pose_shifts.grad.detach()
            n = max(1, int(g.shape[0]) - 1)
            self._pose_last_grad_rms = float((g.square().sum() / (2 * n)).sqrt().item())
            self._pose_last_grad_max = float(g.abs().max().item())
        if self._learn_pose_shifts:
            self._pose_lr_before_step()
        self._optimizer.step()
        if self._learn_pose_shifts:
            with torch.no_grad():
                self._pose_shifts[ref] = self._pose_shifts_init[ref]  # re-pin (decay/momentum)
            self._pose_step_count += 1
            self._pose_lr_step()
        if mean_of is not None:
            with torch.no_grad():
                self._pose_shifts_accum.zero_()
            self._pose_accum_count = 0

    def flush_pose_accum(self) -> None:
        """Take the pose step on whatever has accumulated (call at every epoch boundary).

        With ``steps_per_iter >= 1`` this is the last step of the epoch. A residual much smaller
        than ``M`` means the ``batches_per_epoch`` given to ``set_pose_accum`` does not match
        the loader -- reported loudly, not absorbed.
        """
        if self._optimizer is None or self._pose_accum_count == 0:
            return
        if self._pose_steps_per_iter >= 1 and self._pose_accum_count < self._pose_accum_steps // 2:
            warnings.warn(
                f"flush_pose_accum: only {self._pose_accum_count} of M={self._pose_accum_steps} "
                "batches accumulated at the epoch boundary -- batches_per_epoch passed to "
                "set_pose_accum does not match the training loader",
                stacklevel=2,
            )
        self._pose_step(mean_of=self._pose_accum_count)

    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        """Learnable dataset groups: the base descan / scan-position groups plus ``pose_shifts``
        when shifts are being learned. This dict is also the DDP surface (broadcast + all-reduce
        iterate it), so adding the group here is the whole multi-GPU integration."""
        groups = super().get_optimization_parameters()
        if self._learn_pose_shifts:
            groups["pose_shifts"] = [self._pose_shifts]
        return groups

    def _normalize_optimizer_params(self, params):
        """Fan a single optimizer spec out to every learnable group (incl. ``pose_shifts``);
        an explicit PPLR dict passes through and is key-checked by ``set_optimizer``."""
        norm = OptimizerMixin._normalize_optimizer_params(self, params)
        if set(norm) == {self.DEFAULT_OPTIMIZER_KEY}:
            spec = norm[self.DEFAULT_OPTIMIZER_KEY]
            learnable = [
                key
                for key, on in (
                    ("descan", self.learn_descan),
                    ("scan_positions", self.learn_scan_positions),
                    ("pose_shifts", getattr(self, "_learn_pose_shifts", False)),
                )
                if on
            ]
            if not learnable and not isinstance(spec, OptimizerParams.NoneOptimizer):
                warnings.warn(
                    f"{type(self).__name__}: an optimizer was requested but nothing is learnable "
                    "(learn_descan, learn_scan_positions and learn_pose_shifts are all False); "
                    "the optimizer will be removed.",
                    stacklevel=2,
                )
            return {key: replace(spec) for key in learnable} if learnable else {}
        return norm

    # endregion --- pose refinement ---

    # region --- per-tilt scan geometry (identical across tilts; raster-level properties) ---
    @property
    def gpts(self) -> np.ndarray:
        """Scan grid of ONE tilt (identical across tilts). Note ``num_gpts == num_tilts *
        prod(gpts)`` for this wrapper."""
        return self.tilt_datasets[0].gpts

    @property
    def scan_sampling(self) -> np.ndarray:
        return self.tilt_datasets[0].scan_sampling

    @property
    def scan_units(self) -> list[str]:
        return self.tilt_datasets[0].scan_units

    @property
    def fov(self) -> np.ndarray:
        """Lateral field of view of one tilt's scan (Å)."""
        return self.scan_sampling * (self.gpts - 1)

    # endregion --- per-tilt scan geometry ---

    def preprocess(
        self,
        com_fit_function: Literal["none", "plane", "parabola", "constant", "no_shift"] = (
            "constant"
        ),
        force_com_rotation: float = 0.0,
        force_com_transpose: bool = False,
        bilinear: bool = False,
        obj_padding_px: tuple[int, int] | np.ndarray = (0, 0),
        probe_energy: float | None = None,
        free_per_tilt_arrays: bool = True,
        vectorized: bool = True,
    ) -> None:
        """Preprocess every tilt and aggregate into the flat inherited storage.

        ``force_com_rotation`` / ``force_com_transpose`` default to 0 / False and are applied to
        every tilt: per-tilt *solved* CoM rotations would scramble the cross-tilt geometry, so a
        common (usually zero, for simulated data) diffraction-frame rotation is required.
        ``mean_diffraction_intensity`` is the position-weighted global mean — batches mix tilts
        and the loss normalization must be one scalar.

        ``free_per_tilt_arrays`` drops each tilt's large arrays (raw 4D copy, amplitudes,
        targets, patch indices) after aggregation, keeping only geometry/CoM metadata; rebuilding
        requires re-wrapping the original ``Dataset4dstem`` objects.
        """
        self._preprocessing_params = {
            "com_fit_function": com_fit_function,
            "force_com_rotation": force_com_rotation,
            "force_com_transpose": force_com_transpose,
            "bilinear": bilinear,
            "obj_padding_px": tuple(int(p) for p in np.asarray(obj_padding_px).ravel()),
            "probe_energy": probe_energy,
            "free_per_tilt_arrays": free_per_tilt_arrays,
            "vectorized": vectorized,
        }
        if force_com_rotation is None or force_com_transpose is None:
            raise ValueError(
                "force_com_rotation / force_com_transpose must be explicit for a tilt series "
                "(per-tilt solved rotations would break the shared geometry); use 0.0 / False "
                "for simulated data."
            )
        for ds in self.tilt_datasets:
            ds.preprocess(
                com_fit_function=com_fit_function,
                force_com_rotation=force_com_rotation,
                force_com_transpose=force_com_transpose,
                bilinear=bilinear,
                obj_padding_px=obj_padding_px,
                plot_rotation=False,
                plot_com=False,
                vectorized=vectorized,
                probe_energy=probe_energy,
            )

        if probe_energy is not None:
            self.probe_energy = probe_energy
        self.amplitudes = torch.cat([ds.amplitudes for ds in self.tilt_datasets])
        self.centered_amplitudes = torch.cat([ds.centered_amplitudes for ds in self.tilt_datasets])
        self.intensities = torch.cat([ds.intensities for ds in self.tilt_datasets])
        self.centered_intensities = torch.cat(
            [ds.centered_intensities for ds in self.tilt_datasets]
        )
        descan = torch.cat([ds.descan_shifts.data for ds in self.tilt_datasets])
        self.descan_shifts = descan
        self.initial_descan_shifts = descan.clone()

        counts = np.array([ds.num_gpts for ds in self.tilt_datasets], dtype=float)
        self.mean_diffraction_intensity = float(
            np.average(
                [ds.mean_diffraction_intensity for ds in self.tilt_datasets], weights=counts
            )
        )
        self.mean_diffraction_amplitude = float(
            np.average(
                [ds.mean_diffraction_amplitude for ds in self.tilt_datasets], weights=counts
            )
        )
        ref = self.tilt_datasets[0]
        self._pattern_crop_mask = ref._pattern_crop_mask
        self._pattern_crop_mask_shape = ref._pattern_crop_mask_shape
        self.com_rotation_rad = float(force_com_rotation)
        self.com_transpose = bool(force_com_transpose)

        self._set_initial_scan_positions_px(obj_padding_px)
        self._set_scan_center(obj_padding_px)
        self._set_targets("amplitude")
        if free_per_tilt_arrays:
            self._free_per_tilt_arrays()
        self._preprocessed = True

    def _set_initial_scan_positions_px(
        self,
        obj_padding_px: np.ndarray | tuple[int, ...] | None,
        positions_mask: np.ndarray | None = None,
    ) -> None:
        """Delegate per tilt (identical geometry -> identical per-tilt positions), then
        concatenate into the inherited flat parameters."""
        for ds in self.tilt_datasets:
            ds._set_initial_scan_positions_px(obj_padding_px, positions_mask)
        positions = torch.cat(
            [ds.scan_positions_px.data.detach().cpu() for ds in self.tilt_datasets]
        )
        self.scan_positions_px = positions
        self.initial_scan_positions_px = self.scan_positions_px.data.clone()

    def _set_scan_center(self, obj_padding_px: "np.ndarray | tuple[int, int]") -> None:
        """Set the beam-frame coordinate origin: the center of the scan grid at this padding.

        Positions carry the same padding offset, so the origin is padding-invariant in Å. Must
        be re-called if positions are re-derived at a different padding (the base
        ``obj_padding_px`` setter path does this via ``PtychoTomography.preprocess``).
        """
        full2d = self._obj_shape_full_2d(obj_padding_px)
        self._scan_center_px = torch.tensor(
            [(int(full2d[0]) - 1) / 2.0, (int(full2d[1]) - 1) / 2.0],
            dtype=self._scan_center_px.dtype,
            device=self._scan_center_px.device,
        )

    def _set_patch_indices(self, obj_padding_px: np.ndarray | tuple[int, ...]) -> None:
        """No-op: only implicit (coordinate-queried) object models are supported."""
        return

    def patch_indices_need_update(self) -> bool:
        return False

    def _free_per_tilt_arrays(self) -> None:
        """Drop per-tilt heavy arrays after aggregation (geometry/CoM metadata retained)."""
        for ds in self.tilt_datasets:
            ds._intensities_4d = np.empty((0, 0, 0, 0), dtype=np.float32)
            # ``_amplitudes`` is lazy upstream (recomputed from centered_amplitudes on demand),
            # so None -- not an empty tensor, which the property would hand back as real data.
            ds._amplitudes = None
            ds._centered_amplitudes = torch.empty(0)
            ds._intensities = torch.empty(0)
            ds._centered_intensities = torch.empty(0)
            ds._targets = torch.empty(0)
            ds._patch_indices = torch.empty(0, dtype=torch.int32)

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride] -- intentional payload seam
        self,
        batch_indices: np.ndarray | torch.Tensor,
        obj_padding_px: np.ndarray | tuple[int, ...],
    ) -> tuple[PtychoTomoPatchData, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Build the rotation-carrying object-query payload for a mixed-tilt batch.

        Returns the same 4-tuple as the single-scan dataset: ``(payload, positions_px,
        positions_px_fractional, descan_shifts)`` with zeroed fractional positions (baked into the
        coordinates, as for any implicit object) and ``descan_shifts=None`` (frozen in v1; the
        targets are descan-centered). The first element is the opaque object-query payload the
        reconstruction loop forwards to the (rotation-aware) object model; it carries the
        per-position beam-frame pose shift (``shifts_A``) whenever one is active.
        """
        if not self._implicit_object:
            raise RuntimeError(
                "PtychoTomoDatasetRaster requires an implicit (coordinate-queried) object model; "
                "pair it with an ObjectPtychoTomoBase subclass."
            )
        self.apply_hard_constraints(obj_padding_px)
        bidx = torch.as_tensor(batch_indices).to(self._tilt_offsets.device)
        positions_px = self.scan_positions_px[bidx]
        coords_A = self._scan_coords_A(bidx)
        tilt_idx = self.tilt_index_of(bidx)
        rotations = self.rotations()[tilt_idx]
        # per-tilt -> per-position gather of the beam-frame shift; None (bit-identical to the
        # pre-pose path) unless shifts are being learned or a nonzero table is loaded
        shifts_A = self._pose_shifts[tilt_idx] if self._pose_shifts_active else None
        window_dz = self._window_dz_A(bidx, rotations, shifts_A) if self.slab_window else None
        # transient per-batch stash for the reconstruction loop's probe pre-propagation
        # (PtychoTomography.forward_operator consumes it exactly once and clears it)
        self._last_window_dz_A = window_dz
        payload = PtychoTomoPatchData(
            coords_yx_A=coords_A,
            rotations=rotations,
            tilt_indices=tilt_idx,
            shifts_A=shifts_A,
            window_dz_A=window_dz,
        )
        return payload, positions_px, torch.zeros_like(positions_px), None

    @property
    def slab_window(self) -> bool:
        """Slab-window mode flag (plan-view, 2026-07-16): per-scan-position slab-stack offsets.

        Opt-in via :meth:`set_slab_window`; must be paired with the object model's
        ``set_slab_extent_A`` and the reconstruction loop's probe pre-propagation.
        """
        buf = self._buffers.get("_slab_window_flag")
        return bool(buf.item()) if buf is not None else False

    def set_slab_window(self, enabled: bool) -> None:
        """Enable/disable per-scan-position slab-window offsets (survives ``reset()``)."""
        if "_slab_window_flag" not in self._buffers:
            self.register_buffer(
                "_slab_window_flag", torch.zeros((), dtype=torch.bool, device=self.device)
            )
        with torch.no_grad():
            self._slab_window_flag.fill_(bool(enabled))

    def _window_dz_A(
        self,
        batch_indices: torch.Tensor,
        rotations: torch.Tensor,
        shifts_A: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-batch slab-window offset along the beam, ``(B,)`` in Å.

        Places the window center on the specimen mid-plane (``z_spec = 0``) at each scan
        position: with beam→specimen rotation ``r``, the beam-frame z of that plane under the
        beam center ``(y_c, x_c)`` solves ``r00·z + r01·y_c + r02·x_c = 0``. Convention-free
        (no tilt-sign special case) and exact for any pose; requires ``|r00| = |cos(tilt)|``
        bounded away from 0 (asserts tilt < ~84°). ``shifts_A`` ``(B, 2)`` is the per-position
        beam-frame pose shift, subtracted from the beam center exactly as the object model
        subtracts it from the patch coordinates, so the window follows the shifted specimen.
        """
        center = self._scan_center_px
        samp = self.obj_sampling
        positions = self.scan_positions_px[batch_indices]  # (B, 2) un-rounded px
        y_c = (positions[:, 0] - center[0]) * float(samp[0])
        x_c = (positions[:, 1] - center[1]) * float(samp[1])
        if shifts_A is not None:
            y_c = y_c - shifts_A[:, 0]
            x_c = x_c - shifts_A[:, 1]
        r = rotations.to(device=y_c.device, dtype=y_c.dtype)
        r00 = r[:, 0, 0]
        assert bool((r00.abs() > 0.1).all()), (
            "slab-window mode needs |cos(tilt)| > 0.1 (tilt < ~84°); got a near-grazing pose"
        )
        return -(r[:, 0, 1] * y_c + r[:, 0, 2] * x_c) / r00

    def _scan_coords_A(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """Physical beam-frame ``(row, col)`` patch coordinates in Å, ``(B, Hroi, Wroi, 2)``.

        Same construction as the base ``_scan_coords`` (un-rounded positions + fftfreq ROI
        offsets, no wrap) but emitted in Å relative to ``_scan_center_px`` (the scan-grid
        center == specimen-box center) instead of normalized over a padded pixel grid — the
        object model owns the Å -> [-1, 1] support normalization.
        """
        center = self._scan_center_px
        if bool(torch.isnan(center).any()):
            raise RuntimeError("scan center not set; run preprocess() first")
        positions = self.scan_positions_px[batch_indices]  # (batch, 2), un-rounded px
        hroi, wroi = int(self.roi_shape[0]), int(self.roi_shape[1])
        r_ind = torch.fft.fftfreq(hroi, d=1 / hroi).to(self.device)
        c_ind = torch.fft.fftfreq(wroi, d=1 / wroi).to(self.device)
        rows = positions[:, 0][:, None, None] + r_ind[None, :, None]  # (batch, Hroi, 1)
        cols = positions[:, 1][:, None, None] + c_ind[None, None, :]  # (batch, 1, Wroi)
        rows = rows.expand(-1, -1, wroi)
        cols = cols.expand(-1, hroi, -1)
        samp = self.obj_sampling
        rows_A = (rows - center[0]) * float(samp[0])
        cols_A = (cols - center[1]) * float(samp[1])
        return torch.stack([rows_A, cols_A], dim=-1)  # (batch, Hroi, Wroi, 2), Å

    def reset(self) -> None:
        """Restore positions/descan, zero the (frozen) angle slots and return the shifts to
        their baseline (``set_pose_shift_init``, zeros by default). The learn flag, reference
        tilt, accumulation and LR-schedule settings survive, like ``set_slab_window``."""
        super().reset()
        with torch.no_grad():
            self._pose_z1.zero_()
            self._pose_dtheta.zero_()
            self._pose_z3.zero_()
            self._pose_shifts.copy_(self._pose_shifts_init)
            self._pose_shifts_accum.zero_()
        self._pose_accum_count = 0
        self._pose_step_count = 0
        self._pose_lr_base = None
        self._refresh_pose_shifts_active()


PtychoTomoDatasetType = PtychoTomoDatasetRaster
