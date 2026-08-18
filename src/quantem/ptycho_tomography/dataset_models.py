"""Multi-tilt 4D-STEM dataset model for joint ptycho-tomography.

``PtychoTomoDatasetRaster`` aggregates a tilt series of per-tilt ``PtychographyDatasetRaster``
instances into the flat ``(tilt x scan position)`` index space the ptychography reconstruction
loop expects, by subclassing the same base and concatenating into the *inherited* storage
(targets, scan-position / descan parameters, amplitudes). Mixed-tilt shuffled batches, the
DataLoader protocol, multi-GPU samplers, and ``error_estimate`` then work unchanged.

Per-tilt geometry (tilt angle + pose-correction slots, frozen in v1) lives here, mirroring the
tomography module's design: ``forward`` returns a :class:`PtychoTomoPatchData` payload carrying
beam-frame patch coordinates and per-element beam->specimen rotation matrices, which the
rotation-aware object models consume. Only implicit (coordinate-queried) object models are
supported — there is no integer patch-index path through a rotated volume.
"""

import warnings
from dataclasses import replace
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn

from quantem.core import config
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.ml.optimizer_mixin import (
    OptimizerMixin,
    OptimizerParams,
    OptimizerParamsType,
)
from quantem.diffractive_imaging.dataset_models import (
    DatasetConstraints,
    PtychographyDatasetRaster,
)
from quantem.ptycho_tomography.geometry import PtychoTomoPatchData, rot_beam_to_spec

# A pose/defocus baseline value: a scalar (same at every tilt), a per-tilt sequence, or (for
# the 2-component shifts) a per-tilt sequence of pairs / a single pair.
PoseInitValue = float | Sequence[float] | Sequence[Sequence[float]] | np.ndarray | torch.Tensor


def _drop_attr(obj: nn.Module, name: str) -> None:
    """Delete ``name`` from ``obj`` if it is set (no-op otherwise)."""
    if name in obj.__dict__:
        delattr(obj, name)


class PtychoTomoDatasetRaster(DatasetConstraints):
    """Tilt series of raster 4D-STEM scans presented as one flat ptychography dataset.

    Construction wraps each tilt in a :class:`PtychographyDatasetRaster` (kept in
    ``self.tilt_datasets`` as preprocessing containers); the concatenated diffraction stack backs
    the inherited storage, so ``num_gpts = sum(per-tilt scan points)`` and all per-position
    parameters/targets are flat-indexed. ``_tilt_offsets`` maps flat indices to tilts.

    v1 scope: identical scan geometry across tilts, known tilt angles (pose-correction parameters
    exist but are frozen at zero), no descan / scan-position learning. Per-tilt CoM rotation is
    forced to zero — solving it per tilt would scramble the cross-tilt geometry.
    """

    # registered buffer / parameter types (mirrors the base's _patch_indices declaration)
    _tilt_offsets: torch.Tensor
    _tilt_angles_deg: torch.Tensor
    _slab_window_flag: torch.Tensor
    # transient per-batch stashes, consumed once by PtychoTomography.forward_operator, never
    # serialized non-None. ``_last_window_dz_A`` is the pure slab-window offset (also on the
    # payload, i.e. what moves the OBJECT query); ``_last_probe_dz_A`` is what the probe is
    # Fresnel pre-propagated by -- window offset MINUS the per-tilt defocus offset.
    _last_window_dz_A: torch.Tensor | None = None
    _last_probe_dz_A: torch.Tensor | None = None
    _scan_center_px: torch.Tensor
    _rot_axis_offset_A: torch.Tensor
    _pose_z1_init: torch.Tensor
    _pose_z3_init: torch.Tensor
    _pose_dtheta_init: torch.Tensor
    _pose_shifts_init: torch.Tensor
    _defocus_offset_init_A: torch.Tensor
    _pose_z1: nn.Parameter
    _pose_dtheta: nn.Parameter
    _pose_z3: nn.Parameter
    _pose_shifts: nn.Parameter
    _defocus_offset_A: nn.Parameter
    _learn_pose_shifts: bool
    _learn_pose_z1: bool
    _learn_pose_z3: bool
    _learn_pose_dtheta: bool
    _learn_defocus: bool
    _reference_tilt_idx: int
    _defocus_active: bool
    _pose_accum_M: int
    _pose_accum_count: int
    _pose_steps_taken: int

    # the pose/defocus parameters, in one place (gauge masking + gradient accumulation)
    _POSE_PARAM_NAMES = (
        "_pose_shifts",
        "_pose_z1",
        "_pose_z3",
        "_pose_dtheta",
        "_defocus_offset_A",
    )
    # pose/defocus learn flags, in the order ``_ensure_pose_state`` materializes them
    _POSE_LEARN_FLAGS = (
        "_learn_pose_shifts",
        "_learn_pose_z1",
        "_learn_pose_z3",
        "_learn_pose_dtheta",
        "_learn_defocus",
    )

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
        # angles + beam-frame shifts) plus the per-tilt defocus offset. All default to frozen;
        # ``set_learn_pose`` / ``set_learn_defocus`` flip requires_grad and add the matching
        # optimizer groups (see get_optimization_parameters).
        num_tilts = len(tilt_datasets)
        self._pose_z1 = nn.Parameter(torch.zeros(num_tilts, dtype=real_dtype), requires_grad=False)
        self._pose_dtheta = nn.Parameter(
            torch.zeros(num_tilts, dtype=real_dtype), requires_grad=False
        )
        self._pose_z3 = nn.Parameter(torch.zeros(num_tilts, dtype=real_dtype), requires_grad=False)
        self._pose_shifts = nn.Parameter(
            torch.zeros(num_tilts, 2, dtype=real_dtype), requires_grad=False
        )
        self._defocus_offset_A = nn.Parameter(
            torch.zeros(num_tilts, dtype=real_dtype), requires_grad=False
        )
        # Beam-frame coordinate origin in scan-grid pixels (set at preprocess): the center of
        # the scan grid, anchored to the specimen-box center. Coordinates are emitted in Å
        # relative to this point (see PtychoTomoPatchData).
        self.register_buffer("_scan_center_px", torch.full((2,), torch.nan, dtype=real_dtype))
        # Baseline pose/defocus the parameters reset to (default zeros). set_tilt_axis_pose()
        # uses the z1/z3 baselines to fix a dataset-wide tilt-AXIS convention (e.g. z1=-90,
        # z3=+90 turns the ZXZ x-tilt into a tilt about y — the ASE-simulated AuNP series'
        # convention, found 2026-07-13); set_pose_init / set_defocus_init set the rest.
        self.register_buffer("_pose_z1_init", torch.zeros(num_tilts, dtype=real_dtype))
        self.register_buffer("_pose_z3_init", torch.zeros(num_tilts, dtype=real_dtype))
        self.register_buffer("_pose_dtheta_init", torch.zeros(num_tilts, dtype=real_dtype))
        self.register_buffer("_pose_shifts_init", torch.zeros(num_tilts, 2, dtype=real_dtype))
        self.register_buffer("_defocus_offset_init_A", torch.zeros(num_tilts, dtype=real_dtype))
        for _flag in self._POSE_LEARN_FLAGS:
            setattr(self, _flag, False)
        # Gauge: the reference tilt (smallest |tilt|) is pinned; see set_learn_pose().
        self._reference_tilt_idx = int(np.argmin(np.abs(angles)))
        self._defocus_active = False
        # Pose-gradient accumulation (see set_pose_accum): M = 1 is the per-batch default.
        self._pose_accum_M = 1
        self._pose_accum_count = 0
        self._pose_accum: dict[str, torch.Tensor] | None = None
        self._pose_steps_taken = 0
        self._pose_step_log: list[dict] = []

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
        self._ensure_pose_state()
        return rot_beam_to_spec(
            self._pose_z1,
            self._tilt_angles_deg + self._pose_dtheta,
            self._pose_z3,
            device=self._tilt_angles_deg.device,
            dtype=self._tilt_angles_deg.dtype,
        )

    # endregion --- tilt geometry ---

    # region --- pose / defocus refinement ---
    def _ensure_pose_state(self) -> None:
        """Materialize the pose/defocus parameters, baselines and flags; fold legacy state.

        Objects deserialized from a save/cache written before a given slot existed bypass
        ``__init__`` entirely (gotcha #23), so every pose access funnels through here. It also
        folds a legacy ``_rot_axis_offset_A`` buffer into ``_pose_shifts_init`` exactly once
        (see :meth:`set_rotation_center_offset_A`), so old caches keep the geometry the numbers
        recorded with them were measured at.
        """
        real_dtype = getattr(torch, config.get("dtype_real"))
        n = self.num_tilts
        for name, shape in (
            ("_pose_z1", (n,)),
            ("_pose_dtheta", (n,)),
            ("_pose_z3", (n,)),
            ("_pose_shifts", (n, 2)),
            ("_defocus_offset_A", (n,)),
        ):
            if self._parameters.get(name) is None:
                setattr(
                    self,
                    name,
                    nn.Parameter(
                        torch.zeros(shape, dtype=real_dtype, device=self.device),
                        requires_grad=False,
                    ),
                )
        for init_name, param_name in (
            ("_pose_z1_init", "_pose_z1"),
            ("_pose_z3_init", "_pose_z3"),
            ("_pose_dtheta_init", "_pose_dtheta"),
            ("_pose_shifts_init", "_pose_shifts"),
            ("_defocus_offset_init_A", "_defocus_offset_A"),
        ):
            if init_name not in self._buffers:
                param: nn.Parameter = getattr(self, param_name)
                self.register_buffer(init_name, torch.zeros_like(param.data))
        for flag in self._POSE_LEARN_FLAGS:
            if flag not in self.__dict__:
                setattr(self, flag, False)
        if "_reference_tilt_idx" not in self.__dict__:
            self._reference_tilt_idx = int(torch.argmin(self._tilt_angles_deg.abs()).item())
        if "_pose_accum_M" not in self.__dict__:
            self._pose_accum_M = 1
        if "_pose_accum_count" not in self.__dict__:
            self._pose_accum_count = 0
        if "_pose_accum" not in self.__dict__:
            self._pose_accum = None
        if "_pose_steps_taken" not in self.__dict__:
            self._pose_steps_taken = 0
        if "_pose_step_log" not in self.__dict__:
            self._pose_step_log = []
        self._fold_legacy_rot_axis_offset()
        if "_defocus_active" not in self.__dict__:
            self._refresh_defocus_active()

    def _ensure_pose_init_buffers(self) -> None:
        """Back-compat alias for :meth:`_ensure_pose_state` (kept: external callers exist)."""
        self._ensure_pose_state()

    def _fold_legacy_rot_axis_offset(self) -> None:
        """Fold a legacy ``_rot_axis_offset_A`` buffer into the pose-shift baseline, once.

        The retired rotation-center hack subtracted a constant beam-frame ``(row, col)`` offset
        from the patch coordinates AND from the slab-window offset — exactly what a constant
        per-tilt ``_pose_shifts`` entry now does (the object model subtracts ``shifts_A`` from
        the beam-frame coordinates before rotating, ``object_models.py``). So the fold is an
        identity on the emitted geometry. The legacy buffer is zeroed afterwards, making this
        idempotent; nothing writes it any more.
        """
        self._ensure_rot_offset_buffer()
        off = self._rot_axis_offset_A
        if not bool((off != 0).any()):
            return
        with torch.no_grad():
            add = off.to(device=self._pose_shifts_init.device, dtype=self._pose_shifts_init.dtype)
            self._pose_shifts_init.add_(add.view(1, 2))
            self._pose_shifts.data.add_(add.view(1, 2))
            off.zero_()

    def _refresh_defocus_active(self) -> None:
        """Cache whether the probe needs the per-tilt defocus pre-propagation at all.

        Recomputed only when the defocus state changes (a ``.any()`` per batch would be a
        device sync on the hot path); ``learn_defocus`` forces it on because the parameter can
        leave zero at any optimizer step.
        """
        self._defocus_active = bool(self._learn_defocus) or bool(
            (self._defocus_offset_A.data != 0).any()
        )

    def _as_pose_tensor(
        self,
        value: PoseInitValue,
        shape: tuple[int, ...],
        name: str,
    ) -> torch.Tensor:
        """Broadcast a scalar / per-tilt array to a pose baseline's shape."""
        t = torch.as_tensor(value).to(device=self.device, dtype=self._tilt_angles_deg.dtype)
        if t.ndim == 0:
            t = t.expand(shape)
        elif len(shape) == 2 and tuple(t.shape) == (shape[1],):
            t = t.view(1, -1).expand(shape)
        if tuple(t.shape) != tuple(shape):
            raise ValueError(f"{name} must broadcast to {tuple(shape)}, got {tuple(t.shape)}")
        return t.contiguous()

    @property
    def reference_tilt_idx(self) -> int:
        """Index of the gauge-fixing reference tilt (default ``argmin |tilt|``)."""
        self._ensure_pose_state()
        return int(self._reference_tilt_idx)

    @reference_tilt_idx.setter
    def reference_tilt_idx(self, idx: int) -> None:
        self._ensure_pose_state()
        i = int(idx)
        if not 0 <= i < self.num_tilts:
            raise ValueError(f"reference_tilt_idx must be in [0, {self.num_tilts}), got {i}")
        self._reference_tilt_idx = i

    @property
    def learn_pose_shifts(self) -> bool:
        self._ensure_pose_state()
        return bool(self._learn_pose_shifts)

    @property
    def learn_pose_z1(self) -> bool:
        self._ensure_pose_state()
        return bool(self._learn_pose_z1)

    @property
    def learn_pose_z3(self) -> bool:
        self._ensure_pose_state()
        return bool(self._learn_pose_z3)

    @property
    def learn_pose_dtheta(self) -> bool:
        self._ensure_pose_state()
        return bool(self._learn_pose_dtheta)

    @property
    def learn_defocus(self) -> bool:
        self._ensure_pose_state()
        return bool(self._learn_defocus)

    def set_learn_pose(self, shifts: bool, z1: bool, z3: bool, dtheta: bool = False) -> None:
        """Enable per-tilt pose refinement; flips ``requires_grad`` on the pose parameters.

        Slots (all per tilt, all in this dataset model because the geometry is per tilt):

        - ``shifts`` — beam-frame ``(dy, dx)`` in Å, subtracted from the patch coordinates
          *before* rotation (and from the slab-window offset's beam center, identically).
        - ``z1`` / ``z3`` — the Z-X-Z Euler angles of ``rot_beam_to_spec(z1, tilt + dtheta, z3)``
          in degrees; this is the tomography module's pose parameterization.
        - ``dtheta`` — a per-tilt tilt-ANGLE correction in degrees. Opt-in (default off) and a
          deliberate deviation from ``quantem.tomography``, which holds the tilt angle fixed:
          goniometer readout errors are real on experimental data and are not representable by
          z1/z3.

        Gauge (tomography-module style)
        -------------------------------
        A free-form 3D object has 3 translation + 3 rotation gauge degrees of freedom: moving
        the object and moving every tilt's pose the opposite way leaves the data unchanged, so
        the joint problem is rank-deficient and the whole series would drift. The gauge is fixed
        by **pinning the reference tilt** (``reference_tilt_idx``, default ``argmin |tilt|``):
        its gradient is zeroed before each optimizer step and its pose is restored to its
        *baseline* after each step, for every slot including the defocus offset (see
        :meth:`step_optimizer`).

        Note that pinning loses no physical degree of freedom. The rotation-center error the
        retired :meth:`set_rotation_center_offset_A` corrected is a constant beam-frame shift
        ``s`` at every tilt; modulo the object-translation gauge that is equivalent to the
        per-tilt shifts ``s_t = s - [R_t^T (0, s_y, s_x)]_{y,x}`` (plus the corresponding
        beam-frame z component, which the defocus offset carries), which vanishes at the
        reference tilt and so is fully reachable with the reference pinned.

        The reference tilt is restored to its *baseline*, not hard-zeroed, so a nonzero
        baseline set by :meth:`set_tilt_axis_pose` (z1 = -90 / z3 = +90 for the archived AuNP
        v1-v4 series) or by the deprecated rotation-center wrapper stays intact; with the
        default zero baselines this is literally "re-zero the reference tilt".

        Survives ``reset()`` and is serialized (plain attributes on the module).
        """
        self._ensure_pose_state()
        self._learn_pose_shifts = bool(shifts)
        self._learn_pose_z1 = bool(z1)
        self._learn_pose_z3 = bool(z3)
        self._learn_pose_dtheta = bool(dtheta)
        self._pose_shifts.requires_grad_(bool(shifts))
        self._pose_z1.requires_grad_(bool(z1))
        self._pose_z3.requires_grad_(bool(z3))
        self._pose_dtheta.requires_grad_(bool(dtheta))

    def set_learn_defocus(self, enabled: bool) -> None:
        """Enable the per-tilt learnable defocus offset (see :meth:`set_defocus_init`)."""
        self._ensure_pose_state()
        self._learn_defocus = bool(enabled)
        self._defocus_offset_A.requires_grad_(bool(enabled))
        self._refresh_defocus_active()

    def set_pose_init(
        self,
        shifts: PoseInitValue | None = None,
        z1: PoseInitValue | None = None,
        z3: PoseInitValue | None = None,
        dtheta: PoseInitValue | None = None,
    ) -> None:
        """Set the pose BASELINE (what ``reset()`` restores) and copy it into the live pose.

        Each argument is a scalar (same value at every tilt), a per-tilt array
        (``(num_tilts,)``; ``(num_tilts, 2)`` or a single ``(2,)`` pair for ``shifts``), or
        ``None`` to leave that slot alone. This is both the "start the search here" knob and
        the perturbation knob for the recon-side validation (V1a): set a known nonzero pose,
        enable learning, and check it returns to the truth.
        """
        self._ensure_pose_state()
        n = self.num_tilts
        with torch.no_grad():
            if shifts is not None:
                self._pose_shifts_init.copy_(self._as_pose_tensor(shifts, (n, 2), "shifts"))
                self._pose_shifts.data.copy_(self._pose_shifts_init)
            if z1 is not None:
                self._pose_z1_init.copy_(self._as_pose_tensor(z1, (n,), "z1"))
                self._pose_z1.data.copy_(self._pose_z1_init)
            if z3 is not None:
                self._pose_z3_init.copy_(self._as_pose_tensor(z3, (n,), "z3"))
                self._pose_z3.data.copy_(self._pose_z3_init)
            if dtheta is not None:
                self._pose_dtheta_init.copy_(self._as_pose_tensor(dtheta, (n,), "dtheta"))
                self._pose_dtheta.data.copy_(self._pose_dtheta_init)

    def set_defocus_init(self, offsets_A: PoseInitValue) -> None:
        """Set the per-tilt defocus-offset baseline (Å) and copy it into the live parameter.

        Sign: ``effective defocus at tilt t = the probe model's defocus + offset_A[t]`` — see
        :meth:`forward`.
        """
        self._ensure_pose_state()
        with torch.no_grad():
            self._defocus_offset_init_A.copy_(
                self._as_pose_tensor(offsets_A, (self.num_tilts,), "offsets_A")
            )
            self._defocus_offset_A.data.copy_(self._defocus_offset_init_A)
        self._refresh_defocus_active()

    @property
    def pose_shifts_A(self) -> torch.Tensor:
        """Live per-tilt beam-frame shifts, ``(num_tilts, 2)`` Å."""
        self._ensure_pose_state()
        return self._pose_shifts

    @property
    def defocus_offsets_A(self) -> torch.Tensor:
        """Live per-tilt defocus offsets, ``(num_tilts,)`` Å."""
        self._ensure_pose_state()
        return self._defocus_offset_A

    _POSE_INIT_NAMES = (
        "_pose_shifts_init",
        "_pose_z1_init",
        "_pose_z3_init",
        "_pose_dtheta_init",
        "_defocus_offset_init_A",
    )

    def _reference_pose_slots(self) -> "tuple[tuple[nn.Parameter, torch.Tensor], ...]":
        """``(parameter, baseline)`` pairs the reference tilt is pinned across."""
        return tuple(
            (getattr(self, p), getattr(self, i))
            for p, i in zip(self._POSE_PARAM_NAMES, self._POSE_INIT_NAMES)
        )

    def zero_reference_pose_grads(self) -> None:
        """Zero the reference tilt's pose/defocus gradients (gauge; see :meth:`set_learn_pose`).

        Called from :meth:`step_optimizer` AFTER the DDP all-reduce, so every rank masks the
        same (already averaged) gradient and stays bit-identical.
        """
        self._ensure_pose_state()
        i = self.reference_tilt_idx
        for param, _init in self._reference_pose_slots():
            if param.grad is not None:
                param.grad[i] = 0.0

    def pin_reference_pose(self) -> None:
        """Restore the reference tilt's pose/defocus to its baseline (gauge)."""
        self._ensure_pose_state()
        i = self.reference_tilt_idx
        with torch.no_grad():
            for param, init in self._reference_pose_slots():
                param.data[i] = init[i]

    def set_pose_accum(
        self, steps_per_iter: int | None = 1, batches_per_epoch: int | None = None
    ) -> None:
        """Accumulate the pose/defocus gradient over M batches and step once per M.

        Why this is the standard path (2026-08-18). A mini-batch of scan positions spread over
        the whole tilt series gives each tilt only ``batch_size / num_tilts`` positions, and on
        a low-contrast specimen the resulting per-tilt shift gradient is dominated by *which*
        positions were drawn: measured on the phantom, median ``|mean| / std = 0.27`` over
        random batches, with the per-batch sign agreeing with the descent direction only
        45-70 % of the time. Adam follows the per-batch sign, so it random-walks by ``lr`` per
        step and nets ~2 % of it — independent of ``lr``. Accumulating first restores the
        signal-to-noise, and it is what the tomography module effectively does: its pose step
        sees a whole projection, not a slice of one. (The SNR is data dependent — atomic-
        resolution AuNP positions carry a reliable sign and converge with per-batch Adam — so
        this is a robustness fix, not a correction of something previously wrong.)

        Parameters
        ----------
        steps_per_iter
            Pose optimizer steps per epoch. ``1`` (default) = one step per epoch, i.e.
            ``M = batches_per_epoch``. ``K > 1`` = ``M = batches_per_epoch // K``.
            ``0`` or ``None`` = per-batch stepping (the pre-2026-08-18 behaviour).
        batches_per_epoch
            Length of the training loader. The dataset cannot know the batch size, so the
            runner passes it; if omitted, any previously set value is reused.

        The accumulator is kept SEPARATE from ``.grad`` because the reconstruction loop zeroes
        gradients at the start of every batch. Accumulation happens inside
        :meth:`step_optimizer`, i.e. AFTER the DDP all-reduce, so every rank accumulates the
        same averaged gradient and their sums agree; the reference-tilt mask is applied to the
        accumulated gradient just before the step.
        """
        self._ensure_pose_state()
        if not steps_per_iter:
            self._pose_accum_M = 1
        else:
            if batches_per_epoch is None:
                batches_per_epoch = int(getattr(self, "_pose_batches_per_epoch", 0)) or 1
            self._pose_batches_per_epoch = int(batches_per_epoch)
            self._pose_accum_M = max(1, int(batches_per_epoch) // max(1, int(steps_per_iter)))
        self._pose_accum = None
        self._pose_accum_count = 0

    @property
    def pose_accum_steps(self) -> int:
        """Batches accumulated per pose optimizer step (1 = step every batch)."""
        self._ensure_pose_state()
        return int(self._pose_accum_M)

    def zero_optimizer_grad(self) -> None:
        """Per-batch gradient zeroing — deliberately a plain pass-through.

        The accumulator lives outside ``.grad`` (see :meth:`set_pose_accum`), so the loop's
        per-batch zeroing is exactly what we want: every batch contributes its own clean
        gradient, which :meth:`step_optimizer` adds to the accumulator before it is discarded.
        Overridden only to make that contract explicit — moving the accumulation here would
        read the PREVIOUS batch's gradient, which is zeroed before the backward, not after it.
        """
        super().zero_optimizer_grad()

    def step_optimizer(self) -> None:
        """Optimizer step with the reference tilt's pose held at its baseline (gauge).

        Zeroing the gradient alone is not enough — a stateful optimizer (Adam moments, weight
        decay) can still move a parameter with a zero gradient — so the reference pose is also
        written back after the step.

        With :meth:`set_pose_accum` active (``M > 1``) this is called every batch but only
        *steps* every M-th call, using the MEAN of the M accumulated gradients.
        """
        if self._pose_accum_M <= 1:
            self._masked_step(n_accumulated=1)
            return

        params = {name: getattr(self, name) for name in self._POSE_PARAM_NAMES}
        if self._pose_accum is None:
            self._pose_accum = {}
        for name, param in params.items():
            if param.grad is None:
                continue
            acc = self._pose_accum.get(name)
            if acc is None or acc.shape != param.grad.shape or acc.device != param.grad.device:
                acc = torch.zeros_like(param.grad)
            self._pose_accum[name] = acc + param.grad.detach()
        self._pose_accum_count += 1
        if self._pose_accum_count < self._pose_accum_M:
            return

        for name, param in params.items():
            acc = self._pose_accum.get(name)
            if acc is None:
                continue
            param.grad = acc / float(self._pose_accum_count)
        self._masked_step(n_accumulated=self._pose_accum_count)
        self._pose_accum = None
        self._pose_accum_count = 0

    _POSE_STEP_LOG_MAX = 12

    def _masked_step(self, n_accumulated: int) -> None:
        """Gauge-mask, step, re-pin, and RECORD what the step actually did.

        The record exists because "the pose barely moved" is indistinguishable from "the pose
        never stepped" from the outside: Adam's first step is exactly ``lr * sign(g)`` per
        component, so a run whose parameters move by far less than ``lr`` per step is either
        sign-alternating or not stepping at all, and only the step count and the gradient that
        was actually handed to the optimizer separate the two (2026-08-18).
        """
        self.zero_reference_pose_grads()
        before = {n: getattr(self, n).detach().clone() for n in self._POSE_PARAM_NAMES}
        grad_max = 0.0
        for name in self._POSE_PARAM_NAMES:
            g = getattr(self, name).grad
            if g is not None:
                grad_max = max(grad_max, float(g.abs().max()))
        super().step_optimizer()
        self.pin_reference_pose()
        self._pose_steps_taken = int(getattr(self, "_pose_steps_taken", 0)) + 1
        if len(self._pose_step_log) < self._POSE_STEP_LOG_MAX:
            delta = (self._pose_shifts.detach() - before["_pose_shifts"]).cpu()
            self._pose_step_log.append(
                {
                    "step": self._pose_steps_taken,
                    "n_accumulated": int(n_accumulated),
                    "max_abs_grad": grad_max,
                    "max_abs_shift_delta": float(delta.abs().max()),
                    "shift_delta": delta.tolist(),
                }
            )

    def pose_step_stats(self, reset: bool = False) -> dict:
        """Steps taken so far, plus the per-step record of the first few steps."""
        self._ensure_pose_state()
        out = {
            "steps_taken": int(self._pose_steps_taken),
            "accum_batches_per_step": int(self._pose_accum_M),
            "pending_accumulated": int(self._pose_accum_count),
            "log": list(self._pose_step_log),
        }
        if reset:
            self._pose_steps_taken = 0
            self._pose_step_log = []
        return out

    def flush_pose_accum(self) -> bool:
        """Step now on a partial accumulation (end of an epoch that did not fill M)."""
        if self._pose_accum_M <= 1 or not self._pose_accum_count:
            return False
        for name in self._POSE_PARAM_NAMES:
            acc = (self._pose_accum or {}).get(name)
            if acc is not None:
                getattr(self, name).grad = acc / float(self._pose_accum_count)
        self._masked_step(n_accumulated=self._pose_accum_count)
        self._pose_accum = None
        self._pose_accum_count = 0
        return True

    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        """PPLR groups: the base's ``descan``/``scan_positions`` plus the pose/defocus groups.

        Groups appear only while their learn flag is on: ``pose_shifts``, ``pose_angles``
        (z1 + z3, and dtheta when enabled), ``defocus``. This dict IS the optimizer and DDP
        surface — ``PtychographyBase._broadcast_parameters`` and ``_all_reduce_gradients``
        (``ptychography_base.py:913-956``) both iterate exactly it, so overriding it here is
        the whole multi-GPU integration; nothing else needed checking.
        """
        self._ensure_pose_state()
        groups = super().get_optimization_parameters()
        if self._learn_pose_shifts:
            groups["pose_shifts"] = [self._pose_shifts]
        angles: list[torch.Tensor] = [
            param
            for param, on in (
                (self._pose_z1, self._learn_pose_z1),
                (self._pose_z3, self._learn_pose_z3),
                (self._pose_dtheta, self._learn_pose_dtheta),
            )
            if on
        ]
        if angles:
            groups["pose_angles"] = angles
        if self._learn_defocus:
            groups["defocus"] = [self._defocus_offset_A]
        return groups

    def _normalize_optimizer_params(
        self, params: "OptimizerParamsType | dict[str, object]"
    ) -> "dict[str, OptimizerParamsType]":
        """Fan a single optimizer spec out to whichever pose/defocus groups are enabled.

        Same contract as the base (``diffractive_imaging/dataset_models.py:195-238``) but over
        this class's group set; the group list comes from ``get_optimization_parameters()`` so
        the two can never disagree. An explicit PPLR dict passes through unchanged.
        """
        norm = OptimizerMixin._normalize_optimizer_params(self, params)
        if set(norm) != {self.DEFAULT_OPTIMIZER_KEY}:
            return norm
        spec = norm[self.DEFAULT_OPTIMIZER_KEY]
        learnable = list(self.get_optimization_parameters())
        if not learnable and not isinstance(spec, OptimizerParams.NoneOptimizer):
            warnings.warn(
                f"{type(self).__name__}: an optimizer was requested but nothing is learnable; "
                "the optimizer will be removed. Enable pose/defocus refinement with "
                "set_learn_pose(...) / set_learn_defocus(True).",
                stacklevel=2,
            )
        return {key: replace(spec) for key in learnable} if learnable else {}

    # endregion --- pose / defocus refinement ---

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

    # region --- back-compat shims (pre-merge caches) ---
    @property
    def com_transpose(self) -> bool:
        """Diffraction-frame transpose flag, with a pre-merge-cache fallback.

        Overrides the base getter only to run :meth:`_ensure_transpose_attr` first: this is the
        earliest point a deserialized wrap touches the flag (``PtychographyBase.__init__`` ->
        ``_obj_shape_full_2d`` -> ``_obj_shape_rot_2d`` -> ``_obj_shape_crop_2d``).
        """
        self._ensure_transpose_attr()
        return self._transpose

    @com_transpose.setter
    def com_transpose(self, t: bool) -> None:
        self._transpose = bool(t)

    def _ensure_transpose_attr(self) -> None:
        """Normalize the pre-merge ``_com_transpose`` attribute name onto ``_transpose``.

        Objects deserialized from saves/caches bypass ``__init__``, so a wrap written before the
        2026-08-17 ``diffractive_imaging`` merge restores the flag under its old name and the
        base getter finds neither name (an ``AttributeError`` that ``nn.Module.__getattr__``
        masks into a misleading missing-``_obj_shape_rot_2d`` message). Same situation, and same
        treatment, as :meth:`_ensure_pose_init_buffers`. Renames the key in place so a re-save
        writes the current name; the per-tilt datasets (pickled whole, so equally stale) are
        normalized alongside.
        """
        if "_transpose" in self.__dict__:
            return
        legacy = bool(getattr(self, "_com_transpose", False))
        self.com_transpose = legacy
        _drop_attr(self, "_com_transpose")
        for ds in getattr(self, "tilt_datasets", None) or []:
            if "_transpose" not in ds.__dict__:
                ds.com_transpose = bool(getattr(ds, "_com_transpose", legacy))
                _drop_attr(ds, "_com_transpose")

    # endregion --- back-compat shims ---

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
        reconstruction loop forwards to the (rotation-aware) object model.

        Pose and defocus
        ----------------
        ``payload.shifts_A`` carries the per-tilt beam-frame shift, which the object model
        subtracts from the patch coordinates before rotating; ``_window_dz_A`` subtracts the
        same shift from the beam center, so the two stay consistent.

        Two z-offsets are stashed for the reconstruction loop and they are NOT the same thing:

        - ``payload.window_dz_A`` / ``_last_window_dz_A`` — the pure slab-window offset. This
          moves the OBJECT query along the beam. The learned defocus must never appear here:
          defocus moves the probe, not the specimen.
        - ``_last_probe_dz_A`` — what the probe is Fresnel pre-propagated by, i.e.
          ``window_dz - defocus_offset[tilt]``.

        **Defocus sign rule:** ``effective defocus = probe-model defocus + _defocus_offset_A``.
        Propagating a probe forward by ``dz`` multiplies its Fourier transform by
        ``exp(-i pi lambda dz k^2)``, while defocus ``f`` enters the probe as
        ``exp(+i pi lambda f k^2)`` (``C10 = -f``, ``complex_probe.evaluate_probe``), so
        propagating forward by ``dz`` DECREASES the effective defocus by ``dz`` — hence the
        minus sign on the offset. Pinned by
        ``test_dataset_tomo.py::TestDefocusOffset::test_defocus_offset_sign_matches_probe``.
        """
        if not self._implicit_object:
            raise RuntimeError(
                "PtychoTomoDatasetRaster requires an implicit (coordinate-queried) object model; "
                "pair it with an ObjectPtychoTomoBase subclass."
            )
        self.apply_hard_constraints(obj_padding_px)
        self._ensure_pose_state()
        bidx = torch.as_tensor(batch_indices).to(self._tilt_offsets.device)
        positions_px = self.scan_positions_px[bidx]
        coords_A = self._scan_coords_A(bidx)
        tilt_idx = self.tilt_index_of(bidx)
        rotations = self.rotations()[tilt_idx]
        shifts = self._pose_shifts[tilt_idx]  # (B, 2) beam-frame Å, pre-rotation
        window_dz = self._window_dz_A(bidx, rotations, shifts) if self.slab_window else None
        probe_dz = window_dz
        if self._defocus_active:
            offset = self._defocus_offset_A[tilt_idx]  # (B,) Å
            probe_dz = -offset if window_dz is None else window_dz - offset
        # transient per-batch stashes for the reconstruction loop's probe pre-propagation
        # (PtychoTomography.forward_operator consumes them exactly once and clears them)
        self._last_window_dz_A = window_dz
        self._last_probe_dz_A = probe_dz
        payload = PtychoTomoPatchData(
            coords_yx_A=coords_A,
            rotations=rotations,
            tilt_indices=tilt_idx,
            shifts_A=shifts,
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
        bounded away from 0 (asserts tilt < ~84°).

        ``shifts_A`` (``(B, 2)``, the per-tilt pose shift) is subtracted from the beam center
        with the same sign the object model applies it to the patch coordinates — it used to be
        the constant ``_rot_axis_offset_A``, which is now folded into the pose shifts.
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
        # NOTE: the per-tilt pose shift is NOT applied here — it rides on the payload
        # (``shifts_A``) and the object model subtracts it before rotating, which keeps the
        # shift differentiable through one well-defined seam. The retired
        # ``_rot_axis_offset_A`` subtraction lived here; it is folded into the pose shifts.
        return torch.stack([rows_A, cols_A], dim=-1)  # (batch, Hroi, Wroi, 2), Å

    def _ensure_rot_offset_buffer(self) -> None:
        """Create the rotation-axis offset buffer when absent (cache-loaded objects bypass
        ``__init__``)."""
        if "_rot_axis_offset_A" not in self._buffers:
            real_dtype = getattr(torch, config.get("dtype_real"))
            self.register_buffer(
                "_rot_axis_offset_A", torch.zeros(2, dtype=real_dtype, device=self.device)
            )

    def set_rotation_center_offset_A(self, drow_A: float, dcol_A: float) -> None:
        """DEPRECATED (2026-08-17) — a thin wrapper over the pose shifts; use ``set_pose_init``.

        Was: shift the beam-frame coordinate origin (== the tilt-axis position) by a known
        offset in Å from the scan-grid center, because abTEM ``GridScan`` construction leaves
        the scan-pattern center short of the simulation cell center (the true rotation center)
        by up to half a scan step per axis — a rotation-center error producing arc/"banana"
        atom artifacts growing with scan step (0.30/0.49/0.75/0.99 Å at 0.6/1.0/1.5/2.0 Å
        steps). Pass the (row, col) offset FROM the scan center TO the true rotation center.

        Now: writes that offset as a CONSTANT beam-frame shift into ``_pose_shifts_init`` (all
        tilts), which the object model subtracts pre-rotation exactly as the old coordinate
        subtraction did — the emitted geometry is unchanged to float precision. Because the
        gauge pins the reference tilt to its *baseline* (not to hard zero), a constant nonzero
        baseline is legal and the learned shifts explore around it. Modulo the object-translation
        gauge the same geometry is also reachable as ``s_t = s - [R_t^T (0, s_y, s_x)]_{y,x}``,
        which vanishes at the reference tilt (PLAN §1); the constant form is used here because
        it reproduces the recorded pre-pose numbers exactly.

        Kept (not deleted) until V2a shows the learned shifts recover the offset on their own.
        """
        warnings.warn(
            "set_rotation_center_offset_A is deprecated; it now writes a constant beam-frame "
            "shift into the pose baseline. Use set_pose_init(shifts=(drow_A, dcol_A)) and, "
            "better, learn it with set_learn_pose(shifts=True, ...).",
            DeprecationWarning,
            stacklevel=2,
        )
        self._ensure_pose_state()
        with torch.no_grad():
            self._pose_shifts_init[:, 0] = float(drow_A)
            self._pose_shifts_init[:, 1] = float(dcol_A)
            self._pose_shifts.data.copy_(self._pose_shifts_init)

    def set_tilt_axis_pose(self, z1_deg: float, z3_deg: float) -> None:
        """Fix the series-wide tilt-axis convention via constant z1/z3 Euler offsets.

        ``rot_beam_to_spec(z1, tilt, z3)`` with constant ``z1=-90, z3=+90`` rotates about the
        specimen y axis instead of x — matching tilt series simulated with ASE ``atoms.rotate``
        (the ARCHIVED AuNP v1-v4 datasets; v5+ fix the tilt axis at the simulation source and
        must NOT use this). The values survive ``reset()`` (they define the dataset geometry,
        not a learned correction) and, being baselines, are what the pinned reference tilt is
        held at when pose refinement is on.
        """
        self._ensure_pose_state()
        with torch.no_grad():
            self._pose_z1_init.fill_(float(z1_deg))
            self._pose_z3_init.fill_(float(z3_deg))
            self._pose_z1.copy_(self._pose_z1_init)
            self._pose_z3.copy_(self._pose_z3_init)

    def reset(self) -> None:
        super().reset()
        self._ensure_pose_state()
        with torch.no_grad():
            self._pose_z1.copy_(self._pose_z1_init)
            self._pose_dtheta.copy_(self._pose_dtheta_init)
            self._pose_z3.copy_(self._pose_z3_init)
            self._pose_shifts.copy_(self._pose_shifts_init)
            self._defocus_offset_A.copy_(self._defocus_offset_init_A)
        self._refresh_defocus_active()
        self._pose_accum = None
        self._pose_accum_count = 0
        self._pose_steps_taken = 0
        self._pose_step_log = []


PtychoTomoDatasetType = PtychoTomoDatasetRaster
