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

from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from quantem.core import config
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
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

    v1 scope: identical scan geometry across tilts, known tilt angles (pose-correction parameters
    exist but are frozen at zero), no descan / scan-position learning. Per-tilt CoM rotation is
    forced to zero — solving it per tilt would scramble the cross-tilt geometry.
    """

    # registered buffer / parameter types (mirrors the base's _patch_indices declaration)
    _tilt_offsets: torch.Tensor
    _tilt_angles_deg: torch.Tensor
    _scan_center_px: torch.Tensor
    _rot_axis_offset_A: torch.Tensor
    _pose_z1_init: torch.Tensor
    _pose_z3_init: torch.Tensor
    _pose_z1: nn.Parameter
    _pose_dtheta: nn.Parameter
    _pose_z3: nn.Parameter
    _pose_shifts: nn.Parameter

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
        # angles + beam-frame shifts). Frozen in v1; enabling pose refinement later is a
        # requires_grad flip + an optimizer group, not an interface change.
        num_tilts = len(tilt_datasets)
        self._pose_z1 = nn.Parameter(torch.zeros(num_tilts, dtype=real_dtype), requires_grad=False)
        self._pose_dtheta = nn.Parameter(
            torch.zeros(num_tilts, dtype=real_dtype), requires_grad=False
        )
        self._pose_z3 = nn.Parameter(torch.zeros(num_tilts, dtype=real_dtype), requires_grad=False)
        self._pose_shifts = nn.Parameter(
            torch.zeros(num_tilts, 2, dtype=real_dtype), requires_grad=False
        )
        # Beam-frame coordinate origin in scan-grid pixels (set at preprocess): the center of
        # the scan grid, anchored to the specimen-box center. Coordinates are emitted in Å
        # relative to this point (see PtychoTomoPatchData).
        self.register_buffer("_scan_center_px", torch.full((2,), torch.nan, dtype=real_dtype))
        # Baseline pose the parameters reset to (default zeros). set_tilt_axis_pose() uses this
        # to fix a dataset-wide tilt-AXIS convention (e.g. z1=-90, z3=+90 turns the ZXZ x-tilt
        # into a tilt about y — the ASE-simulated AuNP series' convention, found 2026-07-13).
        self.register_buffer("_pose_z1_init", torch.zeros(num_tilts, dtype=real_dtype))
        self.register_buffer("_pose_z3_init", torch.zeros(num_tilts, dtype=real_dtype))

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
            ds._amplitudes = torch.empty(0)
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
        payload = PtychoTomoPatchData(
            coords_yx_A=coords_A, rotations=rotations, tilt_indices=tilt_idx
        )
        return payload, positions_px, torch.zeros_like(positions_px), None

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
        self._ensure_rot_offset_buffer()
        off = self._rot_axis_offset_A
        if bool((off != 0).any()):
            rows_A = rows_A - off[0]
            cols_A = cols_A - off[1]
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
        """TEMPORARY (2026-07-15) — REMOVE once proper pose optimization lands.

        Shift the beam-frame coordinate origin (== the tilt-axis position) by a known offset in
        Å from the scan-grid center. Needed because abTEM ``GridScan`` construction leaves the
        scan-pattern center short of the simulation cell center (the true rotation center) by up
        to half a scan step per axis — a rotation-center error that produces arc/"banana" atom
        artifacts growing with scan step (measured: 0.30/0.49/0.75/0.99 Å at 0.6/1.0/1.5/2.0 Å
        steps). Pass the (row, col) offset FROM the scan center TO the true rotation center.
        Survives ``reset()`` (dataset geometry, not a learned correction).
        """
        self._ensure_rot_offset_buffer()
        with torch.no_grad():
            self._rot_axis_offset_A[0] = float(drow_A)
            self._rot_axis_offset_A[1] = float(dcol_A)

    def _ensure_pose_init_buffers(self) -> None:
        """Create the pose-baseline buffers when absent (objects deserialized from saves/caches
        that predate them bypass ``__init__``)."""
        if "_pose_z1_init" not in self._buffers:
            self.register_buffer("_pose_z1_init", torch.zeros_like(self._pose_z1.data))
        if "_pose_z3_init" not in self._buffers:
            self.register_buffer("_pose_z3_init", torch.zeros_like(self._pose_z3.data))

    def set_tilt_axis_pose(self, z1_deg: float, z3_deg: float) -> None:
        """Fix the series-wide tilt-axis convention via constant z1/z3 Euler offsets.

        ``rot_beam_to_spec(z1, tilt, z3)`` with constant ``z1=-90, z3=+90`` rotates about the
        specimen y axis instead of x — matching tilt series simulated with ASE ``atoms.rotate``
        (the AuNP datasets). The values survive ``reset()`` (they define the dataset geometry,
        not a learned correction).
        """
        self._ensure_pose_init_buffers()
        with torch.no_grad():
            self._pose_z1_init.fill_(float(z1_deg))
            self._pose_z3_init.fill_(float(z3_deg))
            self._pose_z1.copy_(self._pose_z1_init)
            self._pose_z3.copy_(self._pose_z3_init)

    def reset(self) -> None:
        super().reset()
        self._ensure_pose_init_buffers()
        with torch.no_grad():
            self._pose_z1.copy_(self._pose_z1_init)
            self._pose_dtheta.zero_()
            self._pose_z3.copy_(self._pose_z3_init)
            self._pose_shifts.zero_()


PtychoTomoDatasetType = PtychoTomoDatasetRaster
