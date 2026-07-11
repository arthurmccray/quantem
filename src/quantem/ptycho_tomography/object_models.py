"""Rotated-coordinate 3D object models for joint ptycho-tomography.

The object is a single 3D field in the specimen frame, stored as **phase-density in rad/Å**
(``obj_type="potential"``) or rad/Å of signable phase (``"pure_phase"``), on a grid with **cubic
voxels by default** (z voxel = lateral sampling, matching the tomography module). Per tilt, the
forward model conceptually *rotates the volume and bins it into the multislice slabs*: each slab's
transmission is ``exp(1j * mean_k(density at rotated quadrature points) * slab_thickness)``, with
the quadrature points spaced ~one z-voxel apart through the slab (``samples_per_slab``). Because
the stored field is a density, the representation is independent of the slicing and invariant
under rotation (path lengths are preserved).

Backends share the implicit-model contract used across quantem: an ``nn.Module`` mapping ``(N, 3)``
normalized ``(z, y, x)`` coordinates in ``[-1, 1]^3`` to ``(N, 1)`` values. The **shared
rotated-query forward lives in the base** — ``VoxelGrid`` (dense trilinear-interpolated voxels,
the easy-to-train baseline) and the K-Planes core models are interchangeable backends behind it;
the subclasses differ only in their backend module, factories, optimizer wiring, and hard
projections.

Geometry: the object box is the padded 2D ptychography extent (set by ``_initialize_obj`` during
preprocessing) crossed with ``box_thickness_A = thickness_A + 2 * z_padding_A`` along z. The
z-padding is vacuum headroom along the beam: at high tilt, probes near the lateral edges rotate
into |z| > thickness/2, and for planar (non-vacuum-padded) samples real density lives there.
Coordinates are rotated in physical Å and then renormalized **per axis** to the box (the
normalized coordinates are anisotropic). The beam-frame multislice slab stack spans the same
padded box.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Sequence, cast
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.ml.constraints import BaseConstraints, Constraints
from quantem.core.ml.loss_functions import get_loss_module
from quantem.core.ml.models.kplanes import CPTilted, KPlanes, KPlanesTILTED, KPlanesType
from quantem.core.ml.optimizer_mixin import (
    OptimizerParams,
    OptimizerParamsType,
    SchedulerParamsType,
)
from quantem.core.utils.validators import validate_tensor
from quantem.core.visualization import show_2d
from quantem.core.visualization.custom_normalizations import CustomNormalization
from quantem.diffractive_imaging.object_models import ObjectBase, object_type
from quantem.ptycho_tomography.geometry import PtychoTomoPatchData, slab_z_centers


def _axis_coords(n: int, device, dtype) -> torch.Tensor:
    """Normalized voxel-center coordinates along one axis (matches ``idx/(N-1)*2-1``)."""
    if n == 1:
        return torch.zeros(1, device=device, dtype=dtype)
    return torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)


class PtychoTomoObjConstraintParams:
    """Namespace for ptycho-tomography object constraint sets (cf. PtychoObjConstraintParams)."""

    @dataclass
    class Volume(Constraints):
        """Constraints for the 3D ptycho-tomography object models.

        Soft penalties are evaluated at randomly sampled coordinates (differentiable through any
        backend, never materializing the volume); hard constraints are post-step projections that
        only the voxel backend can apply in place (K-Planes ignores them — use the soft
        ``positivity_weight`` there).

        Attributes
        ----------
        tv_weight : float, default ``0.0``
            Soft. **Isotropic L2** 3D total variation at sampled coordinates: the same weight is
            applied to all three axes (specimen-z and the two lateral axes), matching the
            tomography module's ``tv_vol``. Each axis penalizes the **squared** adjacent-voxel
            difference (L2, as in tomography; the diffractive_imaging ptychography TV uses L1). The
            per-axis sample step is one cubic voxel, so the scaling is identical across z and xy.
        positivity_weight : float, default ``0.0``
            Soft. ``weight * mean(relu(-value))`` at sampled coordinates (``potential`` only).
            The main positivity handle for the K-Planes backend.
        sparsity_weight : float, default ``0.0``
            Soft L1 sparsity: ``weight * mean(|value|)`` at sampled coordinates. An L1 prior on
            the density that drives the under-determined vacuum regions (including the interior
            voids) toward zero (cf. the tomography ``sparsity`` term). An alternative to
            ``fix_potential_baseline`` for cleaning the nonzero vacuum baseline of the implicit
            backends, and unlike that display gauge it shapes the reconstruction itself.
        tv_plane_weight : float, default ``0.0``
            Soft (**K-Planes backend only**; the voxel backend has no factor planes and ignores
            it). Total variation applied **directly on the K-Planes feature planes** rather than
            on the sampled output density: squared adjacent-element differences along both plane
            axes of every feature plane, averaged over channels and summed across multiscale
            levels. Mirrors the tomography module's ``tv_plane``. Because it acts on the O(0.1-1)
            factor planes at *every* element (not a sparse coordinate sample of the sub-1e-3
            density), a given weight bites far harder than ``tv_weight`` and is cheap (no model
            forward). Smooths the tensor-decomposition representation itself — the natural place
            to suppress rotation-induced high-frequency aliasing in the tilted K-Planes object.
        positivity : bool, default ``True``
            Hard (voxel backend only). Clamp the density >= 0 after every optimizer step
            (``potential`` only) — suppresses the phase-winding instability; mirrors the
            validated ``ObjectPixelated`` behavior.
        shrink_quantile : float | None, default ``None``
            Hard (voxel backend only). Per-z-plane background shrinkage before the clamp:
            subtract each plane's ``shrink_quantile`` level every step. The L2-amplitude data
            term is nearly blind to a smooth positive background, so without this haze
            accumulates. Valid while every z-plane is mostly vacuum laterally (isolated
            samples; quantile below the vacuum area fraction). ~0.2 works well.
        fix_potential_baseline : bool, default ``False``
            Display gauge on the materialized volume (``potential`` only): subtract a
            background offset (mask-background mean, else a robust low quantile of the volume)
            and clamp >= 0. Drives a nonzero vacuum baseline to ~0 without touching the
            reconstruction (a constant offset is a diffraction-invariant global phase).
        fix_potential_baseline_factor : float, default ``1.0``
            Scales the subtracted baseline offset.
        """

        tv_weight: float = 0.0
        positivity_weight: float = 0.0
        sparsity_weight: float = 0.0
        tv_plane_weight: float = 0.0
        positivity: bool = True
        shrink_quantile: float | None = None
        fix_potential_baseline: bool = False
        fix_potential_baseline_factor: float = 1.0
        _name: str = "volume"

        soft_constraint_keys = [
            "tv_weight",
            "positivity_weight",
            "sparsity_weight",
            "tv_plane_weight",
        ]
        hard_constraint_keys = [
            "positivity",
            "shrink_quantile",
            "fix_potential_baseline",
            "fix_potential_baseline_factor",
        ]


PtychoTomoObjConstraintsType = PtychoTomoObjConstraintParams.Volume


class VoxelGrid(nn.Module):
    """Dense voxel grid queried by trilinear interpolation at continuous 3D coordinates.

    Satisfies the implicit-model contract: ``(N, 3)`` normalized ``(z, y, x)`` in ``[-1, 1]^3`` ->
    ``(N, 1)``. ``align_corners=True`` matches the ``idx/(N-1)*2-1`` normalization used by the
    dataset coordinates, so querying at exact voxel centers reproduces direct indexing. Samples
    outside ``[-1, 1]^3`` return 0 (vacuum), consistent with the object models' box masking.
    """

    def __init__(self, volume: torch.Tensor):
        super().__init__()
        if volume.ndim != 3:
            raise ValueError(f"volume must be 3D (D, H, W), got shape {tuple(volume.shape)}")
        self.volume = nn.Parameter(volume)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[-1] != 3:
            raise ValueError(f"coords must have shape (N, 3), got {tuple(coords.shape)}")
        # grid_sample's grid is ordered (x, y, z) over volume dims (W, H, D); ours is (z, y, x)
        grid = coords.flip(-1).to(self.volume.dtype).view(1, 1, 1, -1, 3)
        vals = F.grid_sample(
            self.volume[None, None],
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        )
        return vals.view(-1, 1)


class ObjectPtychoTomoBase(BaseConstraints[PtychoTomoObjConstraintParams.Volume], ObjectBase):
    """Base for joint ptycho-tomography object models: the shared rotated-query engine.

    Wraps a coordinate-queried ``nn.Module`` backend (``(N, 3)`` normalized ``(z, y, x)`` ->
    ``(N, 1)`` density) representing one 3D specimen volume. The **forward is defined here, not
    in the subclasses, because it is identical for every backend**: build beam-frame coordinates
    for each multislice slab's quadrature points, rotate them into the specimen frame
    (:class:`PtychoTomoPatchData` payload), query the backend, mask outside the object box to
    vacuum, and integrate across the slab ("rotate the volume, then bin it to slices").
    Subclasses provide the backend module, factories, optimizer wiring, and hard projections.

    Note on ``is_implicit``: this returns True for ALL ptycho-tomo objects — including the voxel
    grid — because it is the existing ptychography wiring flag meaning "the paired dataset must
    emit continuous coordinates" (the rotated query needs them), not "the representation is an
    INR". The voxel grid is a raster queried at arbitrary rotated coordinates.
    TODO: rename the flag (e.g. ``continuous_coords``) when next touching diffractive_imaging.
    """

    DEFAULT_LRS = {"object": 3e-4}  # calibrated for density (rad/Å) units; see nb05
    DEFAULT_CONSTRAINTS: PtychoTomoObjConstraintParams.Volume = (
        PtychoTomoObjConstraintParams.Volume()
    )

    # Quadrature points per multislice slab for the slab-binning integral. None -> automatic:
    # ~one sample per z-voxel (round(slab_thickness / z_voxel)), the "rotate then bin" limit.
    # Compute scales linearly with this; set to 1 for fast midpoint-rule prototyping.
    samples_per_slab: int | None = None

    # Soft cap on rotated-coordinate points materialized per ``forward`` chunk. The S*K
    # slab-quadrature sample sets are processed in chunks of ``budget // (B*Hroi*Wroi)`` sample-sets
    # (then trimmed to the in-box subset before the backend query). Sized for a few million points
    # per chunk -- the same budget used by ``_query_volume_grid`` -- so peak activation memory and
    # the per-chunk dense coordinate tensors stay bounded regardless of batch size.
    # Subclasses override per backend: VoxelGrid gains ~20-25% end-to-end from 64M (fewer, larger
    # grid_sample calls; backward -45%/call) while K-Planes backends should stay at 4M (16M gave
    # ~1% speed for 3.5x memory under checkpointing).
    _forward_point_budget: int = 4_000_000

    # Opt-in gradient checkpointing of the per-chunk gather compute in ``forward``. When True,
    # each chunk's (rotated coords -> in-box mask -> backend query) runs under
    # ``torch.utils.checkpoint.checkpoint(use_reentrant=False)``: only the chunk *outputs*
    # (in-box densities + scatter indices) are retained for backward and the activations are
    # recomputed. Cuts gather-loop retained memory from O(all chunk activations) to O(in-box
    # outputs) -- e.g. K-Planes sps=7/num_slices=16 batch 128 drops from OOM (>40 GB) to
    # ~6.8 GiB -- at ~10-20% extra time per iteration. Required for K-Planes backends at
    # production sampling on 40 GB GPUs; unnecessary for configs that already fit.
    forward_checkpoint: bool = False

    def __init__(
        self,
        model: nn.Module,
        thickness_A: float,
        num_slices: int = 8,
        volume_shape: tuple[int, int, int] | None = None,
        num_z_voxels: int | None = None,
        obj_type: object_type = "potential",
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        super().__init__(device=device, obj_type=obj_type, rng=rng, _token=_token)
        if self.obj_type == "complex":
            raise NotImplementedError(
                "ptycho-tomography objects are real-valued densities; use 'potential' or "
                "'pure_phase'"
            )
        if num_slices < 1:
            raise ValueError(f"num_slices must be greater than 0, got {num_slices}")
        thickness_A = float(thickness_A)
        if thickness_A <= 0:
            raise ValueError(f"thickness_A must be > 0, got {thickness_A}")
        self._num_slices = int(num_slices)
        self._model = model.to(self._device)
        self._thickness_A = thickness_A
        # Physical-box geometry (2026-07-10 refactor): the specimen box (crop target) is
        # (thickness_A, *lateral_box_A); box_margin_A adds representation-support margin OUTSIDE
        # the box on each side per axis (z, y, x). Margins default to zero — with implicit
        # (coordinate-queried) backends there is no wrap-around edge, and margin only matters
        # when true material extends beyond the box (plan-view / real data; under tilt the beam
        # samples up to ±(box_z/2)·sin(θ_max) laterally beyond the 0° footprint).
        self._lateral_box_A: tuple[float, float] | None = None
        self._box_margin_A: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._geometry_set: bool = False
        self.slice_thicknesses = self.box_thickness_A / num_slices if num_slices > 1 else None
        self._set_pretrained_weights(self._model)
        self._num_z_voxels = int(num_z_voxels) if num_z_voxels is not None else None

        # Padded object extent [num_slices, H, W]; set in _initialize_obj. Defines the lateral
        # [-1, 1] coordinate domain.
        self._obj_shape: tuple[int, int, int] | None = None
        # Lazily materialized specimen volume (detached); invalidated each forward().
        self._obj_cache: torch.Tensor | None = None

        # Explicit volume_shape (e.g. wrapping an existing array) is kept as-is; otherwise the
        # lateral grid is matched to the padded 2D object at preprocess (_initialize_obj) and the
        # z count defaults to CUBIC voxels (z voxel = mean lateral sampling) unless num_z_voxels.
        self._volume_shape_explicit = volume_shape is not None
        self._volume_shape: tuple[int, int, int] | None = None
        if volume_shape is not None:
            vs = tuple(int(s) for s in volume_shape)
            if len(vs) != 3 or any(s < 1 for s in vs):
                raise ValueError(f"volume_shape must be 3 positive ints, got {volume_shape}")
            self._volume_shape = (vs[0], vs[1], vs[2])

    # region --- properties ---
    @property
    def name(self) -> str:
        return "ObjectPtychoTomoBase"

    @property
    def is_implicit(self) -> bool:
        """True for all ptycho-tomo objects: the dataset must emit continuous coordinates for
        the rotated query (see class docstring — this does NOT mean the backend is an INR)."""
        return True

    @property
    def num_slices(self) -> int:
        return self._num_slices

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def params(self) -> list[nn.Parameter]:
        """optimization parameters"""
        return list(self._model.parameters())

    @property
    def pretrained_weights(self) -> dict[str, torch.Tensor]:
        return self._pretrained_weights

    def _set_pretrained_weights(self, model: nn.Module) -> None:
        self._pretrained_weights = deepcopy(model.state_dict())

    @property
    def thickness_A(self) -> float:
        """Specimen thickness (Å) — the un-padded object z extent (cropping target)."""
        return self._thickness_A

    @property
    def z_padding_A(self) -> float:
        """DEPRECATED alias of ``box_margin_A[0]`` (z support margin per side, Å)."""
        return self._box_margin_A[0]

    @property
    def box_thickness_A(self) -> float:
        """Total support-box / multislice-slab z extent: ``thickness_A + 2 * box_margin_A[0]``."""
        return self._thickness_A + 2.0 * self._box_margin_A[0]

    @property
    def box_margin_A(self) -> tuple[float, float, float]:
        """Support margin OUTSIDE the specimen box, per side, ``(mz, my, mx)`` in Å.

        Zero by default: implicit backends have no wrap-around edge, so margin is only useful
        when true material extends beyond the specimen box (plan-view / real data — it pushes
        the box-edge vacuum clip away from the measured region, at the cost of diluting feature
        resolution over a larger box). ``obj_cropped`` removes it on all three axes.
        """
        return self._box_margin_A

    @property
    def lateral_box_A(self) -> tuple[float, float] | None:
        """Specimen-box lateral extents ``(by, bx)`` in Å (crop target; snapped to the point
        grid ``n * sampling``). None until :meth:`set_geometry` (or the legacy pixel handshake,
        which leaves it None and derives crops at the ptychography level)."""
        return self._lateral_box_A

    @property
    def specimen_box_A(self) -> tuple[float, float, float]:
        """The crop-target specimen box ``(bz, by, bx)`` in Å: ``(thickness_A, *lateral_box_A)``.

        Axis-aligned with the object grid in the specimen frame; its center is the coordinate
        origin (the tilt axis passes through it). With a rotated scan (common CoM rotation) the
        rotated footprint sits inside this axis-aligned box — corner regions weakly constrained
        at 0° are still constrained by tilted views and are NOT cropped away.
        """
        if self._lateral_box_A is None:
            raise ValueError("lateral_box_A not set; call set_geometry() (new preprocess path).")
        return (self._thickness_A, self._lateral_box_A[0], self._lateral_box_A[1])

    def set_geometry(
        self,
        *,
        lateral_box_A: "tuple[float, float] | np.ndarray",
        sampling: "tuple[float, float] | np.ndarray",
        box_margin_A: "tuple[float, float, float] | np.ndarray" = (0.0, 0.0, 0.0),
    ) -> None:
        """Physical-units geometry handshake (replaces the pixel-padding ``_initialize_obj``).

        Snaps the lateral specimen box and lateral margins to whole voxels (point-grid
        convention: an extent ``n * s`` spans ``n + 1`` grid points), allocates the volume grid
        over box + 2*margin per axis, and updates the slab geometry. Sampling must be equal in
        y and x (raises otherwise); the z voxel is ``box_thickness_A / D`` with
        ``D = round(box_thickness_A / s)`` — equal to the lateral pixel to <0.1% (exact when the
        box thickness is a whole number of pixels).
        """
        samp = np.asarray(sampling, dtype=float).ravel()
        if samp.size == 3:  # tolerate a (z, y, x) triple; lateral components are authoritative
            samp = samp[1:]
        if samp.size != 2:
            raise ValueError(f"sampling must have 2 (y, x) components, got {sampling}")
        if not np.isclose(samp[0], samp[1], rtol=1e-6, atol=0.0):
            raise ValueError(
                f"ptycho-tomography requires equal y/x sampling (cubic voxels), got "
                f"({samp[0]:.8g}, {samp[1]:.8g}) Å"
            )
        s = float(samp[0])
        mz, my, mx = (float(m) for m in np.asarray(box_margin_A, dtype=float).ravel())
        if min(mz, my, mx) < 0:
            raise ValueError(f"box_margin_A components must be >= 0, got {box_margin_A}")
        by, bx = (float(v) for v in np.asarray(lateral_box_A, dtype=float).ravel())
        if min(by, bx) <= 0:
            raise ValueError(f"lateral_box_A extents must be > 0, got {lateral_box_A}")
        n_by, n_bx = max(1, round(by / s)), max(1, round(bx / s))
        n_my, n_mx = round(my / s), round(mx / s)
        self._lateral_box_A = (n_by * s, n_bx * s)
        self._box_margin_A = (mz, n_my * s, n_mx * s)
        self.sampling = (s, s)
        if self.num_slices > 1:
            self.slice_thicknesses = self.box_thickness_A / self.num_slices
        h_pts, w_pts = n_by + 2 * n_my + 1, n_bx + 2 * n_mx + 1
        self._obj_shape = (self.num_slices, h_pts, w_pts)
        if self._volume_shape_explicit:
            if self._volume_shape is not None and self._volume_shape[1:] != (h_pts, w_pts):
                warn(
                    f"explicit volume_shape lateral dims {self._volume_shape[1:]} differ from "
                    f"the support grid {(h_pts, w_pts)}; cropping will be misaligned.",
                    stacklevel=2,
                )
        else:
            # point convention on z too: round(box/s) + 1 points span the box thickness, so a
            # cubic specimen box crops to a cubic array (z spacing == lateral pixel to <0.1%)
            d = self._num_z_voxels or (max(1, round(self.box_thickness_A / s)) + 1)
            new_shape = (int(d), h_pts, w_pts)
            if self._volume_shape != new_shape:
                self._volume_shape = new_shape
                self._allocate_backend()
        self._geometry_set = True
        self._invalidate_obj_cache()

    @property
    def crop_slices(self) -> tuple[slice, slice, slice]:
        """Slices that crop the support volume down to the specimen box on ALL three axes
        (removing ``box_margin_A``); for a cubic specimen box the crop is cubic."""
        d, h, w = self.volume_shape
        mz, my, mx = self._box_margin_A
        s = float(self.sampling[0]) if self.sampling is not None else self.z_voxel_A
        kz = int(round(mz / self.z_voxel_A))
        ky, kx = int(round(my / s)), int(round(mx / s))
        return (slice(kz, d - kz), slice(ky, h - ky), slice(kx, w - kx))

    def set_z_padding_A(self, pad_A: float) -> None:
        """DEPRECATED: use ``set_geometry(box_margin_A=(pad, my, mx))``. Sets the z support
        margin (each side, Å) and updates the slab geometry."""
        warn(
            "set_z_padding_A is deprecated - pass box_margin_A to set_geometry()/preprocess()",
            DeprecationWarning,
            stacklevel=2,
        )
        print("[DEPRECATED] set_z_padding_A -> set_geometry(box_margin_A=...); will be removed")
        pad_A = float(pad_A)
        if pad_A < 0:
            raise ValueError(f"z padding must be >= 0, got {pad_A}")
        self._box_margin_A = (pad_A, self._box_margin_A[1], self._box_margin_A[2])
        if self.num_slices > 1:
            self.slice_thicknesses = self.box_thickness_A / self.num_slices
        self._invalidate_obj_cache()

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        """Storage/materialization grid ``(D, H, W)`` of the specimen-frame volume (padded box)."""
        if self._volume_shape is None:
            raise ValueError(
                "volume_shape not set; pass it explicitly or run preprocess() "
                "(which matches the lateral grid to the padded object)."
            )
        return self._volume_shape

    @property
    def z_voxel_A(self) -> float:
        """Specimen-frame z grid spacing of the volume grid (Å), point convention.

        ``D`` grid points span the box thickness, so spacing is ``box / (D - 1)`` — matching
        the actual ``linspace(-1, 1, D)`` materialization grid (the old ``box / D`` under-read
        the real spacing by one part in D) and the lateral point-grid convention.
        """
        d = int(self.volume_shape[0])
        return self.box_thickness_A / max(d - 1, 1)

    @property
    def slab_thickness_A(self) -> float:
        """Thickness of each multislice slab (uniform), Å. Slabs span the padded box."""
        return self.box_thickness_A / self.num_slices

    @property
    def _slab_z_centers_t(self) -> torch.Tensor:
        real_dtype = getattr(torch, config.get("dtype_real"))
        return slab_z_centers(
            self.num_slices, self.box_thickness_A, device=self.device, dtype=real_dtype
        )

    @property
    def _box_half_extents(self) -> tuple[float, float, float]:
        """Physical half-extents ``(h_z, h_y, h_x)`` of the (padded) object box in Å.

        Lateral extents come from the padded 2D object grid set during preprocessing
        (``_initialize_obj``); z from the padded box thickness.
        """
        if self._obj_shape is None:
            raise ValueError("Object shape not set; call preprocess() (or _initialize_obj) first.")
        samp = self.sampling
        _, h_full, w_full = self._obj_shape
        return (
            self.box_thickness_A / 2.0,
            (int(h_full) - 1) / 2.0 * float(samp[0]),
            (int(w_full) - 1) / 2.0 * float(samp[1]),
        )

    @property
    def obj(self) -> torch.Tensor:
        """Materialized specimen-frame volume ``(D, H, W)`` over the padded box (display gauge
        applied).

        Cold-path only (display / logging / serialization); the training loop queries the model
        directly via ``forward``. Cached and invalidated on each ``forward`` call.
        """
        if self._obj_cache is None:
            raw = self._materialize_obj()
            self._obj_cache = self.apply_hard_constraints(raw, mask=self.mask)
        return self._obj_cache

    @property
    def volume(self) -> torch.Tensor:
        """Alias of ``obj``: the materialized specimen-frame volume (padded box)."""
        return self.obj

    # endregion --- properties ---

    def _invalidate_obj_cache(self) -> None:
        self._obj_cache = None

    def _initialize_obj(
        self,
        shape: "tuple[int, int, int] | np.ndarray",
        sampling: "tuple[float, float] | np.ndarray | None" = None,
    ) -> None:
        """Geometry handshake from preprocessing (padded shape + sampling).

        When ``volume_shape`` was not given explicitly, the volume's lateral grid is matched to
        the padded 2D object (so the materialized volume lives in the same pixel space as the
        scan positions and the inherited cropping works) and the z count defaults to **cubic
        voxels** over the padded box (``box_thickness_A / mean(sampling)``) unless
        ``num_z_voxels`` was set.
        """
        if self._geometry_set:
            # Physical-units geometry (set_geometry) is authoritative; ignore the legacy
            # pixel-padding handshake that the base obj_padding_px setter still triggers.
            return
        if sampling is not None:
            samp = np.asarray(sampling, dtype=float).ravel()
            self.sampling = tuple(samp[-2:]) if samp.size >= 2 else sampling
        shape_t = tuple(int(x) for x in shape)
        if shape_t[0] != self.num_slices:
            raise ValueError(
                f"shape[0] ({shape_t[0]}) does not match num_slices ({self.num_slices})"
            )
        self._obj_shape = shape_t  # type: ignore[assignment]
        self._invalidate_obj_cache()
        lat = (int(shape[1]), int(shape[2]))
        if self._volume_shape_explicit:
            if self._volume_shape is not None and self._volume_shape[1:] != lat:
                warn(
                    f"explicit volume_shape lateral dims {self._volume_shape[1:]} differ from "
                    f"the padded object grid {lat}; obj_cropped/snapshot cropping will be "
                    "misaligned (fine for standalone/validation use).",
                    stacklevel=2,
                )
            return
        if self._num_z_voxels is not None:
            d = self._num_z_voxels
        else:
            samp = self._sampling
            if samp is None:
                raise ValueError(
                    "Cannot infer num_z_voxels without sampling; pass num_z_voxels or sampling."
                )
            d = max(1, round(self.box_thickness_A / float(np.mean(np.asarray(samp)))))
        new_shape = (int(d), *lat)
        if self._volume_shape != new_shape:
            self._volume_shape = new_shape
            self._allocate_backend()

    def _allocate_backend(self) -> None:
        """Hook called when ``volume_shape`` is (re)set at preprocess time.

        No-op for purely implicit backends (volume_shape only sets the materialization grid and
        TV step sizes); the voxel backend (re)allocates its parameter grid here.
        """
        return

    def reset(self) -> None:
        """Reset the model weights to their initial (or pretrained) state."""
        self._model.load_state_dict(deepcopy(self._pretrained_weights))
        self._invalidate_obj_cache()

    def to(self, *args, **kwargs):
        """Move all relevant tensors to a different device."""
        super().to(*args, **kwargs)
        self._model = self._model.to(*args, **kwargs)
        device = kwargs.get("device", args[0] if args else None)
        if device is not None:
            self.device = device
            self._rng_to_device(device)
            self.reconnect_optimizer_to_parameters()
        self._invalidate_obj_cache()
        return self

    def backward(self, *args, **kwargs):
        raise NotImplementedError(
            f"Analytical gradients are not implemented for {self.name}, use autograd=True"
        )

    # region --- forward ---
    def _slab_sample_offsets(self) -> torch.Tensor:
        """Physical z offsets (Å) of the quadrature points within one slab, shape ``(K,)``.

        ``K = samples_per_slab`` (or automatically ~one sample per z-voxel); points are placed at
        sub-slab midpoints so ``K=1`` reduces to the midpoint rule at the slab center.
        """
        real_dtype = getattr(torch, config.get("dtype_real"))
        k = self.samples_per_slab
        if k is None:
            k = max(1, round(self.slab_thickness_A / self.z_voxel_A))
        k = max(1, int(k))
        t = self.slab_thickness_A
        return ((torch.arange(k, device=self.device, dtype=real_dtype) + 0.5) / k) * t - t / 2.0

    def forward(self, patch_data: PtychoTomoPatchData, /) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride] -- payload seam
        """Rotated-coordinate object query with slab binning (shared by all backends).

        For every multislice slab, builds beam-frame physical coordinates at the slab's
        quadrature z-points over each patch, rotates them into the specimen frame, queries the
        backend, masks outside the (padded) object box to vacuum, and integrates across the
        slab. Returns complex transmission patches
        ``exp(1j * slab_mean_density * slab_thickness)`` of shape
        ``(num_slices, batch, Hroi, Wroi)``.
        """
        self._invalidate_obj_cache()
        coords = patch_data.coords_yx_A  # (B, Hroi, Wroi, 2) beam-frame (row, col), physical Å
        rot = patch_data.rotations.to(device=coords.device, dtype=coords.dtype)  # (B, 3, 3)
        h_z, h_y, h_x = self._box_half_extents
        y_b = coords[..., 0]  # (B, Hroi, Wroi), physical Å (origin = specimen-box center)
        x_b = coords[..., 1]
        if (
            patch_data.shifts_A is not None
        ):  # per-tilt alignment shifts (beam frame, pre-rotation), Å
            y_b = y_b - patch_data.shifts_A[:, 0].view(-1, 1, 1)
            x_b = x_b - patch_data.shifts_A[:, 1].view(-1, 1, 1)
        z_centers = self._slab_z_centers_t.to(coords.dtype)  # (S,)
        z_offsets = self._slab_sample_offsets().to(coords.dtype)  # (K,)
        t_slab = self.slab_thickness_A

        r = rot.view(-1, 3, 3)
        rzz, rzy, rzx = (r[:, 0, i].view(-1, 1, 1) for i in range(3))
        ryz, ryy, ryx = (r[:, 1, i].view(-1, 1, 1) for i in range(3))
        rxz, rxy, rxx = (r[:, 2, i].view(-1, 1, 1) for i in range(3))
        # lateral contributions are z-independent: compute once per batch
        lat_z = rzy * y_b + rzx * x_b  # (B, H, W)
        lat_y = ryy * y_b + ryx * x_b
        lat_x = rxy * y_b + rxx * x_b

        # Batch the (slice, z_off) backend queries and query only the in-box samples. The S*K
        # slab-quadrature sample sets are flattened into one (S*K,) z-axis and processed in chunks
        # of ``group`` sample-sets (sized for a fixed points-per-call budget). Within each chunk we
        # compute the rotated coordinates, drop the out-of-box points (they contribute exactly zero
        # density -- the ``inside`` mask), query the backend on ONLY the kept points, and scatter
        # the result back into the per-slice accumulator. This keeps the slab-binning math
        # identical -- per slab the phase is still ``t_slab * mean_k(masked val_k)`` -- while
        # shrinking the dominant ``grid_sampler_2d_backward`` work to the in-box fraction (and
        # collapsing S*K small backend calls into a few large ones). The backward over the backend
        # query dominates wall time, so trimming its point count is the lever that pays off.
        s_idx = torch.arange(self.num_slices, device=coords.device)
        z_grid = z_centers.view(-1, 1) + z_offsets.view(1, -1)  # (S, K) physical z per sample
        z_flat = z_grid.reshape(-1)  # (S*K,)
        slice_of = s_idx.view(-1, 1).expand(self.num_slices, z_offsets.shape[0]).reshape(-1)
        n_samples = z_flat.shape[0]

        b, h, w = coords.shape[:3]
        bhw = b * h * w
        group = max(1, min(n_samples, self._forward_point_budget // max(bhw, 1)))

        acc = torch.zeros((self.num_slices * bhw,), dtype=coords.dtype, device=coords.device)
        # flat per-(slice, sample, b, h, w) position in ``acc`` for scatter; slice offset added below
        base = torch.arange(bhw, device=coords.device)

        def _chunk(zk, sl, rzz, ryz, rxz, lat_z, lat_y, lat_x, base):
            """(g,) z-samples -> (in-box densities, flat scatter indices). Pure given inputs;
            closes over self._model / h_z / h_y / h_x / bhw. All differentiable tensor inputs
            are explicit args so checkpointing keeps grads flowing upstream (probe positions,
            alignment shifts, tilt rotations)."""
            zk = zk.view(-1, 1, 1, 1)  # (g, 1, 1, 1)
            n_z = (rzz * zk + lat_z) / h_z  # (g, B, H, W)
            n_y = (ryz * zk + lat_y) / h_y
            n_x = (rxz * zk + lat_x) / h_x
            inside = (n_z.abs() <= 1.0) & (n_y.abs() <= 1.0) & (n_x.abs() <= 1.0)
            sel = inside.reshape(-1)  # (g*B*H*W,)
            pts = torch.stack([n_z, n_y, n_x], dim=-1).reshape(-1, 3)[sel]  # only in-box points
            # destination index in ``acc``: slice_of[sample] * bhw + (b,h,w position)
            dest = (sl.view(-1, 1) * bhw + base.view(1, -1)).reshape(-1)[sel]
            if pts.shape[0] == 0:
                return pts.new_zeros((0,)), dest
            return self._model(pts).reshape(-1), dest  # (n_inside,)

        for start in range(0, n_samples, group):
            # tensor slicing self-clamps at the end of the range, so no explicit ``g`` needed
            args = (
                z_flat[start : start + group],
                slice_of[start : start + group],
                rzz,
                ryz,
                rxz,
                lat_z,
                lat_y,
                lat_x,
                base,
            )
            if self.forward_checkpoint and torch.is_grad_enabled():
                # Recompute-in-backward. use_reentrant=False is required (data-dependent
                # ``sel`` output shapes; grads to closure-captured module params);
                # preserve_rng_state=False is safe (no RNG ops in the chunk) and avoids a
                # per-chunk CUDA RNG sync. Scatter stays OUTSIDE so retained per-chunk state
                # is O(n_inside), not O(num_slices * B*H*W).
                vals, dest = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    checkpoint(_chunk, *args, use_reentrant=False, preserve_rng_state=False),
                )
            else:
                vals, dest = _chunk(*args)
            # in-place: ``acc`` starts as fresh zeros and no graph node reads its pre-update
            # value, so autograd version counting is satisfied; out-of-place index_add would
            # reallocate the (num_slices*B*H*W) accumulator every chunk.
            acc.index_add_(0, dest, vals)
        phase = acc.view(self.num_slices, b, h, w) * (t_slab / z_offsets.shape[0])
        return torch.exp(1.0j * phase)

    # endregion --- forward ---

    # region --- materialization ---
    def _query_volume_grid(self, max_chunk_points: int = 4_000_000) -> torch.Tensor:
        """Query the model on the full ``volume_shape`` grid (differentiable), ``(D, H, W)``."""
        d, hh, ww = self.volume_shape
        real_dtype = getattr(torch, config.get("dtype_real"))
        zs = _axis_coords(d, self.device, real_dtype)
        ys = _axis_coords(hh, self.device, real_dtype)
        xs = _axis_coords(ww, self.device, real_dtype)
        chunk = max(1, max_chunk_points // max(hh * ww, 1))
        out = []
        for i in range(0, d, chunk):
            zz, yy, xx = torch.meshgrid(zs[i : i + chunk], ys, xs, indexing="ij")
            pts = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)
            out.append(self._model(pts).reshape(-1, hh, ww))
        return torch.cat(out, dim=0)

    def _materialize_obj(self) -> torch.Tensor:
        with torch.no_grad():
            return self._query_volume_grid()

    # endregion --- materialization ---

    # region --- constraints ---
    def apply_hard_constraints(
        self, raw: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project the materialized volume (display only).

        For ``pure_phase`` the density is recentered to zero mean (a global-phase gauge). For
        ``potential``, if ``fix_potential_baseline`` is set, subtract a background offset and clamp
        ``>= 0``. The offset is the mean over the background ``mask`` if one is provided, else a
        robust low quantile of the volume (the vacuum-dominated background level) scaled by
        ``fix_potential_baseline_factor``. A constant offset is a global phase
        (diffraction-invariant), so this does not affect the reconstruction. The global *minimum*
        is deliberately not used: a single noisy negative excursion (common in the implicit
        backends) would make the gauge add a baseline instead of removing it.
        """
        with torch.no_grad():
            if self.obj_type == "pure_phase":
                return raw - raw.mean()
            if self.constraints.fix_potential_baseline:
                if mask is not None and mask.numel() and (mask < 0.5 * mask.max()).any():
                    offset = raw[mask < 0.5 * mask.max()].mean()
                else:
                    flat = raw.reshape(-1).float()
                    if flat.numel() > 2_000_000:  # torch.quantile caps the input size
                        idx = torch.randperm(flat.numel(), device=flat.device)[:2_000_000]
                        flat = flat[idx]
                    offset = torch.quantile(flat, 0.25).to(raw.dtype)
                offset = offset * self.constraints.fix_potential_baseline_factor
                return torch.clamp(raw - offset, min=0.0)
            return raw

    def apply_soft_constraints(
        self, obj: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Coordinate-sampled **isotropic** 3D TV + positivity penalties over the specimen volume.

        ``tv_weight`` applies an isotropic total variation — the same weight on the specimen-z and
        the lateral axes, matching the tomography module's ``tv_vol``. Evaluated at randomly
        sampled coordinates so the penalty is differentiable without materializing the volume.
        ``tv_plane_weight`` additionally applies TV directly on the K-Planes feature planes (a
        no-op on the voxel backend); see :meth:`_plane_tv_loss`.
        """
        self.reset_soft_constraint_losses()
        loss = self._get_zero_loss_tensor()
        w_tv = self.constraints.tv_weight
        if w_tv > 0:
            tv_loss = self._sampled_tv3d_loss(w_tv)
            loss = loss + tv_loss
            self.add_soft_constraint_loss("tv_loss", tv_loss)
        w_plane = self.constraints.tv_plane_weight
        if w_plane > 0:
            plane_loss = self._plane_tv_loss(w_plane)
            loss = loss + plane_loss
            self.add_soft_constraint_loss("tv_plane_loss", plane_loss)
        w_pos = self.constraints.positivity_weight
        if w_pos > 0 and self.obj_type == "potential":
            pos_loss = self._sampled_positivity_loss(w_pos)
            loss = loss + pos_loss
            self.add_soft_constraint_loss("positivity_loss", pos_loss)
        w_l1 = self.constraints.sparsity_weight
        if w_l1 > 0:
            l1_loss = self._sampled_l1_loss(w_l1)
            loss = loss + l1_loss
            self.add_soft_constraint_loss("sparsity_loss", l1_loss)
        self.accumulate_constraint_losses()
        return loss

    def _sample_volume_coords(self, num_samples: int) -> torch.Tensor:
        real_dtype = getattr(torch, config.get("dtype_real"))
        return (
            torch.rand(
                num_samples, 3, device=self.device, dtype=real_dtype, generator=self._rng_torch
            )
            * 2.0
            - 1.0
        )

    def _sampled_positivity_loss(self, weight: float, num_samples: int = 4096) -> torch.Tensor:
        coords = self._sample_volume_coords(num_samples)
        value = self._model(coords).squeeze(-1)
        return weight * torch.relu(-value).mean()

    def _sampled_l1_loss(self, weight: float, num_samples: int = 4096) -> torch.Tensor:
        """Soft L1 sparsity: ``weight * mean(|density|)`` at randomly sampled coordinates."""
        coords = self._sample_volume_coords(num_samples)
        value = self._model(coords).squeeze(-1)
        return weight * value.abs().mean()

    def _sampled_tv3d_loss(self, weight: float, num_samples: int = 4096) -> torch.Tensor:
        """Isotropic **L2 (squared-difference)** TV at sampled coordinates (same weight on z, xy).

        Matches the tomography ``tv_vol`` (mean squared adjacent-voxel difference, equal weight on
        all three axes). NOTE: this L2 form intentionally DIFFERS from the diffractive_imaging
        ptychography TV (``ObjectBase._calc_tv_loss``), which uses L1 (mean ``|Δvalue|``) — chosen
        here to be consistent with the tomography module. An L1-vs-L2 comparison is left for later.
        The per-axis step ``h = 2 / volume_shape[axis]`` is one cubic voxel, so a unit step is the
        same physical distance on every axis and the scaling is isotropic.
        """
        real_dtype = getattr(torch, config.get("dtype_real"))
        coords = self._sample_volume_coords(num_samples)
        value = self._model(coords).squeeze(-1)
        loss = self._get_zero_loss_tensor()
        for axis in range(3):
            h = 2.0 / max(int(self.volume_shape[axis]), 2)  # one cubic-voxel step
            offset = torch.zeros(3, device=self.device, dtype=real_dtype)
            offset[axis] = h
            shifted = self._model(coords + offset).squeeze(-1)
            # L2 (squared) difference to match tomography; ptychography uses L1 (abs).
            loss = loss + weight * torch.mean((shifted - value) ** 2)
        return loss

    def _plane_tv_loss(self, weight: float) -> torch.Tensor:
        """Plane-TV hook. Only the K-Planes backend implements it (voxel returns zero).

        The dense voxel grid has no factor planes, so plane-TV is meaningless there and this base
        returns zero; :class:`ObjectKPlanesTomo` overrides it with the real penalty on the grids.
        """
        return self._get_zero_loss_tensor()

    # endregion --- constraints ---


class ObjectVoxelTomo(ObjectPtychoTomoBase):
    """Dense voxel-grid backend — the easy-to-train baseline ptycho-tomography object.

    The specimen volume is a single ``(D, H, W)`` ``nn.Parameter`` (phase-density, rad/Å) queried
    by trilinear interpolation at the rotated coordinates. Dense direct parameterization gives
    well-conditioned gradients; use it to validate geometry and training recipes before moving to
    the K-Planes backend. Warm starts go through :meth:`set_volume` (there is no pretraining —
    the parameter IS the volume).

    Hard constraints (``constraints.positivity`` / ``constraints.shrink_quantile``) are applied
    in place after every optimizer step via ``project_parameters`` — see
    :class:`PtychoTomoObjConstraintParams.Volume`.
    """

    # VoxelGrid's backend query is a single fused grid_sample, so large chunks are cheap:
    # 64M drops the gather backward ~45%/call for ~20-25% end-to-end (peak ~7.5 GiB at batch
    # 128, ~18.5 GiB at batch 512 on A100-40G). K-Planes backends keep the 4M base default.
    _forward_point_budget: int = 64_000_000

    @property
    def name(self) -> str:
        return "ObjectVoxelTomo"

    def project_parameters(self) -> None:
        """Post-step hard projection: per-plane background shrinkage + non-negativity clamp."""
        c = self.constraints
        if not c.positivity or self.obj_type != "potential":
            return
        model = self._model
        assert isinstance(model, VoxelGrid)
        with torch.no_grad():
            if c.shrink_quantile is not None and 0.0 < c.shrink_quantile < 1.0:
                planes = model.volume.view(model.volume.shape[0], -1)
                floor = torch.quantile(planes, c.shrink_quantile, dim=1, keepdim=True)
                planes.sub_(torch.clamp(floor, min=0.0))
            model.volume.clamp_(min=0.0)
        self._invalidate_obj_cache()

    @classmethod
    def from_uniform(
        cls,
        volume_shape: tuple[int, int, int] | None = None,
        thickness_A: float = 1.0,
        num_slices: int = 8,
        num_z_voxels: int | None = None,
        obj_type: object_type = "potential",
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ) -> "ObjectVoxelTomo":
        """Vacuum-initialized (all-zero) voxel volume.

        With ``volume_shape=None`` (the usual reconstruction path) the grid is allocated at
        preprocess time: lateral dims match the padded object and the z count is ``num_z_voxels``
        (default cubic voxels). Pass an explicit ``volume_shape`` for standalone use.
        """
        real_dtype = getattr(torch, config.get("dtype_real"))
        shape = tuple(int(s) for s in volume_shape) if volume_shape is not None else (1, 1, 1)
        model = VoxelGrid(torch.zeros(shape, dtype=real_dtype))
        return cls(
            model=model,
            thickness_A=thickness_A,
            num_slices=num_slices,
            volume_shape=volume_shape,
            num_z_voxels=num_z_voxels,
            obj_type=obj_type,
            device=device,
            rng=rng,
            _token=cls._token,
        )

    @classmethod
    def from_array(
        cls,
        volume: torch.Tensor | np.ndarray,
        thickness_A: float,
        num_slices: int = 8,
        obj_type: object_type = "potential",
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ) -> "ObjectVoxelTomo":
        """Wrap an existing specimen-frame volume (density in rad/Å), e.g. a ground-truth phantom
        for forward-model validation or a warm start."""
        real_dtype = getattr(torch, config.get("dtype_real"))
        vol = torch.as_tensor(np.asarray(volume), dtype=real_dtype).clone()
        if vol.ndim != 3:
            raise ValueError(f"volume must be 3D (D, H, W), got shape {tuple(vol.shape)}")
        model = VoxelGrid(vol)
        return cls(
            model=model,
            thickness_A=thickness_A,
            num_slices=num_slices,
            volume_shape=(int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])),
            obj_type=obj_type,
            device=device,
            rng=rng,
            _token=cls._token,
        )

    def _allocate_backend(self) -> None:
        """(Re)allocate the voxel parameter on the current ``volume_shape`` grid (vacuum)."""
        real_dtype = getattr(torch, config.get("dtype_real"))
        model = self._model
        assert isinstance(model, VoxelGrid)
        model.volume = nn.Parameter(
            torch.zeros(self.volume_shape, dtype=real_dtype, device=model.volume.device)
        )
        self._set_pretrained_weights(model)
        self._invalidate_obj_cache()
        self.reconnect_optimizer_to_parameters()

    def set_volume(self, volume: torch.Tensor | np.ndarray, set_as_initial: bool = True) -> None:
        """Load a specimen-frame volume (density, rad/Å) into the voxel grid.

        Shape must match ``volume_shape`` (e.g. a ground-truth phantom resampled onto the
        preprocess-time grid, for forward-model validation or warm starts). With
        ``set_as_initial`` the loaded volume becomes the ``reset()`` state — pass ``False`` for
        transient loads (e.g. evaluating the loss at the ground truth) so ``reset()`` still
        returns to the original (vacuum) state.
        """
        real_dtype = getattr(torch, config.get("dtype_real"))
        vol = torch.as_tensor(np.asarray(volume), dtype=real_dtype)
        if tuple(vol.shape) != self.volume_shape:
            raise ValueError(
                f"volume shape {tuple(vol.shape)} != volume_shape {self.volume_shape}"
            )
        model = self._model
        assert isinstance(model, VoxelGrid)
        with torch.no_grad():
            model.volume.copy_(vol.to(model.volume.device))
        if set_as_initial:
            self._set_pretrained_weights(model)
        self._invalidate_obj_cache()

    def _materialize_obj(self) -> torch.Tensor:
        # fast path: the parameter IS the specimen volume on the volume_shape grid
        model = self._model
        assert isinstance(model, VoxelGrid)
        return model.volume.detach().clone()


class ObjectKPlanesTomo(ObjectPtychoTomoBase):
    """K-Planes backend for joint ptycho-tomography.

    Swaps the dense voxel grid for a tensor-decomposition model from
    :mod:`quantem.core.ml.models.kplanes` (:class:`KPlanes`, the tilted :class:`KPlanesTILTED`
    with T learned SO(3) rotations — r9 ``SO3ParamR9SVD`` parameterization recommended — or
    :class:`CPTilted`). The models consume the same ``(N, 3)`` ``(z, y, x)`` coordinates, so the
    base's rotated-query forward / materialization / sampled-3D-TV are reused unchanged; this
    class adds the PPLR optimizer wiring and volume-regression pretraining (warm starts).

    Hard constraints (``positivity`` / ``shrink_quantile``) cannot be applied to feature planes —
    use the soft ``positivity_weight`` here.

    These models expose multiple parameter groups (``grids``/``sigma_net``, plus ``so3`` for the
    tilted variants), so ``optimizer_params`` must be a PPLR dict keyed by ``model.param_keys``,
    e.g. ``{"grids": OptimizerParams.Adam(lr=1e-2), "sigma_net": OptimizerParams.Adam(lr=1e-3)}``.
    ``resolution`` is the feature-plane resolution ``(z, y, x)``, independent of both the padded
    object grid and ``volume_shape`` (which only sets the materialization grid here).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pretraining state (volume-regression warm starts; K-Planes only).
        self.register_buffer("_pretrain_target", torch.tensor([]))
        self._pretrain_losses: list[float] = []
        self._pretrain_lrs: list[float] = []

    @property
    def name(self) -> str:
        return "ObjectKPlanesTomo"

    @property
    def model(self) -> KPlanesType:
        return cast(KPlanesType, self._model)

    def _plane_tv_loss(self, weight: float) -> torch.Tensor:
        """Total variation on the K-Planes **feature planes** themselves (dense, cheap).

        For every multiscale grid, penalize the squared adjacent-element difference along both
        plane axes (H and W), averaged over feature channels; sum the per-level penalties. For the
        tilted backend (:class:`KPlanesTILTED`) the ``3`` planes of each of the ``T`` learned
        rotations are summed and then averaged over rotations, matching the tomography module's
        ``_get_plane_tv_loss``. ``CPTilted`` line factors ``(3*T, C, L)`` are handled by
        differencing the single spatial axis. Backprops straight into the ``grids`` parameters, so
        it is a much stronger and cheaper smoothness prior than the coordinate-sampled 3D TV on the
        sub-1e-3 output density.
        """
        model = self.model
        grids = getattr(model, "grids", None)
        if grids is None or len(grids) == 0:
            return self._get_zero_loss_tensor()
        is_tilted = bool(getattr(model, "tilted", False))
        per_level = []
        for p in grids:
            if p.ndim == 4:  # (3*T, C, H, W) feature planes
                dh = (p[:, :, 1:, :] - p[:, :, :-1, :]).pow(2).mean(dim=(1, 2, 3))
                dw = (p[:, :, :, 1:] - p[:, :, :, :-1]).pow(2).mean(dim=(1, 2, 3))
                per_plane = dh + dw  # (3*T,)
            else:  # (3*T, C, L) CP line factors
                per_plane = (p[..., 1:] - p[..., :-1]).pow(2).mean(dim=tuple(range(1, p.ndim)))
            if is_tilted:
                # sum the 3 planes of each rotation, then average across the T rotations
                level_tv = per_plane.view(cast(int, model.T), 3).sum(dim=1).mean()
            else:
                level_tv = per_plane.sum()
            per_level.append(level_tv)
        return weight * torch.stack(per_level).sum()

    @classmethod
    def from_model(
        cls,
        model: KPlanesType,
        thickness_A: float,
        num_slices: int = 8,
        volume_shape: tuple[int, int, int] | None = None,
        num_z_voxels: int | None = None,
        obj_type: object_type = "potential",
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ) -> "ObjectKPlanesTomo":
        """Wrap a user-built K-Planes-family model as a ptycho-tomography object."""
        if not isinstance(model, (KPlanes, CPTilted)):  # KPlanesTILTED is a KPlanes subclass
            raise TypeError(
                f"model must be a KPlanes/KPlanesTILTED/CPTilted instance, got {type(model)}"
            )
        activation = getattr(model, "density_activation", None)
        if activation is not None and not isinstance(activation, nn.Module):
            warn(
                "KPlanes.density_activation is a plain callable (e.g. a lambda); saving this "
                "object will fail because AutoSerialize pickles the whole module. Use an "
                "nn.Module activation (e.g. nn.Identity), e.g. via ObjectKPlanesTomo.from_uniform.",
                stacklevel=2,
            )
        obj = cls(
            model=model,
            thickness_A=thickness_A,
            num_slices=num_slices,
            volume_shape=volume_shape,
            num_z_voxels=num_z_voxels,
            obj_type=obj_type,
            device=device,
            rng=rng,
            _token=cls._token,
        )
        obj.to(device)
        return obj

    @classmethod
    def from_uniform(
        cls,
        thickness_A: float,
        num_slices: int = 8,
        volume_shape: tuple[int, int, int] | None = None,
        num_z_voxels: int | None = None,
        M_features: int = 16,
        resolution: Sequence[int] = (64, 64, 64),
        multiscale_res_multipliers: Sequence[float] | None = (0.25, 0.5, 1.0),
        use_hybrid_mlp: bool = False,
        hybrid_hidden_dim: int = 64,
        hybrid_num_layers: int = 2,
        tilted: bool = False,
        T: int = 4,
        obj_type: object_type = "potential",
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ) -> "ObjectKPlanesTomo":
        """Build a default K-Planes-backed ptycho-tomo object, initialized to vacuum.

        Mirrors ``ObjectTensorDecomp.from_uniform``: identity decoder activation (positivity is
        the soft ``positivity_weight`` constraint) with the final decoder layer zeroed so the
        object starts at vacuum. ``tilted=True`` builds a :class:`KPlanesTILTED` (T learned SO(3)
        rotations, r9 parameterization by default in the core model).
        """
        density_activation: nn.Module = nn.Identity()
        ms = list(multiscale_res_multipliers) if multiscale_res_multipliers is not None else None
        model: KPlanesType
        if tilted:
            model = KPlanesTILTED(
                M_features=M_features,
                resolution=resolution,
                multiscale_res_multipliers=ms,
                density_activation=density_activation,
                T=T,
                use_hybrid_mlp=use_hybrid_mlp,
                hybrid_hidden_dim=hybrid_hidden_dim,
                hybrid_num_layers=hybrid_num_layers,
            )
        else:
            model = KPlanes(
                M_features=M_features,
                resolution=resolution,
                multiscale_res_multipliers=ms,
                density_activation=density_activation,
                use_hybrid_mlp=use_hybrid_mlp,
                hybrid_hidden_dim=hybrid_hidden_dim,
                hybrid_num_layers=hybrid_num_layers,
            )
        # Zero the final decoder layer so the object starts at vacuum.
        with torch.no_grad():
            final_linear = (
                model.sigma_net[-1]
                if isinstance(model.sigma_net, nn.Sequential)
                else model.sigma_net
            )
            final_linear = cast(nn.Linear, final_linear)
            final_linear.weight.zero_()
            if final_linear.bias is not None:
                final_linear.bias.zero_()
        return cls.from_model(
            model,
            thickness_A=thickness_A,
            num_slices=num_slices,
            volume_shape=volume_shape,
            num_z_voxels=num_z_voxels,
            obj_type=obj_type,
            device=device,
            rng=rng,
        )

    # region --- PPLR optimizer wiring ---
    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        """PPLR: one param group per ``model.param_keys``."""
        model = self.model
        groups = model.get_params()
        return {key: list(groups[key]) for key in model.param_keys}

    def _normalize_optimizer_params(self, params):
        """Require a dict keyed by ``model.param_keys`` (PPLR); reject single-optimizer specs.

        The framework's "disabled" sentinel — a bare ``NoneOptimizer`` or an all-``NoneOptimizer``
        dict (set at init / by ``remove_optimizer`` and replayed through ``reset_optimizer`` on
        ``reconstruct(reset=True)``) — passes straight to the base normalizer. Mirrors
        ``ObjectTensorDecomp._normalize_optimizer_params``.
        """
        if isinstance(params, OptimizerParams.NoneOptimizer) or (
            isinstance(params, dict)
            and len(params) > 0
            and all(isinstance(v, OptimizerParams.NoneOptimizer) for v in params.values())
        ):
            return super()._normalize_optimizer_params(params)
        if not isinstance(params, dict) or self._is_single_optimizer_dict(params):
            raise TypeError(
                f"{type(self).__name__} requires dict[str, OptimizerParamsType] keyed by "
                f"param_keys {self.model.param_keys}; got {type(params)}"
            )
        expected = set(self.model.param_keys)
        got = set(params.keys())
        if got != expected:
            raise ValueError(
                f"optimizer_params keys must match model.param_keys: got {got}, expected {expected}"
            )
        return super()._normalize_optimizer_params(params)

    # endregion --- PPLR optimizer wiring ---

    # region --- pretraining (volume-regression warm starts) ---
    @property
    def pretrain_target(self) -> torch.Tensor:
        """Target volume (density, ``volume_shape``) fitted by ``pretrain()``."""
        return self._pretrain_target

    @pretrain_target.setter
    def pretrain_target(self, target: torch.Tensor | np.ndarray | None) -> None:
        if target is None:
            self._pretrain_target = torch.tensor([], device=self.device)
            return
        t = validate_tensor(
            target,
            name="pretrain_target",
            ndim=3,
            dtype=config.get("dtype_real"),
            expand_dims=True,
        )
        self._pretrain_target = t.to(self.device)

    @property
    def pretrain_losses(self) -> np.ndarray:
        return np.array(self._pretrain_losses)

    @property
    def pretrain_lrs(self) -> np.ndarray:
        return np.array(self._pretrain_lrs)

    def pretrain(
        self,
        pretrain_target: torch.Tensor | np.ndarray | None = None,
        num_iters: int = 200,
        optimizer_params: "dict | OptimizerParamsType | None" = None,
        scheduler_params: "dict | SchedulerParamsType | None" = None,
        loss_fn: Callable | str = "l2",
        device: str | int | None = None,
        show: bool = True,
        normalize_object_plotting: bool = True,
    ) -> None:
        """Warm-start the K-Planes model by regressing it onto a specimen-frame volume.

        ``pretrain_target`` must have shape ``volume_shape`` (density in rad/Å) — e.g. a voxel
        reconstruction. The fitted weights become the ``reset()`` state.
        """
        if device is not None:
            dev, _ = config.validate_device(device)
            self.to(dev)
        if pretrain_target is not None:
            self.pretrain_target = pretrain_target
        if self._pretrain_target is None or self._pretrain_target.numel() == 0:
            raise ValueError("No pretrain target set; pass pretrain_target.")
        if tuple(self._pretrain_target.shape) != self.volume_shape:
            raise ValueError(
                f"pretrain_target shape {tuple(self._pretrain_target.shape)} != volume_shape "
                f"{self.volume_shape}"
            )
        if optimizer_params is not None:
            self.set_optimizer(optimizer_params)
        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_iters)
        loss_module = get_loss_module(loss_fn, getattr(torch, config.get("dtype_real")))
        self._pretrain(
            num_iters, loss_module, show=show, normalize_object_plotting=normalize_object_plotting
        )
        self._set_pretrained_weights(self._model)
        self._invalidate_obj_cache()

    def _pretrain(
        self,
        num_iters: int,
        loss_fn: Callable,
        show: bool = False,
        normalize_object_plotting: bool = True,
    ) -> None:
        optimizer = self.optimizer
        if optimizer is None:
            raise ValueError("Optimizer not set. Pass optimizer_params to pretrain().")
        scheduler = self.scheduler
        target = self._pretrain_target.to(self.device)
        self._model.train()
        pbar = tqdm(range(num_iters))
        output = self._query_volume_grid()
        for _ in pbar:
            optimizer.zero_grad()
            output = self._query_volume_grid()
            loss: torch.Tensor = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss.item())
                else:
                    scheduler.step()
            self._pretrain_losses.append(loss.item())
            self._pretrain_lrs.append(optimizer.param_groups[0]["lr"])
            pbar.set_description(
                f"Iter {len(self._pretrain_losses)}/{num_iters}, Loss: {loss.item():.3e}"
            )
        if show:
            self.visualize_pretrain(output.detach(), normalize_object_plotting)

    def visualize_pretrain(
        self, pred_obj: torch.Tensor, normalize_object_plotting: bool = True
    ) -> None:
        """Plot the pretraining loss / LR curves and the pred vs target volume projections."""
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
        ax = fig.add_subplot(gs[0])
        lines = []
        lines.extend(
            ax.semilogy(
                np.arange(len(self._pretrain_losses)), self._pretrain_losses, c="k", label="loss"
            )
        )
        ax.set_ylabel("Loss", color="k")
        ax.set_xlabel("Iterations")
        nx = ax.twinx()
        nx.spines["left"].set_visible(False)
        lines.extend(
            nx.semilogy(
                np.arange(len(self._pretrain_lrs)), self._pretrain_lrs, c="tab:orange", label="LR"
            )
        )
        labs = [lin.get_label() for lin in lines]
        nx.legend(lines, labs, loc="upper center")
        nx.set_ylabel("LRs")

        gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1])
        axs_bot = np.array([fig.add_subplot(gs_bot[0, i]) for i in range(2)])
        target = self._pretrain_target
        norm = None
        if normalize_object_plotting:
            target_mean = target.mean(0).cpu().detach().numpy()
            target_norm = CustomNormalization(interval_type="quantile", data=target_mean)
            norm = {
                "interval_type": "manual",
                "vmin": target_norm.vmin,
                "vmax": target_norm.vmax,
            }
        show_2d(
            [
                pred_obj.mean(0).cpu().detach().numpy(),
                target.mean(0).cpu().detach().numpy(),
            ],
            figax=(fig, axs_bot),
            title=[f"Pred volume ({self.obj_type})", f"Target volume ({self.obj_type})"],
            cmap="magma",
            cbar=True,
            norm=norm,
        )
        plt.suptitle(
            f"Final loss: {self._pretrain_losses[-1]:.3e} | Iters: {len(self._pretrain_losses)}",
            fontsize=14,
            y=0.94,
        )
        plt.show()

    # endregion --- pretraining ---


ObjectPtychoTomoType = ObjectVoxelTomo | ObjectKPlanesTomo
