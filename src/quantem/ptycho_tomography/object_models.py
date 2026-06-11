"""Rotated-coordinate 3D object models for joint ptycho-tomography.

The object is a single 3D field in the specimen frame, stored as **phase-density in rad/Å**
(``obj_type="potential"``) or rad/Å of signable phase (``"pure_phase"``). Per tilt, the multislice
slab centers are rotated from the beam frame into the specimen frame (``geometry.rot_beam_to_spec``)
and the field is queried at those coordinates; each slice's transmission is
``exp(1j * value * slab_thickness)`` (midpoint-rule slab line integral). Because the stored field is
a density, the representation is independent of the multislice slicing and invariant under
rotation (path lengths are preserved).

Backends share the implicit-model contract used across quantem: an ``nn.Module`` mapping ``(N, 3)``
normalized ``(z, y, x)`` coordinates in ``[-1, 1]^3`` to ``(N, 1)`` values. ``VoxelGrid`` (dense
trilinear-interpolated voxels, the easy-to-train baseline) is defined here; the K-Planes core model
already satisfies the contract.

Geometry: the object box is the padded 2D ptychography extent (set by ``_initialize_obj`` during
preprocessing) crossed with ``thickness_A`` along z. Coordinates are rotated in physical Å and then
renormalized **per axis** to the box (the normalized coordinates are anisotropic). The beam-frame
multislice slab stack spans the same ``thickness_A``, which covers the rotated object support as
long as the support fits in a ball of diameter <= thickness_A (true for the isolated round v1
samples; revisit for planar/extended samples).
"""

from typing import Callable, Sequence, cast
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.ml.loss_functions import get_loss_module
from quantem.core.ml.models.kplanes import CPTilted, KPlanes, KPlanesTILTED, KPlanesType
from quantem.core.ml.optimizer_mixin import (
    OptimizerParams,
    OptimizerParamsType,
    SchedulerParamsType,
)
from quantem.diffractive_imaging.object_models import ObjectINR, object_type
from quantem.ptycho_tomography.geometry import PtychoTomoPatchData, slab_z_centers


def _axis_coords(n: int, device, dtype) -> torch.Tensor:
    """Normalized voxel-center coordinates along one axis (matches ``idx/(N-1)*2-1``)."""
    if n == 1:
        return torch.zeros(1, device=device, dtype=dtype)
    return torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)


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


class ObjectPtychoTomoBase(ObjectINR):
    """Base for joint ptycho-tomography object models (rotated-coordinate query).

    Subclasses ``ObjectINR`` to inherit the implicit-object plumbing (``is_implicit`` wiring,
    optimizer/reset/pretrained-weights machinery, INR-style soft-constraint routing and the
    ``PtychoObjConstraintParams.INR`` constraint set). The differences:

    - ``forward`` consumes a :class:`PtychoTomoPatchData` payload (beam-frame patch coordinates +
      per-element beam->specimen rotations) instead of bare 2D coordinates.
    - z is the **specimen** axis of a real 3D volume, decoupled from the multislice slab count
      ``num_slices``; the model stores phase-density (rad/Å) and ``forward`` multiplies by the
      slab thickness.
    - the materialized object (``.obj`` / ``.volume``) is the specimen-frame volume on the
      ``volume_shape`` grid, not a per-slice stack.
    """

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
        thickness_A = float(thickness_A)
        if thickness_A <= 0:
            raise ValueError(f"thickness_A must be > 0, got {thickness_A}")
        super().__init__(
            model=model,
            num_slices=num_slices,
            slice_thicknesses=thickness_A / num_slices if num_slices > 1 else None,
            obj_type=obj_type,
            device=device,
            rng=rng,
            _token=_token,
        )
        self._thickness_A = thickness_A
        self._num_z_voxels = int(num_z_voxels) if num_z_voxels is not None else None
        # Explicit volume_shape (e.g. wrapping an existing array) is kept as-is; otherwise the
        # lateral grid is matched to the padded 2D object at preprocess (_initialize_obj) and the
        # z count defaults to ~cubic voxels (or num_z_voxels).
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
    def thickness_A(self) -> float:
        """Total beam-frame multislice extent == object-box z extent (Å)."""
        return self._thickness_A

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        """Storage/materialization grid ``(D, H, W)`` of the specimen-frame volume."""
        if self._volume_shape is None:
            raise ValueError(
                "volume_shape not set; pass it explicitly or run preprocess() "
                "(which matches the lateral grid to the padded object)."
            )
        return self._volume_shape

    @property
    def slab_thickness_A(self) -> float:
        """Thickness of each multislice slab (uniform), Å."""
        return self._thickness_A / self.num_slices

    @property
    def _slab_z_centers_t(self) -> torch.Tensor:
        real_dtype = getattr(torch, config.get("dtype_real"))
        return slab_z_centers(
            self.num_slices, self._thickness_A, device=self.device, dtype=real_dtype
        )

    @property
    def _box_half_extents(self) -> tuple[float, float, float]:
        """Physical half-extents ``(h_z, h_y, h_x)`` of the object box in Å.

        Lateral extents come from the padded 2D object grid set during preprocessing
        (``_initialize_obj``); z from ``thickness_A``.
        """
        if self._obj_shape is None:
            raise ValueError("Object shape not set; call preprocess() (or _initialize_obj) first.")
        samp = self.sampling
        _, h_full, w_full = self._obj_shape
        return (
            self._thickness_A / 2.0,
            (int(h_full) - 1) / 2.0 * float(samp[0]),
            (int(w_full) - 1) / 2.0 * float(samp[1]),
        )

    @property
    def volume(self) -> torch.Tensor:
        """Materialized specimen-frame volume ``(D, H, W)`` (display gauge applied)."""
        return self.obj

    # endregion --- properties ---

    def forward(self, patch_data: PtychoTomoPatchData, /) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride] -- payload seam
        """Rotated-coordinate object query.

        Builds beam-frame physical coordinates for every multislice slab center over each patch,
        rotates them into the specimen frame, queries the model, masks outside the object box to
        vacuum, and returns complex transmission patches ``exp(1j * value * slab_thickness)`` of
        shape ``(num_slices, batch, Hroi, Wroi)``.
        """
        self._invalidate_obj_cache()
        coords = patch_data.coords_yx  # (B, Hroi, Wroi, 2) normalized beam-frame (row, col)
        rot = patch_data.rotations.to(device=coords.device, dtype=coords.dtype)  # (B, 3, 3)
        h_z, h_y, h_x = self._box_half_extents
        y_b = coords[..., 0] * h_y  # (B, Hroi, Wroi), physical Å
        x_b = coords[..., 1] * h_x
        if (
            patch_data.shifts_px is not None
        ):  # per-tilt alignment shifts (beam frame, pre-rotation)
            samp = self.sampling
            y_b = y_b - (patch_data.shifts_px[:, 0] * float(samp[0])).view(-1, 1, 1)
            x_b = x_b - (patch_data.shifts_px[:, 1] * float(samp[1])).view(-1, 1, 1)
        z_centers = self._slab_z_centers_t.to(coords.dtype)  # (S,)
        t_slab = self.slab_thickness_A

        r = rot.view(-1, 3, 3)
        rzz, rzy, rzx = (r[:, 0, i].view(-1, 1, 1) for i in range(3))
        ryz, ryy, ryx = (r[:, 1, i].view(-1, 1, 1) for i in range(3))
        rxz, rxy, rxx = (r[:, 2, i].view(-1, 1, 1) for i in range(3))

        phases = []
        for s in range(self.num_slices):
            z_s = z_centers[s]
            n_z = (rzz * z_s + rzy * y_b + rzx * x_b) / h_z
            n_y = (ryz * z_s + ryy * y_b + ryx * x_b) / h_y
            n_x = (rxz * z_s + rxy * y_b + rxx * x_b) / h_x
            inside = (n_z.abs() <= 1.0) & (n_y.abs() <= 1.0) & (n_x.abs() <= 1.0)
            pts = torch.stack([n_z, n_y, n_x], dim=-1).reshape(-1, 3)
            vals = self._model(pts).reshape(coords.shape[:3])
            phases.append(vals * inside.to(vals.dtype) * t_slab)
        phase = torch.stack(phases, dim=0)  # (S, B, Hroi, Wroi)
        return torch.exp(1.0j * phase)

    def _initialize_obj(
        self,
        shape: "tuple[int, int, int] | np.ndarray",
        sampling: "tuple[float, float] | np.ndarray | None" = None,
    ) -> None:
        """Geometry handshake from preprocessing (padded shape + sampling).

        When ``volume_shape`` was not given explicitly, the volume's lateral grid is matched to
        the padded 2D object (so the materialized volume lives in the same pixel space as the
        scan positions and the inherited cropping works) and the z count defaults to ~cubic
        voxels (``thickness_A / mean(sampling)``) unless ``num_z_voxels`` was set.
        """
        super()._initialize_obj(shape, sampling)
        lat = (int(shape[1]), int(shape[2]))
        if self._volume_shape_explicit:
            if self._volume_shape is not None and self._volume_shape[1:] != lat:
                from warnings import warn

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
            d = max(1, round(self._thickness_A / float(np.mean(np.asarray(samp)))))
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

    # region --- pretraining ---
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
        """Warm-start the model by regressing it onto a specimen-frame volume.

        ``pretrain_target`` must have shape ``volume_shape`` (density in rad/Å). The fitted
        weights become the reset state. Mirrors ``ObjectINR.pretrain`` with the full-grid query
        replaced by the volume-grid query.
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

    # endregion --- pretraining ---

    # region --- constraints ---
    def apply_soft_constraints(
        self, obj: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Coordinate-sampled 3D TV + positivity penalties over the specimen volume.

        Unlike the 2.5D ``ObjectINR`` version, ``tv_weight_z`` penalizes the specimen z axis of
        the volume (always meaningful, independent of the multislice slab count) and
        ``tv_weight_xy`` the lateral axes.
        """
        self.reset_soft_constraint_losses()
        loss = self._get_zero_loss_tensor()
        w_z = self.constraints.tv_weight_z
        w_xy = self.constraints.tv_weight_xy
        if w_z > 0 or w_xy > 0:
            tv_loss = self._sampled_tv3d_loss(w_z, w_xy)
            loss = loss + tv_loss
            self.add_soft_constraint_loss("tv_loss", tv_loss)
        w_pos = self.constraints.positivity_weight
        if w_pos > 0 and self.obj_type == "potential":
            pos_loss = self._sampled_positivity_loss(w_pos)
            loss = loss + pos_loss
            self.add_soft_constraint_loss("positivity_loss", pos_loss)
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

    def _sampled_tv3d_loss(self, w_z: float, w_xy: float, num_samples: int = 4096) -> torch.Tensor:
        """Finite-difference TV over the specimen volume at randomly sampled coordinates."""
        real_dtype = getattr(torch, config.get("dtype_real"))
        coords = self._sample_volume_coords(num_samples)
        value = self._model(coords).squeeze(-1)
        loss = self._get_zero_loss_tensor()
        weights = (w_z, w_xy, w_xy)
        for axis in range(3):
            w = weights[axis]
            if w <= 0:
                continue
            h = 2.0 / max(int(self.volume_shape[axis]), 2)  # ~one voxel step
            offset = torch.zeros(3, device=self.device, dtype=real_dtype)
            offset[axis] = h
            shifted = self._model(coords + offset).squeeze(-1)
            loss = loss + w * torch.mean(torch.abs(shifted - value))
        return loss

    # endregion --- constraints ---


class ObjectVoxelTomo(ObjectPtychoTomoBase):
    """Dense voxel-grid backend — the easy-to-train baseline ptycho-tomography object.

    The specimen volume is a single ``(D, H, W)`` ``nn.Parameter`` (phase-density, rad/Å) queried
    by trilinear interpolation at the rotated coordinates. Dense direct parameterization gives
    well-conditioned gradients; use it to validate geometry and training recipes before moving to
    the K-Planes backend.

    For ``obj_type="potential"`` the density is hard-projected non-negative after every optimizer
    step (``project_parameters``, called by the reconstruction loop) — mirroring the validated
    ``ObjectPixelated`` positivity behavior and suppressing the phase-winding instability. Set
    ``hard_positivity = False`` to opt out (soft ``positivity_weight`` only).

    ``shrink_quantile`` additionally subtracts each z-plane's low-quantile level before clamping
    (per-plane background shrinkage, the volume analog of ``positivity_mode="shrink"``): the
    L2-amplitude data term is nearly blind to a smooth positive background, so without it haze
    accumulates throughout the volume (one-sided positivity + bright-field-dominated loss). Valid
    while every z-plane is mostly vacuum laterally (isolated samples; quantile below the vacuum
    area fraction).
    """

    hard_positivity: bool = True
    shrink_quantile: float | None = None

    @property
    def name(self) -> str:
        return "ObjectVoxelTomo"

    def project_parameters(self) -> None:
        """Post-step hard projection: per-plane background shrinkage + non-negativity clamp."""
        if not self.hard_positivity or self.obj_type != "potential":
            return
        model = self._model
        assert isinstance(model, VoxelGrid)
        with torch.no_grad():
            if self.shrink_quantile is not None and 0.0 < self.shrink_quantile < 1.0:
                planes = model.volume.view(model.volume.shape[0], -1)
                floor = torch.quantile(planes, self.shrink_quantile, dim=1, keepdim=True)
                planes.sub_(torch.clamp(floor, min=0.0))
            model.volume.clamp_(min=0.0)
        self._invalidate_obj_cache()

    @classmethod
    def from_uniform(  # pyright: ignore[reportIncompatibleMethodOverride] -- model-family factory
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
        (default ~cubic voxels). Pass an explicit ``volume_shape`` for standalone use.
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
        ``set_as_initial`` the loaded volume becomes the ``reset()`` state.
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
    :class:`CPTilted`). The models consume the same ``(N, 3)`` ``(z, y, x)`` coordinates, so every
    rotated-query path of :class:`ObjectPtychoTomoBase` (forward, materialization, sampled-3D-TV,
    pretrain) is reused unchanged; only the optimizer wiring differs.

    These models expose multiple parameter groups (``grids``/``sigma_net``, plus ``so3`` for the
    tilted variants), so ``optimizer_params`` must be a PPLR dict keyed by ``model.param_keys``,
    e.g. ``{"grids": OptimizerParams.Adam(lr=1e-2), "sigma_net": OptimizerParams.Adam(lr=1e-3)}``.
    ``resolution`` is the feature-plane resolution ``(z, y, x)``, independent of both the padded
    object grid and ``volume_shape`` (which only sets the materialization grid here).
    """

    @property
    def name(self) -> str:
        return "ObjectKPlanesTomo"

    @property
    def model(self) -> KPlanesType:
        return cast(KPlanesType, self._model)

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
    def from_uniform(  # pyright: ignore[reportIncompatibleMethodOverride] -- model-family factory
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


ObjectPtychoTomoType = ObjectVoxelTomo | ObjectKPlanesTomo
