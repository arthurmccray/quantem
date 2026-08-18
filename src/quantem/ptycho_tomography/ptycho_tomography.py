"""Joint ptychography-tomography reconstruction class.

``PtychoTomography`` is a thin subclass of :class:`Ptychography`: the entire reconstruction loop
(batching, losses, optimizers/schedulers, snapshots, multi-GPU spawn, torchrun) is inherited
unchanged. The tilt series enters through the paired models — a
:class:`~quantem.ptycho_tomography.dataset_models.PtychoTomoDatasetRaster` (flat ``tilt x scan``
index space, rotation-carrying object-query payload) and a rotation-aware
:class:`~quantem.ptycho_tomography.object_models.ObjectPtychoTomoBase` object (one 3D specimen
volume queried at rotated coordinates per tilt). This class only adapts construction/validation,
preprocessing geometry, snapshots, and visualization to the 3D object.

Typical use::

    dset = PtychoTomoDatasetRaster.from_dataset4dstem_list(dsets_4d, tilt_angles_deg)
    obj = ObjectVoxelTomo.from_uniform(thickness_A=56.0, num_slices=16)
    probe = ProbeParametric.from_params(probe_params={"energy": 300e3, ...})
    pt = PtychoTomography.from_models(dset, obj, probe, DetectorPixelated(), device="gpu")
    pt.preprocess(specimen_box_A=(84.0, 84.0, 84.0))  # physical box (Å); see preprocess()
    pt.reconstruct(num_iters=300, store_snapshots_every=25,
                   optimizer_params={"object": ..., "probe": ...})
    vol = pt.volume_cropped  # (D, h, w) specimen-frame density, rad/Å (padding cropped)
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Self, Sequence, cast
from warnings import warn

import numpy as np
import torch

from quantem.core import config
from quantem.diffractive_imaging.detector_models import DetectorModelType
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.probe_models import ProbeModelType
from quantem.diffractive_imaging.ptychography import Ptychography
from quantem.ptycho_tomography.dataset_models import PtychoTomoDatasetRaster
from quantem.ptycho_tomography.object_models import ObjectPtychoTomoBase
from quantem.ptycho_tomography.ptycho_tomography_visualizations import (
    PtychoTomographyVisualizations,
)


class PtychoTomography(PtychoTomographyVisualizations, Ptychography):
    """Joint multislice-ptychography + tomography reconstruction of one 3D object."""

    # cached (key, ROI k² grid) for the slab-window probe pre-propagation (built lazily per
    # (roi_shape, sampling); plain attr, not module state)
    _window_k2_cache: "tuple[tuple[int, int, float, float], torch.Tensor] | None" = None
    # per-tilt pose / defocus state carried across a save -> from_file(dset=fresh_wrapper)
    # round trip (the dataset itself is excluded from the archive by default, and the caller
    # rebuilds a FRESH wrapper whose pose slots are back at their defaults). Written by save(),
    # re-applied by from_file(); see _collect_pose_state.
    _pose_state_metadata: "dict[str, Any] | None" = None

    @classmethod
    def from_models(  # pyright: ignore[reportIncompatibleMethodOverride] -- narrowed model types
        cls,
        dset: PtychoTomoDatasetRaster,
        obj_model: ObjectPtychoTomoBase,
        probe_model: ProbeModelType,
        detector_model: DetectorModelType,
        logger: LoggerPtychography | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        # isinstance with autoreload-bug string fallback, matching the base-class style
        if not (
            isinstance(dset, PtychoTomoDatasetRaster) or "PtychoTomoDataset" in str(type(dset))
        ):
            raise TypeError(
                f"dset must be a PtychoTomoDatasetRaster (tilt series), got {type(dset)}"
            )
        if not (
            isinstance(obj_model, ObjectPtychoTomoBase) or "PtychoTomo" in str(type(obj_model))
        ):
            raise TypeError(
                f"obj_model must be an ObjectPtychoTomoBase subclass (rotation-aware), got "
                f"{type(obj_model)}"
            )
        return cls(
            dset=dset,
            obj_model=obj_model,
            probe_model=probe_model,
            detector_model=detector_model,
            logger=logger,
            device=device,
            verbose=verbose,
            rng=rng,
            _token=cls._token,
        )

    @classmethod
    def from_file(  # pyright: ignore[reportIncompatibleMethodOverride] -- narrowed dset type
        cls,
        path: "str | Path",
        dset: PtychoTomoDatasetRaster | None = None,
        device: str | int | None = None,
        verbose: int | bool | None = None,
        auto_reload_dataset: bool = True,
    ) -> "PtychoTomography":
        """Load a saved reconstruction, attaching a (rebuilt + preprocessed) tilt-series dataset.

        Tilt-series saves exclude the raw data by default, so pass ``dset=`` (rebuild the wrapper
        from the per-tilt files and preprocess it with the same parameters). The attached
        wrapper's ``implicit_object`` flag is re-synced here — a freshly built wrapper defaults to
        False and the base ``from_file`` does not re-run the object-model wiring.
        """
        pt = super().from_file(
            path,
            # the wrapper is a sibling of the single-scan raster class, not a subclass, so the
            # static type can't narrow; runtime duck-typing through the base from_file is fine
            dset=cast("Any", dset),
            device=device,
            verbose=verbose,
            auto_reload_dataset=auto_reload_dataset,
        )
        if not isinstance(pt, PtychoTomography):
            raise TypeError(f"loaded object is not a PtychoTomography, got {type(pt)}")
        if dset is not None:
            pt.dset.implicit_object = pt.obj_model.is_implicit
            pt._apply_pose_state(pt._pose_state_metadata)
        return pt

    # region --- pose / defocus resume state ---
    _POSE_STATE_PARAMS = (
        "_pose_shifts",
        "_pose_z1",
        "_pose_z3",
        "_pose_dtheta",
        "_defocus_offset_A",
    )
    _POSE_STATE_BUFFERS = (
        "_pose_shifts_init",
        "_pose_z1_init",
        "_pose_z3_init",
        "_pose_dtheta_init",
        "_defocus_offset_init_A",
    )

    def _collect_pose_state(self) -> "dict[str, Any]":
        """Snapshot the dataset's pose/defocus parameters, baselines and learn flags (CPU)."""
        dset_t = cast(PtychoTomoDatasetRaster, self.dset)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        dset_t._ensure_pose_state()
        state: dict[str, Any] = {"reference_tilt_idx": dset_t.reference_tilt_idx}
        for name in self._POSE_STATE_PARAMS:
            state[name] = getattr(dset_t, name).data.detach().cpu().clone()
        for name in self._POSE_STATE_BUFFERS:
            state[name] = getattr(dset_t, name).detach().cpu().clone()
        for flag in PtychoTomoDatasetRaster._POSE_LEARN_FLAGS:
            state[flag] = bool(getattr(dset_t, flag))
        return state

    def _apply_pose_state(self, state: "dict[str, Any] | None") -> None:
        """Re-apply a :meth:`_collect_pose_state` snapshot onto the attached dataset.

        A rebuilt wrapper starts from default (zero, frozen) pose slots, so without this a
        ``--resume-from`` would silently drop every learned pose and defocus offset and restart
        the refinement from scratch (gotcha #14: the resume handshake is where geometry state
        goes missing). Mismatched tilt counts are skipped with a warning rather than raising —
        the object is still usable, just without the pose history.
        """
        if not state:
            return
        dset_t = cast(PtychoTomoDatasetRaster, self.dset)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        dset_t._ensure_pose_state()
        n = dset_t.num_tilts
        saved = state.get("_pose_z1")
        if saved is not None and int(saved.shape[0]) != n:
            warn(
                f"saved pose state has {int(saved.shape[0])} tilts but the attached dataset has "
                f"{n}; pose/defocus state was NOT restored",
                stacklevel=2,
            )
            return
        with torch.no_grad():
            for name in self._POSE_STATE_PARAMS:
                value = state.get(name)
                if value is not None:
                    target = getattr(dset_t, name)
                    target.data.copy_(value.to(device=target.device, dtype=target.dtype))
            for name in self._POSE_STATE_BUFFERS:
                value = state.get(name)
                if value is not None:
                    target = getattr(dset_t, name)
                    target.copy_(value.to(device=target.device, dtype=target.dtype))
        idx = state.get("reference_tilt_idx")
        if idx is not None:
            dset_t.reference_tilt_idx = int(idx)
        dset_t.set_learn_pose(
            shifts=bool(state.get("_learn_pose_shifts", False)),
            z1=bool(state.get("_learn_pose_z1", False)),
            z3=bool(state.get("_learn_pose_z3", False)),
            dtheta=bool(state.get("_learn_pose_dtheta", False)),
        )
        dset_t.set_learn_defocus(bool(state.get("_learn_defocus", False)))

    # endregion --- pose / defocus resume state ---

    # region --- preprocessing ---
    def preprocess(  # pyright: ignore[reportIncompatibleMethodOverride] -- physical-units geometry
        self,
        specimen_box_A: "tuple[float, float] | tuple[float, float, float] | None" = None,
        box_margin_A: tuple[float, float, float] = (0.0, 0.0, 0.0),
        val_ratio: float = 0.0,
        val_mode: Literal["grid", "random"] = "grid",
        vectorized: bool = True,
        batch_size: int | None = None,  # accepted for signature parity; no probe-overlap mask
        com_fit_function: Literal["none", "plane", "parabola", "constant", "no_shift"] = (
            "constant"
        ),
        force_com_rotation: float | None = 0.0,
        force_com_transpose: bool | None = False,
        padded_diffraction_intensities_shape: tuple[int, int] | None = None,
        plot_rotation: bool = False,
        plot_com: str | bool = False,
        plot_probe_overlap: bool = False,
        *,
        probe_energy: float | None = None,
        free_per_tilt_arrays: bool = True,
        obj_padding_px: tuple[int, int] | None = None,  # DEPRECATED
        z_padding_px: int | None = None,  # DEPRECATED
    ) -> Self:
        """Preprocess the tilt series and run the physical-units geometry handshake.

        Geometry (2026-07-10 refactor — all in Å, shared by every backend):

        - ``specimen_box_A``: the physical object box (crop target), ``(by, bx)`` lateral or
          ``(bz, by, bx)`` (``bz`` must equal the object's ``thickness_A``). Axis-aligned with
          the object grid in the specimen frame; its center is the coordinate origin and the
          tilt axis passes through it (the scan-footprint center is anchored there). Default:
          scan-grid extent plus one probe-ROI extent laterally (covers probe tails) —
          the right semantic for plan-view samples too. For simulated data pass the true box
          (e.g. the AuNP's cubic 84 Å box → cubic ``volume_cropped``, square projections).
        - ``box_margin_A``: ``(mz, my, mx)`` extra support margin OUTSIDE the box per side,
          default zeros. Only useful when true material extends beyond the box (plan-view /
          real data); under tilt the beam samples up to ``±(bz/2)·sin(θ_max)`` laterally beyond
          the 0° footprint. ``volume_cropped`` removes margins on ALL three axes.

        Tilt-series specifics as before: a common (usually zero) CoM rotation is forced across
        tilts, and the 2D probe-overlap FOV mask is replaced by a trivial mask.
        """
        del batch_size, plot_rotation, plot_com, plot_probe_overlap  # unused (parity only)
        if padded_diffraction_intensities_shape is not None:
            raise NotImplementedError(
                "padded_diffraction_intensities_shape is not supported for tilt series yet"
            )
        if force_com_rotation is None or force_com_transpose is None:
            raise ValueError(
                "force_com_rotation / force_com_transpose must be explicit for a tilt series "
                "(per-tilt solved rotations would break the shared geometry); use 0.0 / False "
                "for simulated data."
            )

        # ---- deprecated pixel-padding arguments (legacy mapping keeps old runs reproducible)
        legacy = obj_padding_px is not None or z_padding_px is not None
        if legacy:
            warn(
                "obj_padding_px / z_padding_px are deprecated - use specimen_box_A (Å) and "
                "box_margin_A (Å); the pixel arguments will be removed",
                DeprecationWarning,
                stacklevel=2,
            )
            print(
                "[DEPRECATED] preprocess(obj_padding_px/z_padding_px) -> "
                "preprocess(specimen_box_A=..., box_margin_A=...); will be removed"
            )
            if specimen_box_A is not None:
                raise ValueError("pass either specimen_box_A or the deprecated pixel paddings")
        _pad = obj_padding_px if obj_padding_px is not None else (0, 0)
        pad_yx: tuple[int, int] = (int(_pad[0]), int(_pad[1]))

        # cast: dset is validated as a PtychoTomoDatasetRaster in from_models (it is a
        # sibling of the single-scan raster class, so isinstance narrowing can't apply)
        dset_t = cast(PtychoTomoDatasetRaster, self.dset)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        if not self.dset.preprocessed:
            self.vprint("Dataset was not preprocessed, proceeding with defaults.")
            dset_t.preprocess(
                com_fit_function=com_fit_function,
                force_com_rotation=force_com_rotation,
                force_com_transpose=force_com_transpose,
                obj_padding_px=pad_yx,
                probe_energy=probe_energy,
                free_per_tilt_arrays=free_per_tilt_arrays,
                vectorized=vectorized,
            )
            self._probe_model.set_initial_probe(
                self.roi_shape,
                self.reciprocal_sampling,
                self.dset.mean_diffraction_intensity,
                device=self._single_device,
            )

        obj = cast(ObjectPtychoTomoBase, self.obj_model)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        samp = np.asarray(self.dset.obj_sampling, dtype=float)  # (y, x), Å/px
        s = float(np.mean(samp))

        # base machinery attrs first: the setter may power-2-adjust the padding and re-derives
        # scan positions, so the coordinate origin is set from the FINAL padding afterwards.
        # Suppress the setter's legacy pixel handshake — set_geometry below is authoritative
        # (documented seam in ObjectPtychoTomoBase._initialize_obj).
        obj._geometry_set = True
        self.obj_padding_px = pad_yx
        dset_t._set_scan_center(self.obj_padding_px)

        # ---- resolve the physical geometry
        margins: tuple[float, float, float] = (
            float(box_margin_A[0]),
            float(box_margin_A[1]),
            float(box_margin_A[2]),
        )
        if legacy:
            # compat mapping: reproduces the old padded-grid extents exactly —
            # lateral box = un-padded grid point-extent, margins = padding in Å
            full2d_nopad = self.dset._obj_shape_full_2d((0, 0))
            lateral_box = (
                (int(full2d_nopad[0]) - 1) * float(samp[0]),
                (int(full2d_nopad[1]) - 1) * float(samp[1]),
            )
            p_final = tuple(int(p) for p in np.asarray(self.obj_padding_px).ravel())
            margins = (
                float(z_padding_px or 0) * s,
                p_final[0] * float(samp[0]),
                p_final[1] * float(samp[1]),
            )
        elif specimen_box_A is not None:
            box = tuple(float(v) for v in np.asarray(specimen_box_A, dtype=float).ravel())
            if len(box) == 3:
                if not np.isclose(box[0], obj.thickness_A, rtol=1e-6):
                    raise ValueError(
                        f"specimen_box_A z extent ({box[0]} Å) must equal the object's "
                        f"thickness_A ({obj.thickness_A} Å) — set thickness_A at construction"
                    )
                lateral_box = (box[1], box[2])
            elif len(box) == 2:
                lateral_box = (box[0], box[1])
            else:
                raise ValueError(f"specimen_box_A must have 2 or 3 components, got {box}")
        else:
            # default: scan-grid extent + one probe-ROI extent (probe-tail coverage);
            # plan-view-appropriate (there is no "true" box for extended samples)
            full2d_nopad = self.dset._obj_shape_full_2d((0, 0))
            roi = np.asarray(self.roi_shape, dtype=float)
            lateral_box = (
                (int(full2d_nopad[0]) - 1) * float(samp[0]) + float(roi[0]) * float(samp[0]),
                (int(full2d_nopad[1]) - 1) * float(samp[1]) + float(roi[1]) * float(samp[1]),
            )

        obj.set_geometry(lateral_box_A=lateral_box, sampling=samp, box_margin_A=margins)
        self.compute_propagator_arrays()

        # trivial FOV mask (ndim-3 expanded by the setter); obj_model.mask stays empty so the
        # sampled-coordinate constraints skip it
        full2d = self.dset._obj_shape_full_2d(self.obj_padding_px)
        self.obj_fov_mask = np.ones(tuple(int(v) for v in full2d), dtype=config.get("dtype_real"))

        self._check_slab_coverage()
        self._preprocessed = True
        self.val_ratio = float(val_ratio)
        self.val_mode = val_mode
        return self

    def _check_slab_coverage(self) -> None:
        """Warn if the beam-frame multislice slab cannot cover the rotated object support.

        Classic mode: the slab spans the padded box along the beam; warn on the clear
        inconsistency ``slab extent < box thickness`` (always wrong). Slab-window mode
        (2026-07-16): each scan position gets its own window centered on the specimen mid-plane,
        so the necessary extent is the box thickness seen along the tilted beam plus the patch's
        z-spread: ``box_thickness/cos(θmax) + 2·tan(θmax)·r_patch`` — warn below that. Coverage
        failures are otherwise SILENT (out-of-support quadrature points contribute zero), so
        this check is the only runtime guard.
        """
        obj = self.obj_model
        if not isinstance(obj, ObjectPtychoTomoBase):  # pragma: no cover - guarded by from_models
            return
        slab = obj.slab_thickness_A * obj.num_slices
        dset = self.dset
        if isinstance(dset, PtychoTomoDatasetRaster) and dset.slab_window:
            theta = float(torch.deg2rad(dset._tilt_angles_deg.abs().max()))
            c, t = float(np.cos(theta)), float(np.tan(theta))
            samp = np.asarray(self.sampling, dtype=float).ravel()[-2:]  # lateral (y, x)
            r_patch = float(np.max(np.asarray(self.roi_shape) * samp)) / 2.0
            required = obj.box_thickness_A / max(c, 1e-3) + 2.0 * t * r_patch
            if slab < required * (1 - 1e-6):
                warn(
                    f"slab-window mode: slab extent ({slab:.1f} Å) < box/cos(θmax) + "
                    f"2·tan(θmax)·r_patch ({required:.1f} Å at θmax = {np.rad2deg(theta):.0f}°); "
                    "material at the patch edges will be silently cropped along the beam.",
                    stacklevel=2,
                )
        elif slab < obj.box_thickness_A * (1 - 1e-6):
            warn(
                f"multislice slab extent ({slab:.2f} Å) is smaller than the object box thickness "
                f"({obj.box_thickness_A:.2f} Å); the rotated object will be cropped along the "
                "beam.",
                stacklevel=2,
            )

    def compute_propagator_arrays(self):
        """Base propagators, but with the LATERAL (y, x) sampling explicitly.

        ``self.sampling`` is a (z, y, x) triple for the 3D object; the base call passes it
        straight into a 2-element zip against ``roi_shape``, silently building the row k-grid
        with the z-voxel spacing (off from the lateral pixel by <0.1% after the point-grid
        snapping). Harmless in practice but wrong semantics — and the slab-window probe
        pre-propagation must share the exact k-grid with the inter-slice propagators.
        """
        samp = np.asarray(self.sampling, dtype=float).ravel()[-2:]
        self.propagators = self.probe_model._compute_propagator_arrays(
            (float(samp[0]), float(samp[1])), self.num_slices, self.slice_thicknesses
        )

    def forward_operator(
        self,
        obj_patches: torch.Tensor,
        shifted_input_probes: torch.Tensor,
        descan: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Base forward operator plus the probe z-offset pre-propagation (2026-07-16 / 08-17).

        The dataset stashes ``_last_probe_dz_A`` (Å along the beam) during ``dset.forward``:
        the slab-window offset MINUS the per-tilt learned defocus offset. The physical probe is
        fixed in the lab, so the probe entering a window displaced by ``dz`` is the probe
        Fresnel-propagated by ``dz`` — the same ``exp(-iπ λ dz k²)`` factor as the inter-slice
        propagators, per batch element — and a defocus offset is the same operation with the
        opposite sign (``effective defocus = probe defocus + offset``; see
        ``PtychoTomoDatasetRaster.forward``). Note this is deliberately NOT
        ``_last_window_dz_A``: that one also moves the object query, and the defocus must move
        only the probe. The stash is consumed exactly once per batch step (train and validation
        both rebuild it via ``dset.forward``); nothing changes when both are off (``None``).
        """
        dz = getattr(self.dset, "_last_probe_dz_A", None)
        if dz is None:  # pre-defocus wrappers stash only the window offset
            dz = getattr(self.dset, "_last_window_dz_A", None)
        if dz is not None:
            # consume-once: no staleness, clean serialization
            dset_t = cast(PtychoTomoDatasetRaster, self.dset)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
            dset_t._last_probe_dz_A = None
            dset_t._last_window_dz_A = None
            shifted_input_probes = self._pre_propagate_probes(shifted_input_probes, dz)
        return super().forward_operator(obj_patches, shifted_input_probes, descan)

    def _pre_propagate_probes(self, probes: torch.Tensor, dz_A: torch.Tensor) -> torch.Tensor:
        """Fresnel-propagate probes ``(P, B, H, W)`` by per-batch ``dz_A`` (B,) in Å."""
        from quantem.core.utils.utils import electron_wavelength_angstrom

        h, w = int(self.roi_shape[0]), int(self.roi_shape[1])
        samp = np.asarray(self.sampling, dtype=float).ravel()[-2:]  # lateral (y, x)
        key = (h, w, float(samp[0]), float(samp[1]))
        cache = self._window_k2_cache
        # device check: a deserialized cache comes back on CPU (plain attr, not moved by .to())
        if cache is None or cache[0] != key or cache[1].device != probes.device:
            kr = torch.fft.fftfreq(h, d=float(samp[0]), device=probes.device)
            kc = torch.fft.fftfreq(w, d=float(samp[1]), device=probes.device)
            k2 = (kr[:, None] ** 2 + kc[None, :] ** 2).to(self._dtype_real)
            cache = (key, k2)
            self._window_k2_cache = cache
        lam = float(electron_wavelength_angstrom(self.probe_model.probe_params["energy"]))
        dz = dz_A.to(device=probes.device, dtype=self._dtype_real)
        phase = torch.exp(-1.0j * torch.pi * lam * dz.view(1, -1, 1, 1) * cache[1][None, None])
        return torch.fft.ifft2(torch.fft.fft2(probes) * phase)

    # endregion --- preprocessing ---

    def _soft_constraints(self) -> torch.Tensor:
        """Soft constraints with the object's penalties evaluated at sampled coordinates.

        Replaces the base class's ``isinstance(obj_model, ObjectINR)`` branch: every
        ptycho-tomography object model is coordinate-queried, so its soft constraints never
        require the materialized volume (which would force a full-grid query each batch).
        """
        obj_model = cast(ObjectPtychoTomoBase, self.obj_model)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        total_loss = torch.tensor(0, device=self._single_device, dtype=self._dtype_real)
        total_loss = total_loss + obj_model.apply_soft_constraints(mask=obj_model.mask)
        total_loss = total_loss + self.probe_model.apply_soft_constraints(self.probe_model.probe)
        total_loss = total_loss + self.dset.apply_soft_constraints(self.dset.descan_shifts)
        return total_loss

    # region --- snapshots ---
    def _store_current_iter_snapshot(self) -> None:
        """Checkpoint the object backend at the current iteration (object-only, lightweight).

        Stores a CPU copy of the backend ``state_dict`` rather than a materialized volume: tiny
        for the K-Planes backend (feature grids + decoder) and at most the volume size for the
        dense voxel grid, but never forced onto the GPU. ``get_snapshot_by_iter`` re-materializes
        the volume on demand. The probe is shared across tilts and is not snapshotted.
        """
        obj_model = cast(ObjectPtychoTomoBase, self.obj_model)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        state = {k: v.detach().cpu().clone() for k, v in obj_model.model.state_dict().items()}
        # snapshots carry an object state_dict rather than the base (obj, probe) arrays
        self._snapshots.append({"iteration": self.num_iters, "state_dict": state})  # pyright: ignore[reportArgumentType] -- tomo snapshot payload; the Snapshot TypedDict lives in diffractive_imaging

    def get_snapshot_by_iter(  # pyright: ignore[reportIncompatibleMethodOverride] -- volume snapshot
        self, iteration: int, closest: bool = False, cropped: bool = True
    ) -> dict[str, Any]:
        """Materialize the object volume checkpointed at a stored iteration.

        Snapshots hold only the backend ``state_dict``; this temporarily loads it into the live
        backend, reads the volume (cropped to the specimen by default), then restores the current
        state. Returns ``{"iteration", "obj"}`` where ``obj`` is the specimen-frame volume
        ``(D, h, w)`` (``cropped=True``) or the padded ``(D, H, W)`` volume.
        """
        if len(self._snapshots) == 0:
            raise ValueError(
                "No snapshots available. Pass store_snapshots_every=... to reconstruct()."
            )
        iteration = int(iteration)
        if iteration < 0:
            iteration = self.num_iters + iteration
        iters = [s["iteration"] for s in self._snapshots]
        if closest:
            snp = min(self._snapshots, key=lambda s: abs(s["iteration"] - iteration))
        elif iteration in iters:
            snp = self._snapshots[iters.index(iteration)]
        else:
            raise ValueError(
                f"No snapshot at iteration {iteration}; set closest=True for the nearest stored."
            )
        obj_model = cast(ObjectPtychoTomoBase, self.obj_model)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        model = obj_model.model
        saved = deepcopy(model.state_dict())
        try:
            model.load_state_dict(
                {k: v.to(obj_model.device) for k, v in snp["state_dict"].items()}  # pyright: ignore[reportGeneralTypeIssues] -- tomo snapshot key; the Snapshot TypedDict lives in diffractive_imaging
            )
            obj_model._invalidate_obj_cache()
            vol = self.volume_cropped if cropped else self.volume
        finally:
            model.load_state_dict(saved)
            obj_model._invalidate_obj_cache()
        return {"iteration": snp["iteration"], "obj": vol}

    # endregion --- snapshots ---

    # region --- properties ---
    @property
    def sampling(self) -> np.ndarray:  # pyright: ignore[reportIncompatibleMethodOverride] -- 3D
        """``(z, y, x)`` voxel size of the reconstruction volume, Å. Cubic by construction:
        y == x exactly (enforced), z equal to <0.1% (grid rounding over the box thickness)."""
        obj = self.obj_model
        assert isinstance(obj, ObjectPtychoTomoBase)
        lat = np.asarray(self.dset.obj_sampling, dtype=float)
        try:
            zv = obj.z_voxel_A
        except ValueError:  # volume grid not allocated yet (mid-construction, pre-preprocess)
            zv = float(lat[0])  # z voxel == lateral pixel by design once geometry is set
        return np.array([zv, lat[0], lat[1]], dtype=float)

    @property
    def obj_shape_crop(self) -> np.ndarray:
        """``(D, h, w)`` of the specimen box (all support margins removed on all three axes)."""
        obj = self.obj_model
        assert isinstance(obj, ObjectPtychoTomoBase)
        if obj.lateral_box_A is not None:
            sl = obj.crop_slices
            d, h, w = obj.volume_shape
            return np.array(
                [
                    len(range(*sl[0].indices(d))),
                    len(range(*sl[1].indices(h))),
                    len(range(*sl[2].indices(w))),
                ]
            )
        # legacy fallback (pre-set_geometry objects): scan-FOV lateral crop, thickness z crop
        shp = np.floor(self.dset.fov / np.asarray(self.dset.obj_sampling, dtype=float))
        shp += shp % 2
        d_specimen = max(1, round(obj.thickness_A / obj.z_voxel_A))
        d_specimen = min(d_specimen, obj.volume_shape[0])
        return np.concatenate([[d_specimen], shp]).astype("int")

    @property
    def obj_cropped(self) -> np.ndarray:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Specimen-box volume: the support volume with ALL margins cropped (z, y and x).

        For a cubic specimen box (e.g. the AuNP's 84 Å cube) this is a cubic array — square
        projections along every axis. Falls back to the legacy FOV crop for objects loaded
        without the physical-box geometry.
        """
        obj = self.obj_model
        assert isinstance(obj, ObjectPtychoTomoBase)
        if obj.lateral_box_A is not None:
            return self._to_numpy(self.obj_model.obj[obj.crop_slices])
        return super().obj_cropped

    @property
    def volume(self) -> np.ndarray:
        """Materialized specimen-frame volume ``(D, H, W)`` (density, rad/Å) on the padded grid
        (includes z padding; use ``volume_cropped`` for the specimen)."""
        return self.obj

    @property
    def volume_cropped(self) -> np.ndarray:
        """Specimen-frame volume cropped to the scan FOV and specimen thickness, ``(D, h, w)``.

        Intentionally **non-cubic**: z spans the full specimen thickness (``thickness_A``) while
        the lateral dims span the scan FOV. For a compact object smaller than the box this shows
        vacuum headroom along z that the (often narrower) scan FOV crops away laterally — so depth
        sections can look vacuum-padded top/bottom relative to the projection. Use
        ``volume_cropped_cubic`` for a consistent cubic view.
        """
        return self.obj_cropped

    @property
    def volume_cropped_cubic(self) -> np.ndarray:
        """``volume_cropped`` center-cropped to a cube (side = the smallest of its three extents).

        Convenience view for consistent display when ``volume_cropped`` is non-cubic (z = full
        specimen thickness vs lateral = scan FOV). Does not change the reconstruction or the
        default ``volume``/``volume_cropped`` returned for scoring.
        """
        vol = self.volume_cropped
        n = int(min(vol.shape))
        sl = tuple(slice((s - n) // 2, (s - n) // 2 + n) for s in vol.shape)
        return vol[sl]

    @property
    def thickness_A(self) -> float:
        obj = self.obj_model
        assert isinstance(obj, ObjectPtychoTomoBase)
        return obj.thickness_A

    @property
    def z_sampling(self) -> float:
        """Specimen-frame z voxel size of the volume grid (Å)."""
        obj = self.obj_model
        assert isinstance(obj, ObjectPtychoTomoBase)
        return obj.z_voxel_A

    # endregion --- properties ---

    def save(
        self,
        path: "str | Path",
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: "str | type | Sequence[str | type]" = (),
        compression_level: int | None = 4,
        save_raw_data: bool = False,
        verbose: int | bool = True,
    ):
        """Save the reconstruction (raw tilt data excluded by default — rebuild the wrapper and
        attach it via ``from_file(path, dset=...)`` to visualize or continue training).

        Iteration snapshots (lightweight object state_dicts) round-trip like the base class's;
        pass ``skip=("_snapshots",)`` to drop them for a leaner save.

        Because the dataset is excluded, the per-tilt pose/defocus parameters (which live on
        the wrapper) are stashed in ``_pose_state_metadata`` here and re-applied by
        ``from_file`` onto the freshly rebuilt wrapper — otherwise a resume would silently
        restart the pose refinement from its defaults. Cleared afterwards so a live object
        never carries a stale copy.
        """
        stashed = not save_raw_data and getattr(self, "_dset", None) is not None
        if stashed:
            self._pose_state_metadata = self._collect_pose_state()
        try:
            return super().save(
                path,
                mode=mode,
                store=store,
                skip=skip,
                compression_level=compression_level,
                save_raw_data=save_raw_data,
                verbose=verbose,
            )
        finally:
            if stashed:
                self._pose_state_metadata = None
