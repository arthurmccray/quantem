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

Stack backend (2026-11, a few mrad between scans of one zone axis): pair the wrapper with
``ObjectStackTomo`` (a single-scan multislice stack, optionally a DIP) and preprocess with the
single-scan lateral padding, ``pt.preprocess(obj_padding_px=(p, p))``; each dataset's tilt is
then applied as a tilted propagator in ``forward_operator`` (see there).
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Self, Sequence, cast

import numpy as np
import torch

from quantem.core import config
from quantem.core.utils.utils import generate_batches
from quantem.diffractive_imaging.detector_models import DetectorModelType
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.probe_models import ProbeModelType
from quantem.diffractive_imaging.ptycho_utils import fourier_translation_operator, sum_patches
from quantem.diffractive_imaging.ptychography import Ptychography
from quantem.diffractive_imaging.ptychography_base import PtychographyBase
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
    # pose-refinement state carried across save/from_file (the dataset itself is not saved):
    # stashed by ``save`` like the base ``_dataset_metadata``, re-applied by ``from_file``
    _pose_metadata: "dict[str, Any] | None" = None
    # per-iteration learned-shift history (rank 0), appended by ``_record_iter``
    _pose_history: "list[dict[str, Any]] | None" = None
    # cached (key, kr, kc) lateral frequency axes (1/Å) for the stack backend's tilt ramps
    _tilt_k_cache: "tuple[tuple[int, int, float, float], torch.Tensor, torch.Tensor] | None" = None

    @property
    def obj_model(self):  # pyright: ignore[reportIncompatibleVariableOverride] -- same type, extra wiring
        return self._obj_model

    @obj_model.setter
    def obj_model(self, model) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Base wiring (``implicit_object``) plus the stack-backend flag on the dataset."""
        fset = PtychographyBase.obj_model.fset
        assert fset is not None
        fset(self, model)
        if hasattr(self, "_dset"):
            self._sync_whole_pixel_flag()

    def _sync_whole_pixel_flag(self) -> None:
        if hasattr(self.dset, "gathers_whole_pixels"):
            whole = bool(getattr(self._obj_model, "gathers_whole_pixels", False))
            cast(PtychoTomoDatasetRaster, self.dset).gathers_whole_pixels = whole  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam

    @property
    def _is_stack(self) -> bool:
        """Whether the object backend is the multislice stack (``ObjectStackTomo``)."""
        return bool(getattr(self._obj_model, "gathers_whole_pixels", False))

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
            pt._sync_whole_pixel_flag()
            if pt._pose_metadata is not None:
                pt._apply_pose_metadata(pt._pose_metadata)
        return pt

    # region --- pose refinement state (save / resume) ---
    def _collect_pose_metadata(self) -> "dict[str, Any]":
        dset = cast(PtychoTomoDatasetRaster, self.dset)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        return {
            "pose_shifts": dset.pose_shifts_A.cpu(),
            "pose_shifts_init": dset.pose_shifts_init_A.cpu(),
            "learn_pose_shifts": bool(dset.learn_pose_shifts),
            "reference_tilt_idx": int(dset.reference_tilt_idx),
            # phase 2: tilt-axis angles (deg); phase-1 zips lack these keys and load as zeros
            "pose_z1_deg": dset.pose_z1_deg.cpu(),
            "pose_z3_deg": dset.pose_z3_deg.cpu(),
            "pose_z1_init_deg": dset.pose_z1_init_deg.cpu(),
            "pose_z3_init_deg": dset.pose_z3_init_deg.cpu(),
            "learn_pose_angles": bool(dset.learn_pose_angles),
            # task 2: per-dataset defocus offset (Å); pre-defocus zips lack these keys
            "defocus_offset_A": dset.defocus_offset_A.cpu(),
            "defocus_offset_init_A": dset.defocus_offset_init_A.cpu(),
            "learn_defocus": bool(dset.learn_defocus),
            # 2026-11: loaded / learned scan positions, the tilt offset and the intensity scale;
            # older zips lack these keys and keep today's behaviour
            "scan_positions_load_A": [
                None if p is None else p.cpu() for p in dset._loaded_scan_positions_A()
            ],
            "scan_positions_px": dset.scan_positions_px.detach().cpu().clone(),
            "learn_scan_positions": bool(dset.learn_scan_positions),
            "scan_position_constraints": dict(dset.scan_position_constraints),
            "pose_dtheta_deg": dset.pose_dtheta_deg.cpu(),
            "pose_dtheta_init_deg": dset.pose_dtheta_init_deg.cpu(),
            "learn_pose_tilt": bool(dset.learn_pose_tilt),
            "intensity_scale": dset.intensity_scale.cpu(),
        }

    def _apply_pose_metadata(self, meta: "dict[str, Any]") -> None:
        """Re-apply a saved pose state onto the (rebuilt) attached wrapper.

        A tilt-count mismatch is an error, not a warning: a resumed run silently starting from
        zero shifts is exactly the failure this state exists to prevent.
        """
        dset = cast(PtychoTomoDatasetRaster, self.dset)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        shifts = torch.as_tensor(meta["pose_shifts"])
        if tuple(shifts.shape) != (dset.num_tilts, 2):
            raise ValueError(
                f"saved pose shifts have shape {tuple(shifts.shape)} but the attached wrapper "
                f"has {dset.num_tilts} tilts"
            )
        dset.reference_tilt_idx = int(meta["reference_tilt_idx"])
        dset.set_pose_shift_init(meta["pose_shifts_init"])
        with torch.no_grad():
            dset._pose_shifts.copy_(shifts.to(dset._pose_shifts.device, dset._pose_shifts.dtype))
        if "pose_z1_deg" in meta:
            z1 = torch.as_tensor(meta["pose_z1_deg"])
            z3 = torch.as_tensor(meta["pose_z3_deg"])
            if tuple(z1.shape) != (dset.num_tilts,) or tuple(z3.shape) != (dset.num_tilts,):
                raise ValueError(
                    f"saved pose angles have shapes {tuple(z1.shape)} / {tuple(z3.shape)} but "
                    f"the attached wrapper has {dset.num_tilts} tilts"
                )
            dset.set_pose_angle_init(meta["pose_z1_init_deg"], meta["pose_z3_init_deg"])
            with torch.no_grad():
                dset._pose_z1.copy_(z1.to(dset._pose_z1.device, dset._pose_z1.dtype))
                dset._pose_z3.copy_(z3.to(dset._pose_z3.device, dset._pose_z3.dtype))
        if "defocus_offset_A" in meta:
            df = torch.as_tensor(meta["defocus_offset_A"])
            if tuple(df.shape) != (dset.num_tilts,):
                raise ValueError(
                    f"saved defocus offsets have shape {tuple(df.shape)} but the attached "
                    f"wrapper has {dset.num_tilts} tilts"
                )
            dset.set_defocus_init(meta["defocus_offset_init_A"])
            with torch.no_grad():
                p = dset._defocus_offset_A
                p.copy_(df.to(p.device, p.dtype))
        if "pose_dtheta_deg" in meta:
            dth = torch.as_tensor(meta["pose_dtheta_deg"])
            if tuple(dth.shape) != (dset.num_tilts,):
                raise ValueError(
                    f"saved tilt offsets have shape {tuple(dth.shape)} but the attached wrapper "
                    f"has {dset.num_tilts} tilts"
                )
            dset.set_pose_tilt_init(meta["pose_dtheta_init_deg"])
            with torch.no_grad():
                p = dset._pose_dtheta
                p.copy_(dth.to(p.device, p.dtype))
        if "intensity_scale" in meta:
            dset.set_intensity_scale(torch.as_tensor(meta["intensity_scale"]).numpy())
        if "scan_positions_px" in meta:
            pos = torch.as_tensor(meta["scan_positions_px"])
            if tuple(pos.shape) != tuple(dset.scan_positions_px.shape):
                raise ValueError(
                    f"saved scan positions have shape {tuple(pos.shape)} but the attached "
                    f"wrapper has {tuple(dset.scan_positions_px.shape)}"
                )
            loads = meta.get("scan_positions_load_A")
            if loads is not None and any(v is not None for v in loads):
                dset.set_scan_positions_init(list(loads))  # rebuilds the initial positions
            with torch.no_grad():
                dset._scan_positions_px.copy_(
                    pos.to(dset._scan_positions_px.device, dset._scan_positions_px.dtype)
                )
            cons = dict(meta.get("scan_position_constraints") or {})
            dset.set_learn_scan_positions(
                bool(meta.get("learn_scan_positions", False)),
                remove_mean_displacement=bool(cons.get("remove_mean_displacement", True)),
                max_displacement_A=cons.get("max_displacement_A"),
            )
        dset.set_learn_pose_shifts(bool(meta["learn_pose_shifts"]))
        dset.set_learn_pose_angles(bool(meta.get("learn_pose_angles", False)))
        dset.set_learn_pose_tilt(bool(meta.get("learn_pose_tilt", False)))
        # last: the flag call refreshes the "gather in forward" switch from the live table
        dset.set_learn_defocus(bool(meta.get("learn_defocus", False)))

    @property
    def pose_history(self) -> "list[dict[str, Any]]":
        """Per-iteration ``{iteration, loss, shifts_A (num_tilts, 2) Å, lr, pose_steps, grad_rms,
        grad_max}`` records (rank 0), recorded whenever a pose slot was being learned; when the
        angles are learned each record also carries ``z1_deg, z3_deg (num_tilts,), lr_angles,
        angle_grad_rms, angle_grad_max``, when the defocus is learned ``defocus_A
        (num_tilts,), lr_defocus, defocus_grad_rms, defocus_grad_max``, when the tilt offset is
        learned ``dtheta_deg (num_tilts,), lr_tilt`` and when the scan positions are learned
        ``scan_disp_rms_A, scan_disp_max_A`` (displacement from the initial positions)."""
        return list(self._pose_history or [])

    def _record_iter(self, iter_loss: float, autograd: bool) -> None:
        super()._record_iter(iter_loss, autograd)
        dset = cast(PtychoTomoDatasetRaster, self.dset)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        learn_angles = bool(getattr(dset, "learn_pose_angles", False))
        learn_defocus = bool(getattr(dset, "learn_defocus", False))
        learn_tilt = bool(getattr(dset, "learn_pose_tilt", False))
        learn_positions = bool(getattr(dset, "learn_scan_positions", False))
        if (
            not getattr(dset, "learn_pose_shifts", False)
            and not learn_angles
            and not learn_defocus
            and not learn_tilt
            and not learn_positions
        ):
            return
        if self._pose_history is None:
            self._pose_history = []
        record: dict[str, Any] = {
            "iteration": int(self.num_iters),
            "loss": float(iter_loss),
            "shifts_A": dset.pose_shifts_A.cpu(),
            "lr": float(dset.get_current_lr()),
            "pose_steps": int(dset.pose_step_count),
            "grad_rms": float(dset.pose_last_grad_stats[0]),
            "grad_max": float(dset.pose_last_grad_stats[1]),
        }
        if learn_angles:
            record["z1_deg"] = dset.pose_z1_deg.cpu()
            record["z3_deg"] = dset.pose_z3_deg.cpu()
            record["lr_angles"] = float(dset.pose_group_lr("pose_angles"))
            record["angle_grad_rms"] = float(dset.pose_last_angle_grad_stats[0])
            record["angle_grad_max"] = float(dset.pose_last_angle_grad_stats[1])
        if learn_defocus:
            record["defocus_A"] = dset.defocus_offset_A.cpu()
            record["lr_defocus"] = float(dset.pose_group_lr("pose_defocus"))
            record["defocus_grad_rms"] = float(dset.pose_last_defocus_grad_stats[0])
            record["defocus_grad_max"] = float(dset.pose_last_defocus_grad_stats[1])
        if learn_tilt:
            record["dtheta_deg"] = dset.pose_dtheta_deg.cpu()
            record["lr_tilt"] = float(dset.pose_group_lr("pose_tilt"))
        if learn_positions:
            disp = torch.cat(dset.scan_displacements_A).norm(dim=-1)
            record["scan_disp_rms_A"] = float(disp.square().mean().sqrt())
            record["scan_disp_max_A"] = float(disp.max())
        self._pose_history.append(record)

    def _reset_iter_constraints(self) -> None:
        """Epoch-boundary hook: take any pending accumulated pose step before the next epoch
        starts (runs on every rank, so the ranks stay in step)."""
        dset = cast(PtychoTomoDatasetRaster, self.dset)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        flush = getattr(dset, "flush_pose_accum", None)
        if flush is not None:
            flush()
        super()._reset_iter_constraints()

    def reset_recon(self) -> None:
        super().reset_recon()
        self._pose_history = []

    # endregion --- pose refinement state ---

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
        warp_box_frac: "tuple[float | None, float | None, float | None] | None" = None,
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
        - ``warp_box_frac``: optional ``(z, y, x)`` per-axis coordinate warp, off by default.
          Each entry is the share of the object model's ``[-1, 1]`` backend range that the
          specimen box should occupy on that axis (``None`` leaves an axis alone). With a
          margin the backend otherwise spends its features evenly over box + 2*margin; the warp
          moves that spending back into the box. It changes only the coordinates handed to the
          backend — the box, the materialized volume and the crops keep their meaning.

        Tilt-series specifics as before: a common (usually zero) CoM rotation is forced across
        tilts, and the 2D probe-overlap FOV mask is replaced by a trivial mask.

        Stack backend (``ObjectStackTomo``): none of the box / margin / warp geometry applies.
        The lateral grid is the single-scan one -- the reference dataset's object grid padded by
        ``obj_padding_px`` (the native argument here, not deprecated) -- the thickness is the
        stack's, and the probe-overlap FOV mask is built as in single-scan ptychography
        (``batch_size`` bounds its batches).
        """
        del plot_rotation, plot_com, plot_probe_overlap  # unused (parity only)
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

        if self._is_stack:
            if specimen_box_A is not None or warp_box_frac is not None or z_padding_px:
                raise ValueError(
                    "the stack backend has no specimen box / warp / z padding: its lateral grid "
                    "is the reference dataset's object grid padded by obj_padding_px, its "
                    "thickness is num_slices x slice thickness"
                )
            if any(float(m) != 0.0 for m in box_margin_A):
                raise ValueError("box_margin_A does not apply to the stack backend")
            _p = obj_padding_px if obj_padding_px is not None else (0, 0)
            return self._preprocess_stack(
                pad_yx=(int(_p[0]), int(_p[1])),
                val_ratio=val_ratio,
                val_mode=val_mode,
                vectorized=vectorized,
                batch_size=batch_size,
                com_fit_function=com_fit_function,
                force_com_rotation=float(force_com_rotation),
                force_com_transpose=bool(force_com_transpose),
                probe_energy=probe_energy,
                free_per_tilt_arrays=free_per_tilt_arrays,
            )

        # ---- deprecated pixel-padding arguments (legacy mapping keeps old runs reproducible)
        legacy = obj_padding_px is not None or z_padding_px is not None
        if legacy:
            from warnings import warn

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

        obj.set_geometry(
            lateral_box_A=lateral_box,
            sampling=samp,
            box_margin_A=margins,
            warp_box_frac=warp_box_frac,
        )
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

    def _preprocess_stack(
        self,
        *,
        pad_yx: tuple[int, int],
        val_ratio: float,
        val_mode: Literal["grid", "random"],
        vectorized: bool,
        batch_size: int | None,
        com_fit_function: Literal["none", "plane", "parabola", "constant", "no_shift"],
        force_com_rotation: float,
        force_com_transpose: bool,
        probe_energy: float | None,
        free_per_tilt_arrays: bool,
    ) -> Self:
        """The single-scan preprocessing path for the stack backend (see ``preprocess``)."""
        dset_t = cast(PtychoTomoDatasetRaster, self.dset)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        if dset_t.slab_window:
            raise ValueError("slab-window mode does not apply to the stack backend")
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
        self.obj_padding_px = pad_yx  # allocates the stack and re-derives the positions
        dset_t._set_scan_center(self.obj_padding_px)
        self.compute_propagator_arrays()
        self._set_obj_fov_mask(batch_size=batch_size)
        self._preprocessed = True
        self.val_ratio = float(val_ratio)
        self.val_mode = val_mode
        return self

    def _get_probe_overlap(self, max_batch_size: int | None = None) -> np.ndarray:
        """Summed probe intensity on the object grid. Stack backend: the single-scan sum with
        the patch indices computed per batch (the wrapper stores none)."""
        if not self._is_stack:
            return super()._get_probe_overlap(max_batch_size)
        dset_t = cast(PtychoTomoDatasetRaster, self.dset)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        prb = self.probe_model.probe[0]
        num_dps = self.dset.num_positions
        shape2d = (int(self.obj_shape_full[-2]), int(self.obj_shape_full[-1]))
        batch = min(num_dps, 4096) if max_batch_size is None else int(max_batch_size)
        overlap = torch.zeros(shape2d, dtype=self._dtype_real, device=self._single_device)
        for start, end in generate_batches(num_dps, max_batch=batch):
            idx = dset_t.whole_pixel_patch_indices(torch.arange(start, end), shape2d)
            intensity = (torch.abs(prb) ** 2).expand(end - start, *prb.shape)
            overlap += sum_patches(intensity, idx.to(overlap.device), shape2d)
        return self._to_numpy(overlap)

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
        if self._is_stack:
            return  # the stack is laterally periodic and spans its own thickness
        from warnings import warn

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
        """Base forward operator plus the probe z-offset pre-propagation (2026-07-16 / 2026-09),
        the per-dataset intensity scale and the stack backend's tilted propagator (2026-11).

        The dataset stashes ``_last_probe_dz_A`` (Å along the beam, per batch element) during
        ``dset.forward``: the slab-window offset minus the per-dataset learned defocus offset.
        The physical probe is fixed in the lab, so the probe entering a window displaced by
        ``dz`` is the probe Fresnel-propagated by ``dz`` — the same ``exp(-iπ λ dz k²)`` factor
        as the inter-slice propagators — and propagating by ``dz`` lowers the probe's defocus
        by ``dz``, so a defocus offset of ``+δ`` is propagation by ``-δ`` (sign pinned by
        ``TestPoseDefocus``). The slab-window part also moves the object query through the
        payload's ``window_dz_A``; the defocus part moves only the probe. The stash is consumed
        exactly once per batch step (train and validation both rebuild it via ``dset.forward``);
        it is ``None``, and nothing here runs, when neither mode is active.

        ``_last_amp_scale`` (per pattern ``sqrt(intensity scale)``, ``set_intensity_scale``)
        multiplies the probe. ``_last_tilt_tan`` (stack backend, per pattern ``(tan_r,
        tan_c)``) turns every inter-slice propagator into the tilted one
        ``exp(-iπλ dz k²) · exp(-2πi dz (tan_r k_r + tan_c k_c))``, which shifts the wave by
        ``+dz · tan`` per slice, and the incident probe is first shifted by
        ``-(Z/2) · tan`` (``Z`` = sum of the slice spacings, i.e. first-to-last slice) so the
        shear pivots about the middle of the stack: changing the tilt does not translate the
        image. Same stash rule: None (the plain path, bit-identical) while every tilt is zero.
        """
        dset = cast(PtychoTomoDatasetRaster, self.dset)  # pyright: ignore[reportInvalidCast] -- sibling-class payload seam
        dz = getattr(dset, "_last_probe_dz_A", None)
        tilt = getattr(dset, "_last_tilt_tan", None)
        amp = getattr(dset, "_last_amp_scale", None)
        if dz is not None:
            dset._last_probe_dz_A = None  # consume-once: no staleness, clean serialization
        if tilt is not None:
            dset._last_tilt_tan = None
        if amp is not None:
            dset._last_amp_scale = None
            shifted_input_probes = shifted_input_probes * amp.to(
                device=shifted_input_probes.device, dtype=self._dtype_real
            ).view(1, -1, 1, 1)
        pivot_shift = None
        if tilt is not None and self.num_slices > 1:
            z_half = 0.5 * float(np.sum(self.slice_thicknesses))
            pivot_shift = -z_half * tilt
        if dz is not None or pivot_shift is not None:
            shifted_input_probes = self._pre_propagate_probes(
                shifted_input_probes, dz, shift_A=pivot_shift
            )
        if tilt is None or self.num_slices == 1:
            return super().forward_operator(obj_patches, shifted_input_probes, descan)
        # tilted copy of the base operator (base: probe-tilt refresh, overlap, descan ramp)
        if self.probe_model.learn_probe_tilt:
            self.compute_propagator_arrays()
        propagated_probes, overlap = self._overlap_projection_tilted(
            obj_patches, shifted_input_probes, tilt
        )
        if descan is not None:
            shifts = fourier_translation_operator(descan, tuple(self.roi_shape))
            overlap = overlap * shifts[None]
        return propagated_probes, overlap

    def _lateral_k(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Row / column spatial frequencies (1/Å) of the probe ROI, the propagators' grid."""
        h, w = int(self.roi_shape[0]), int(self.roi_shape[1])
        samp = np.asarray(self.sampling, dtype=float).ravel()[-2:]
        key = (h, w, float(samp[0]), float(samp[1]))
        cache = self._tilt_k_cache
        if cache is None or cache[0] != key or cache[1].device != device:
            kr = torch.fft.fftfreq(h, d=float(samp[0]), device=device).to(self._dtype_real)
            kc = torch.fft.fftfreq(w, d=float(samp[1]), device=device).to(self._dtype_real)
            cache = (key, kr, kc)
            self._tilt_k_cache = cache
        return cache[1], cache[2]

    def _overlap_projection_tilted(
        self, obj_patches: torch.Tensor, input_probe: torch.Tensor, tilt_tan: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``overlap_projection`` with a per-pattern tilted propagator: waves are ``(P, B, H,
        W)``, so the ramp ``(1, B, H, W)`` broadcasts over mixed-dataset batches."""
        kr, kc = self._lateral_k(input_probe.device)
        t = tilt_tan.to(device=input_probe.device, dtype=self._dtype_real)
        unit = t[:, 0, None, None] * kr[None, :, None] + t[:, 1, None, None] * kc[None, None, :]
        dzs = [float(v) for v in np.asarray(self.slice_thicknesses, dtype=float).ravel()]
        propagated_probes = [input_probe]
        overlap = obj_patches[0] * input_probe
        for s in range(1, self.num_slices):
            ramp = torch.exp((-2.0j * torch.pi * dzs[s - 1]) * unit)[None]  # (1, B, H, W)
            propagated_probe = self._propagate_array(overlap, self._propagators[s - 1] * ramp)
            overlap = obj_patches[s] * propagated_probe
            propagated_probes.append(propagated_probe)
        return torch.stack(propagated_probes, dim=0), overlap

    def _pre_propagate_probes(
        self,
        probes: torch.Tensor,
        dz_A: torch.Tensor | None,
        shift_A: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fresnel-propagate probes ``(P, B, H, W)`` by per-batch ``dz_A`` (B,) in Å and/or
        shift them laterally by ``shift_A`` (B, 2) Å (row, col), in one Fourier multiply."""
        from quantem.core.utils.utils import electron_wavelength_angstrom

        phase: torch.Tensor | None = None
        if dz_A is not None:
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
        if shift_A is not None:
            kr, kc = self._lateral_k(probes.device)
            d = shift_A.to(device=probes.device, dtype=self._dtype_real)
            arg = d[:, 0, None, None] * kr[None, :, None] + d[:, 1, None, None] * kc[None, None, :]
            ramp = torch.exp(-2.0j * torch.pi * arg)[None]  # shifts by +d
            phase = ramp if phase is None else phase * ramp
        if phase is None:
            return probes
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
        y == x exactly (enforced), z equal to <0.1% (grid rounding over the box thickness).
        Stack backend: the lateral ``(y, x)`` pixel only, as in single-scan ptychography."""
        obj = self.obj_model
        assert isinstance(obj, ObjectPtychoTomoBase)
        lat = np.asarray(self.dset.obj_sampling, dtype=float)
        if self._is_stack:
            return lat
        try:
            zv = obj.z_voxel_A
        except ValueError:  # volume grid not allocated yet (mid-construction, pre-preprocess)
            zv = float(lat[0])  # z voxel == lateral pixel by design once geometry is set
        return np.array([zv, lat[0], lat[1]], dtype=float)

    @property
    def obj_shape_crop(self) -> np.ndarray:
        """``(D, h, w)`` of the specimen box (all support margins removed on all three axes).
        Stack backend: ``(num_slices, h, w)`` of the scan field of view (single-scan crop)."""
        obj = self.obj_model
        assert isinstance(obj, ObjectPtychoTomoBase)
        if self._is_stack:
            return PtychographyBase.obj_shape_crop.fget(self)  # pyright: ignore[reportOptionalCall]
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
        pass ``skip=("_snapshots",)`` to drop them for a leaner save. The learned pose state
        (shift, angle and defocus-offset tables with their baselines and learn flags, and the
        reference tilt) is stashed alongside the base dataset metadata and re-applied by
        ``from_file(dset=...)``.
        """
        if not save_raw_data and self._dset is not None:
            self._pose_metadata = self._collect_pose_metadata()
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
            if not save_raw_data:
                self._pose_metadata = None
