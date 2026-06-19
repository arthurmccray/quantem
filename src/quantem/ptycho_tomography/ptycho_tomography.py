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
    pt.preprocess(obj_padding_px=(32, 32), z_padding_px=48)
    pt.reconstruct(num_iters=300, store_snapshots_every=25,
                   optimizer_params={"object": ..., "probe": ...})
    vol = pt.volume_cropped  # (D, h, w) specimen-frame density, rad/Å (padding cropped)
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Self, Sequence, cast

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
        return pt

    # region --- preprocessing ---
    def preprocess(
        self,
        obj_padding_px: tuple[int, int] = (0, 0),
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
        z_padding_px: int = 0,
        probe_energy: float | None = None,
        free_per_tilt_arrays: bool = True,
    ) -> Self:
        """Preprocess the tilt series and run the 3D object-geometry handshake.

        Mirrors ``PtychographyBase.preprocess`` with tilt-series specifics: per-tilt CoM rotation
        is forced (default 0; per-tilt solved rotations would scramble the shared geometry), the
        2D probe-overlap FOV mask is replaced by a trivial mask (it is meaningless for a rotated
        3D object — support/positivity constraints play its role), and the object model's volume
        grid is matched to the padded object via ``obj_padding_px``'s ``_initialize_obj`` call.

        ``z_padding_px`` adds vacuum headroom along the beam on EACH side of the specimen
        (anisotropic padding: z is set separately from the lateral ``obj_padding_px``). At high
        tilt, probes near the lateral edges rotate into ``|z| > thickness/2``; for planar
        (non-vacuum-padded) samples real density lives there. Costs ~no compute (the slab count
        is fixed), only volume memory. ``volume_cropped`` removes it again.
        TODO(padding-units): pixel-denominated padding doesn't make sense for implicitly defined
        objects — consider physical-unit (Å) padding throughout.
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
        if not self.dset.preprocessed:
            self.vprint("Dataset was not preprocessed, proceeding with defaults.")
            # cast: dset is validated as a PtychoTomoDatasetRaster in from_models (it is a
            # sibling of the single-scan raster class, so isinstance narrowing can't apply)
            dset = cast(PtychoTomoDatasetRaster, self.dset)
            dset.preprocess(
                com_fit_function=com_fit_function,
                force_com_rotation=force_com_rotation,
                force_com_transpose=force_com_transpose,
                obj_padding_px=obj_padding_px,
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

        # z padding must be set BEFORE the geometry handshake (it changes the box/slab extent
        # and hence the cubic z-voxel count and propagator spacings)
        obj = cast(ObjectPtychoTomoBase, self.obj_model)
        obj.set_z_padding_A(float(z_padding_px) * float(np.mean(self.sampling)))

        # geometry handshake: triggers obj_model._initialize_obj(obj_shape_full, sampling)
        # (volume grid allocation) and re-derives scan positions on the padded grid
        self.obj_padding_px = obj_padding_px
        self.compute_propagator_arrays()

        # trivial FOV mask (ndim-3 expanded by the setter); obj_model.mask stays empty so the
        # sampled-coordinate constraints skip it
        full2d = self.dset._obj_shape_full_2d(self.obj_padding_px)
        self.obj_fov_mask = np.ones(tuple(int(s) for s in full2d), dtype=config.get("dtype_real"))

        self._check_slab_coverage()
        self._preprocessed = True
        self.val_ratio = float(val_ratio)
        self.val_mode = val_mode
        return self

    def _check_slab_coverage(self) -> None:
        """Warn if the beam-frame multislice slab cannot cover the rotated object support.

        The slab spans the padded box along the beam; an isolated object whose support fits in a
        ball of diameter <= box_thickness_A is covered at every tilt. We can't know the true
        support, so warn only on the clear inconsistency ``slab extent < box thickness`` (always
        wrong) and leave wide/planar-sample coverage (lateral extent >> thickness) to the user.
        """
        obj = self.obj_model
        if not isinstance(obj, ObjectPtychoTomoBase):  # pragma: no cover - guarded by from_models
            return
        slab = obj.slab_thickness_A * obj.num_slices
        if slab < obj.box_thickness_A * (1 - 1e-6):
            from warnings import warn

            warn(
                f"multislice slab extent ({slab:.2f} Å) is smaller than the object box thickness "
                f"({obj.box_thickness_A:.2f} Å); the rotated object will be cropped along the "
                "beam.",
                stacklevel=2,
            )

    # endregion --- preprocessing ---

    def _soft_constraints(self) -> torch.Tensor:
        """Soft constraints with the object's penalties evaluated at sampled coordinates.

        Replaces the base class's ``isinstance(obj_model, ObjectINR)`` branch: every
        ptycho-tomography object model is coordinate-queried, so its soft constraints never
        require the materialized volume (which would force a full-grid query each batch).
        """
        obj_model = cast(ObjectPtychoTomoBase, self.obj_model)
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
        obj_model = cast(ObjectPtychoTomoBase, self.obj_model)
        state = {k: v.detach().cpu().clone() for k, v in obj_model.model.state_dict().items()}
        # snapshots carry an object state_dict rather than the base (obj, probe) arrays
        self._snapshots.append({"iteration": self.num_iters, "state_dict": state})  # type: ignore[typeddict-item]

    def get_snapshot_by_iter(  # pyright: ignore[reportIncompatibleMethodOverride] -- volume snapshot
        self, iteration: int, closest: bool = False, cropped: bool = True
    ) -> dict:
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
        obj_model = cast(ObjectPtychoTomoBase, self.obj_model)
        model = obj_model.model
        saved = deepcopy(model.state_dict())
        try:
            model.load_state_dict(
                {k: v.to(obj_model.device) for k, v in snp["state_dict"].items()}  # type: ignore[typeddict-item]
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
    def obj_shape_crop(self) -> np.ndarray:
        """``(D, h, w)``: the specimen volume cropped to the scan FOV laterally AND to the
        un-padded specimen thickness along z (the z padding is vacuum headroom, not specimen)."""
        shp = np.floor(self.dset.fov / self.sampling)
        shp += shp % 2
        obj = self.obj_model
        assert isinstance(obj, ObjectPtychoTomoBase)
        d_specimen = max(1, round(obj.thickness_A / obj.z_voxel_A))
        d_specimen = min(d_specimen, obj.volume_shape[0])
        return np.concatenate([[d_specimen], shp]).astype("int")

    @property
    def volume(self) -> np.ndarray:
        """Materialized specimen-frame volume ``(D, H, W)`` (density, rad/Å) on the padded grid
        (includes z padding; use ``volume_cropped`` for the specimen)."""
        return self.obj

    @property
    def volume_cropped(self) -> np.ndarray:
        """Specimen-frame volume cropped to the scan FOV and specimen thickness, ``(D, h, w)``."""
        return self.obj_cropped

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
        """
        return super().save(
            path,
            mode=mode,
            store=store,
            skip=skip,
            compression_level=compression_level,
            save_raw_data=save_raw_data,
            verbose=verbose,
        )
