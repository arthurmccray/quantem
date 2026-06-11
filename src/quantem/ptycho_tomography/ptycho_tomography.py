"""Joint ptychography-tomography reconstruction class.

``PtychoTomography`` is a thin subclass of :class:`Ptychography`: the entire reconstruction loop
(batching, losses, optimizers/schedulers, snapshots, multi-GPU spawn, torchrun) is inherited
unchanged. The tilt series enters through the paired models — a
:class:`~quantem.ptycho_tomography.dataset_models.PtychoTomoDatasetRaster` (flat ``tilt x scan``
index space, rotation-carrying object-query payload) and a rotation-aware
:class:`~quantem.ptycho_tomography.object_models.ObjectPtychoTomoBase` object (one 3D specimen
volume queried at rotated coordinates per tilt). This class only adapts construction/validation,
preprocessing geometry, and visualization to the 3D object.

Typical use::

    dset = PtychoTomoDatasetRaster.from_dataset4dstem_list(dsets_4d, tilt_angles_deg)
    obj = ObjectVoxelTomo.from_uniform(thickness_A=56.0, num_slices=16)
    probe = ProbeParametric.from_params(probe_params={"energy": 300e3, ...})
    pt = PtychoTomography.from_models(dset, obj, probe, DetectorPixelated(), device="gpu")
    pt.preprocess(obj_padding_px=(32, 32))
    pt.reconstruct(num_iters=300, optimizer_params={"object": ..., "probe": ...})
    vol = pt.volume_cropped  # (D, h, w) specimen-frame density, rad/Å
"""

from pathlib import Path
from typing import Literal, Self, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np

from quantem.core import config
from quantem.core.visualization import show_2d
from quantem.diffractive_imaging.detector_models import DetectorModelType
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.probe_models import ProbeModelType
from quantem.diffractive_imaging.ptychography import Ptychography
from quantem.ptycho_tomography.dataset_models import PtychoTomoDatasetRaster
from quantem.ptycho_tomography.object_models import ObjectPtychoTomoBase


class PtychoTomography(Ptychography):
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
        probe_energy: float | None = None,
        free_per_tilt_arrays: bool = True,
    ) -> Self:
        """Preprocess the tilt series and run the 3D object-geometry handshake.

        Mirrors ``PtychographyBase.preprocess`` with tilt-series specifics: per-tilt CoM rotation
        is forced (default 0; per-tilt solved rotations would scramble the shared geometry), the
        2D probe-overlap FOV mask is replaced by a trivial mask (it is meaningless for a rotated
        3D object — support/positivity constraints play its role), and the object model's volume
        grid is matched to the padded object via ``obj_padding_px``'s ``_initialize_obj`` call.
        The plotting/batch_size/DP-padding arguments exist for base-signature parity:
        plots and the probe-overlap mask are skipped, and DP padding is not supported.
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

        # geometry handshake: triggers obj_model._initialize_obj(obj_shape_full, sampling)
        # (volume grid allocation) and re-derives scan positions on the padded grid
        self.obj_padding_px = obj_padding_px
        self.compute_propagator_arrays()

        # trivial FOV mask (ndim-3 expanded by the setter); obj_model.mask stays empty so the
        # INR-style constraints skip it
        full2d = self.dset._obj_shape_full_2d(self.obj_padding_px)
        self.obj_fov_mask = np.ones(tuple(int(s) for s in full2d), dtype=config.get("dtype_real"))

        self._check_slab_coverage()
        self._preprocessed = True
        self.val_ratio = float(val_ratio)
        self.val_mode = val_mode
        return self

    def _check_slab_coverage(self) -> None:
        """Warn if the beam-frame multislice slab cannot cover the rotated object support.

        The slab spans ``thickness_A`` along the beam; an isolated object whose support fits in a
        ball of diameter <= thickness_A is covered at every tilt. We can't know the true support,
        so warn only on the clear inconsistency ``slab extent < thickness_A`` (always wrong) and
        leave wide/planar-sample coverage (lateral extent >> thickness) to the user.
        """
        obj = self.obj_model
        if not isinstance(obj, ObjectPtychoTomoBase):  # pragma: no cover - guarded by from_models
            return
        slab = obj.slab_thickness_A * obj.num_slices
        if slab < obj.thickness_A * (1 - 1e-6):
            from warnings import warn

            warn(
                f"multislice slab extent ({slab:.2f} Å) is smaller than the object box thickness "
                f"({obj.thickness_A:.2f} Å); the rotated object will be cropped along the beam.",
                stacklevel=2,
            )

    # endregion --- preprocessing ---

    # region --- properties ---
    @property
    def obj_shape_crop(self) -> np.ndarray:
        """``(D, h, w)``: the specimen volume cropped laterally to the scan FOV (full z)."""
        shp = np.floor(self.dset.fov / self.sampling)
        shp += shp % 2
        obj = self.obj_model
        assert isinstance(obj, ObjectPtychoTomoBase)
        return np.concatenate([[obj.volume_shape[0]], shp]).astype("int")

    @property
    def volume(self) -> np.ndarray:
        """Materialized specimen-frame volume ``(D, H, W)`` (density, rad/Å) on the padded grid."""
        return self.obj

    @property
    def volume_cropped(self) -> np.ndarray:
        """Specimen-frame volume cropped laterally to the scan FOV, ``(D, h, w)``."""
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
        return obj.thickness_A / obj.volume_shape[0]

    # endregion --- properties ---

    def save(
        self,
        path: "str | Path",
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: "str | type | Sequence[str | type]" = (),
        compression_level: int | None = 4,
        save_raw_data: bool = True,
        verbose: int | bool = True,
    ):
        """Save the reconstruction. ``save_raw_data`` defaults to True (unlike single-scan
        ptychography): the concatenated tilt stack has no single source file path, so the
        load-time dataset-rebuild path is not available yet."""
        if not save_raw_data:
            from warnings import warn

            warn(
                "save_raw_data=False: reloading will not be able to rebuild the tilt-series "
                "dataset automatically (no single source file); pass dset= to from_file.",
                stacklevel=2,
            )
        return super().save(
            path,
            mode=mode,
            store=store,
            skip=skip,
            compression_level=compression_level,
            save_raw_data=save_raw_data,
            verbose=verbose,
        )

    # region --- visualization ---
    def _volume_sections(self, slab_frac: float = 0.1) -> list[np.ndarray]:
        """Slab-averaged central cross-sections [(y,x), (z,x), (z,y)] of the cropped volume."""
        vol = self.volume_cropped
        secs = []
        for ax in range(3):
            n = vol.shape[ax]
            half = max(1, int(round(n * slab_frac / 2)))
            lo, hi = max(0, n // 2 - half), min(n, n // 2 + half)
            secs.append(vol.take(range(lo, hi), axis=ax).mean(axis=ax))
        return secs

    def show_volume_sections(
        self,
        slab_frac: float = 0.1,
        cmap: str = "magma",
        axsize: tuple[float, float] = (4.0, 4.0),
        returnfig: bool = False,
    ):
        """Central slab-averaged cross-sections of the reconstructed volume.

        The (z, x) and (z, y) panels are drawn with the true physical aspect
        (``z_sampling / lateral sampling``) so depth sections read to scale.
        """
        secs = self._volume_sections(slab_frac)
        titles = ["volume (y, x)", "volume (z, x)", "volume (z, y)"]
        fig, axs = plt.subplots(1, 3, figsize=(3 * axsize[0], axsize[1]))
        z_aspect = self.z_sampling / float(np.mean(self.sampling))
        for i, (ax, sec, title) in enumerate(zip(axs, secs, titles)):
            im = ax.imshow(sec, cmap=cmap, aspect=(z_aspect if i > 0 else 1.0))
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        if returnfig:
            return fig, axs
        plt.show()

    def visualize(self, cbar: bool = True, return_fig: bool = False, *, cmap: str = "magma"):
        """Loss curve + central volume cross-sections + probe intensity."""
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 2])
        ax_loss = fig.add_subplot(gs[0, :])
        losses = np.asarray(self._iter_losses)
        if losses.size:
            ax_loss.semilogy(np.arange(len(losses)), losses, c="k")
        ax_loss.set_xlabel("iteration")
        ax_loss.set_ylabel("loss")

        secs = self._volume_sections()
        titles = ["volume (y, x)", "volume (z, x)", "volume (z, y)"]
        z_aspect = self.z_sampling / float(np.mean(self.sampling))
        axs = []
        for i, (sec, title) in enumerate(zip(secs, titles)):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(sec, cmap=cmap, aspect=(z_aspect if i > 0 else 1.0))
            ax.set_title(title)
            if cbar:
                plt.colorbar(im, ax=ax, fraction=0.046)
            axs.append(ax)

        ax_p = fig.add_subplot(gs[1, 3])
        probe = self.probe
        probe0 = probe[0] if probe.ndim == 3 else probe
        show_2d(np.abs(probe0) ** 2, figax=(fig, ax_p), title="probe intensity", cbar=cbar)
        axs.append(ax_p)
        plt.tight_layout()
        if return_fig:
            return fig, (ax_loss, np.asarray(axs))
        plt.show()

    # endregion --- visualization ---
