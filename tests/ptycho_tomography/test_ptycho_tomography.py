"""End-to-end tests for PtychoTomography.

The forward-consistency tests generate tilt-series data through the class's OWN forward model
(deliberate inverse crime — they validate the optimization machinery and wiring, not the physics;
the geometry/physics validation lives in test_geometry.py / test_object_voxel.py and the
real-data M0 notebook): a frozen ground-truth voxel object + known probe produce the measured
intensities, then (a) the loss evaluated at the ground truth is near zero, and (b) a fresh vacuum
object recovers the volume by gradient descent.
"""

import numpy as np
import pytest
import torch

from quantem.core import config
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffractive_imaging.detector_models import DetectorPixelated
from quantem.diffractive_imaging.probe_models import ProbePixelated
from quantem.ptycho_tomography.dataset_models import PtychoTomoDatasetRaster
from quantem.ptycho_tomography.object_models import ObjectVoxelTomo
from quantem.ptycho_tomography.ptycho_tomography import PtychoTomography

if config.NUM_DEVICES > 0:
    config.set_device("gpu")

N = 32  # detector / roi size (px)
Q_MAX = 0.5  # inverse Angstroms; recip sampling 2*Q_MAX/N = 1/32 -> obj sampling 1.0 A/px
Q_PROBE = Q_MAX / 2
PROBE_ENERGY = 300e3
C10 = 30.0  # defocus (A)
SCAN_STEP_A = 2.0
SCAN_GPTS = (6, 6)
PAD = 10  # obj padding px (pre power-of-2 adjustment)
TILTS = [-60.0, -30.0, 0.0, 30.0, 60.0]
THICKNESS_A = 12.0
NUM_SLICES = 6
# z storage matched to the multislice sampling: with few tilts a finer z grid has a large
# depth nullspace (density can redistribute along z without changing the data)
NUM_Z_VOX = 6


def _probe_array() -> np.ndarray:
    sampling = 1 / Q_MAX / 2
    reciprocal_sampling = 2 * Q_MAX / N
    qx = qy = np.fft.fftfreq(N, sampling)
    q = np.sqrt(qx[:, None] ** 2 + qy[None, :] ** 2)
    aperture = np.sqrt(np.clip((Q_PROBE - q) / reciprocal_sampling + 0.5, 0, 1))
    chi = q**2 * electron_wavelength_angstrom(PROBE_ENERGY) * np.pi * C10
    probe_fourier = aperture * np.exp(-1j * chi)
    probe_fourier /= np.sqrt(np.sum(np.abs(probe_fourier) ** 2))
    return (np.fft.ifft2(probe_fourier) * N).astype(np.complex64)


def _semiangle_mrad() -> float:
    return electron_wavelength_angstrom(PROBE_ENERGY) * Q_PROBE * 1e3


def _make_wrapper(arrays_per_tilt: list[np.ndarray]) -> PtychoTomoDatasetRaster:
    recip = 2 * Q_MAX / N
    dsets = [
        Dataset4dstem.from_array(
            array=arr.astype(np.float32),
            sampling=(SCAN_STEP_A, SCAN_STEP_A, recip, recip),
            units=("A", "A", "A^-1", "A^-1"),
        )
        for arr in arrays_per_tilt
    ]
    return PtychoTomoDatasetRaster.from_dataset4dstem_list(dsets, TILTS, verbose=0)


def _make_ptycho(wrapper: PtychoTomoDatasetRaster) -> PtychoTomography:
    obj = ObjectVoxelTomo.from_uniform(
        thickness_A=THICKNESS_A, num_slices=NUM_SLICES, num_z_voxels=NUM_Z_VOX, rng=0
    )
    probe_model = ProbePixelated.from_array(
        num_probes=1,
        probe_params={"energy": PROBE_ENERGY, "C10": C10, "semiangle_cutoff": _semiangle_mrad()},
        probe_array=_probe_array(),
    )
    pt = PtychoTomography.from_models(
        dset=wrapper,
        obj_model=obj,
        probe_model=probe_model,
        detector_model=DetectorPixelated(),
        rng=0,
        verbose=False,
    )
    pt.preprocess(obj_padding_px=(PAD, PAD))
    return pt


def _gt_volume(obj: ObjectVoxelTomo) -> torch.Tensor:
    """Asymmetric 3-gaussian density (rad/Å) on the object's volume grid."""
    d, hh, ww = obj.volume_shape
    h_z, h_y, h_x = obj._box_half_extents

    def ax(n, h):
        return (torch.zeros(1) if n == 1 else torch.linspace(-1, 1, n)) * h

    zz, yy, xx = torch.meshgrid(ax(d, h_z), ax(hh, h_y), ax(ww, h_x), indexing="ij")
    pts = torch.stack([zz, yy, xx], dim=-1)
    centers = torch.tensor([[-2.5, 1.5, 3.0], [2.0, -2.5, -1.0], [3.5, 3.0, -3.0]])
    amps = torch.tensor([0.4, 0.3, 0.5])
    d2 = ((pts[..., None, :] - centers) ** 2).sum(-1)
    return (amps * torch.exp(-d2 / (2 * 1.8**2))).sum(-1)


def _forward_all(pt: PtychoTomography, batch: int = 64) -> torch.Tensor:
    """Predicted detector intensities for every (tilt, scan) index, ``(num_gpts, N, N)``."""
    preds = []
    with torch.no_grad():
        for start in range(0, pt.dset.num_gpts, batch):
            idx = torch.arange(start, min(start + batch, pt.dset.num_gpts))
            patch_data, _pos, frac, descan = pt.dset.forward(idx, pt.obj_padding_px)
            probes = pt.probe_model.forward(frac)
            patches = pt.obj_model.forward(patch_data)
            _, overlap = pt.forward_operator(patches, probes, descan)
            preds.append(pt.detector_model.forward(overlap).cpu())
    return torch.cat(preds)


def _eval_data_loss(pt: PtychoTomography) -> float:
    pt.dset._set_targets(pt._criterion.target_space)
    total = 0.0
    n = pt.dset.num_gpts
    with torch.no_grad():
        for start in range(0, n, 64):
            idx = torch.arange(start, min(start + 64, n))
            targets = pt.dset.targets[idx].to(pt._single_device)
            patch_data, _pos, frac, descan = pt.dset.forward(idx, pt.obj_padding_px)
            probes = pt.probe_model.forward(frac)
            patches = pt.obj_model.forward(patch_data)
            _, overlap = pt.forward_operator(patches, probes, descan)
            pred = pt.detector_model.forward(overlap)
            loss, _ = pt.error_estimate(pred, targets=targets, global_n=n)
            total += loss.item()
    return total


@pytest.fixture(scope="module")
def inverse_crime_setup():
    """GT-generated tilt-series data + the GT volume on the reconstruction grid."""
    rng = np.random.default_rng(0)
    placeholder = [
        rng.uniform(0.5, 1.0, size=(*SCAN_GPTS, N, N)).astype(np.float32) for _ in TILTS
    ]
    pt0 = _make_ptycho(_make_wrapper(placeholder))
    obj0 = pt0.obj_model
    assert isinstance(obj0, ObjectVoxelTomo)
    gt = _gt_volume(obj0)
    obj0.set_volume(gt)
    preds = _forward_all(pt0).numpy()
    arrays = [
        preds[i * np.prod(SCAN_GPTS) : (i + 1) * np.prod(SCAN_GPTS)].reshape((*SCAN_GPTS, N, N))
        for i in range(len(TILTS))
    ]
    return arrays, gt


class TestWiring:
    def test_preprocess_wiring(self, inverse_crime_setup):
        arrays, _ = inverse_crime_setup
        pt = _make_ptycho(_make_wrapper(arrays))
        obj = pt.obj_model
        assert isinstance(obj, ObjectVoxelTomo)
        assert pt.dset.implicit_object is True
        full2d = pt.dset._obj_shape_full_2d(pt.obj_padding_px)
        assert obj.volume_shape == (NUM_Z_VOX, int(full2d[0]), int(full2d[1]))
        assert (full2d % 8 == 0).all()  # power-of-2 padding adjustment applied
        assert pt.propagators.shape == (NUM_SLICES - 1, N, N)
        assert np.allclose(pt.obj_fov_mask, 1.0)
        h_z, h_y, h_x = obj._box_half_extents
        assert h_z == pytest.approx(THICKNESS_A / 2)
        assert h_y == pytest.approx((full2d[0] - 1) / 2 * pt.sampling[0])

    def test_volume_and_crop_shapes(self, inverse_crime_setup):
        arrays, _ = inverse_crime_setup
        pt = _make_ptycho(_make_wrapper(arrays))
        obj = pt.obj_model
        assert isinstance(obj, ObjectVoxelTomo)
        assert pt.volume.shape == obj.volume_shape
        crop = pt.volume_cropped
        assert crop.shape == tuple(pt.obj_shape_crop)
        assert crop.shape[0] == NUM_Z_VOX

    def test_from_models_type_validation(self, inverse_crime_setup):
        arrays, _ = inverse_crime_setup
        wrapper = _make_wrapper(arrays)
        probe_model = ProbePixelated.from_array(
            num_probes=1,
            probe_params={
                "energy": PROBE_ENERGY,
                "C10": C10,
                "semiangle_cutoff": _semiangle_mrad(),
            },
            probe_array=_probe_array(),
        )
        from quantem.diffractive_imaging.object_models import ObjectPixelated

        bad_obj = ObjectPixelated.from_uniform(num_slices=1, obj_type="potential")
        with pytest.raises(TypeError, match="rotation-aware"):
            PtychoTomography.from_models(wrapper, bad_obj, probe_model, DetectorPixelated())


class TestForwardConsistency:
    def test_loss_at_gt_much_smaller_than_vacuum(self, inverse_crime_setup):
        arrays, gt = inverse_crime_setup
        pt = _make_ptycho(_make_wrapper(arrays))
        obj = pt.obj_model
        assert isinstance(obj, ObjectVoxelTomo)
        obj.set_volume(gt, set_as_initial=False)
        loss_gt = _eval_data_loss(pt)
        obj.reset()  # back to vacuum
        loss_vac = _eval_data_loss(pt)
        assert loss_gt < 0.01 * loss_vac, f"loss at GT {loss_gt:.3e} vs vacuum {loss_vac:.3e}"

    def test_reconstruct_smoke_loss_decreases(self, inverse_crime_setup):
        arrays, _ = inverse_crime_setup
        pt = _make_ptycho(_make_wrapper(arrays))
        pt.reconstruct(
            num_iters=10,
            optimizer_params={"object": {"name": "adam", "lr": 2e-2}},
            batch_size=64,
        )
        losses = np.asarray(pt._iter_losses)
        assert np.isfinite(losses).all()
        assert losses[-1] < losses[0]

    @pytest.mark.slow
    def test_volume_recovery(self, inverse_crime_setup):
        arrays, gt = inverse_crime_setup
        pt = _make_ptycho(_make_wrapper(arrays))
        pt.reconstruct(
            num_iters=200,
            optimizer_params={"object": {"name": "adam", "lr": 3e-2}},
            constraints={
                "object": {
                    "positivity_weight": 0.3,
                    "tv_weight_z": 3e-3,
                    "tv_weight_xy": 1e-3,
                }
            },
            batch_size=pt.dset.num_gpts,
        )
        losses = np.asarray(pt._iter_losses)
        assert losses[-1] < 0.1 * losses[0]
        vol = torch.as_tensor(pt.volume)
        gt_t = gt.to(vol.dtype)
        proj_corr = torch.corrcoef(torch.stack([vol.sum(0).flatten(), gt_t.sum(0).flatten()]))[
            0, 1
        ].item()
        vol_corr = torch.corrcoef(torch.stack([vol.flatten(), gt_t.flatten()]))[0, 1].item()
        # validated: loss ratio ~0.004, proj corr ~0.88, vol corr ~0.55 (depth is the
        # under-determined direction with 5 tilts; projection is well-determined)
        assert proj_corr > 0.75, f"projected correlation too low: {proj_corr:.3f}"
        assert vol_corr > 0.4, f"volume correlation too low: {vol_corr:.3f}"


class TestSerialization:
    def test_save_from_file_roundtrip_and_visualize(self, inverse_crime_setup, tmp_path):
        """The notebook flow: save (no raw data) -> rebuild wrapper -> from_file(dset=...) ->
        visualize. Catches the implicit_object resync and the loaded-object render path."""
        import matplotlib

        matplotlib.use("Agg")
        arrays, _ = inverse_crime_setup
        pt = _make_ptycho(_make_wrapper(arrays))
        pt.reconstruct(
            num_iters=3,
            optimizer_params={"object": {"name": "adam", "lr": 1e-2}},
            batch_size=64,
        )
        vol_before = pt.volume.copy()
        path = tmp_path / "ptycho_tomo.zip"
        pt.save(path, mode="o")  # raw tilt data excluded by default
        wrapper2 = _make_wrapper(arrays)  # rebuilt + preprocessed fresh, like the notebook
        wrapper2.preprocess(obj_padding_px=(PAD, PAD))
        loaded = PtychoTomography.from_file(path, dset=wrapper2)
        assert isinstance(loaded, PtychoTomography)
        assert loaded.dset.implicit_object is True  # re-synced for the fresh wrapper
        assert loaded.dset.num_tilts == len(TILTS)
        np.testing.assert_allclose(loaded.volume, vol_before, rtol=1e-5, atol=1e-6)
        # the loaded object must visualize (this exact path failed before the from_file fix)
        fig, _axs = loaded.visualize(return_fig=True)
        assert fig is not None
        # timings survive the round trip
        assert len(loaded.recon_timings) == 1
        assert loaded.recon_timings[0]["iters"] == 3
        assert loaded.recon_timings[0]["s_per_iter"] > 0
        # continued reconstruction runs after reload
        loaded.reconstruct(
            num_iters=2,
            optimizer_params={"object": {"name": "adam", "lr": 1e-2}},
            batch_size=64,
        )
        assert len(loaded.recon_timings) == 2


class TestSnapshotsAndPadding:
    def test_snapshots_are_banded_and_cropped(self, inverse_crime_setup):
        arrays, _ = inverse_crime_setup
        pt = _make_ptycho(_make_wrapper(arrays))
        pt.reconstruct(
            num_iters=4,
            optimizer_params={"object": {"name": "adam", "lr": 1e-2}},
            batch_size=64,
            store_snapshots_every=2,
        )
        assert len(pt.snapshots) >= 2
        snp = pt.snapshots[-1]
        crop = tuple(pt.obj_shape_crop)
        assert snp["obj"].shape == (min(NUM_SLICES, crop[0]), crop[1], crop[2])

    def test_z_padding_preprocess_and_crop(self, inverse_crime_setup):
        arrays, _ = inverse_crime_setup
        wrapper = _make_wrapper(arrays)
        obj = ObjectVoxelTomo.from_uniform(
            thickness_A=THICKNESS_A, num_slices=NUM_SLICES, num_z_voxels=None, rng=0
        )
        probe_model = ProbePixelated.from_array(
            num_probes=1,
            probe_params={
                "energy": PROBE_ENERGY,
                "C10": C10,
                "semiangle_cutoff": _semiangle_mrad(),
            },
            probe_array=_probe_array(),
        )
        pt = PtychoTomography.from_models(
            wrapper, obj, probe_model, DetectorPixelated(), rng=0, verbose=False
        )
        z_pad_px = 4
        pt.preprocess(obj_padding_px=(PAD, PAD), z_padding_px=z_pad_px)
        pad_A = z_pad_px * float(np.mean(pt.sampling))
        assert obj.box_thickness_A == pytest.approx(THICKNESS_A + 2 * pad_A)
        # cubic z count covers the padded box; the crop removes the padding again
        assert obj.volume_shape[0] == round(obj.box_thickness_A / float(np.mean(pt.sampling)))
        crop = pt.volume_cropped
        assert crop.shape[0] == round(THICKNESS_A / obj.z_voxel_A)
        assert crop.shape[0] < obj.volume_shape[0]
        # propagator spacing follows the padded slab thickness
        assert obj.slab_thickness_A == pytest.approx(obj.box_thickness_A / NUM_SLICES)
        # reconstruction runs with padding active
        pt.reconstruct(
            num_iters=2,
            optimizer_params={"object": {"name": "adam", "lr": 1e-2}},
            batch_size=64,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
