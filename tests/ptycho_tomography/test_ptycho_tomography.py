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


def _make_probe() -> ProbePixelated:
    return ProbePixelated.from_array(
        num_probes=1,
        probe_params={"energy": PROBE_ENERGY, "C10": C10, "semiangle_cutoff": _semiangle_mrad()},
        probe_array=_probe_array(),
    )


def _make_ptycho(wrapper: PtychoTomoDatasetRaster) -> PtychoTomography:
    obj = ObjectVoxelTomo.from_uniform(
        thickness_A=THICKNESS_A, num_slices=NUM_SLICES, num_z_voxels=NUM_Z_VOX, rng=0
    )
    pt = PtychoTomography.from_models(
        dset=wrapper,
        obj_model=obj,
        probe_model=_make_probe(),
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
        # pt.sampling is now a (z, y, x) 3-vector; index 1 is the lateral (row) sampling
        assert h_y == pytest.approx((full2d[0] - 1) / 2 * pt.sampling[1])

    def test_volume_and_crop_shapes(self, inverse_crime_setup):
        arrays, _ = inverse_crime_setup
        pt = _make_ptycho(_make_wrapper(arrays))
        obj = pt.obj_model
        assert isinstance(obj, ObjectVoxelTomo)
        assert pt.volume.shape == obj.volume_shape
        crop = pt.volume_cropped
        assert crop.shape == tuple(pt.obj_shape_crop)
        assert crop.shape[0] == NUM_Z_VOX

    def test_volume_cropped_cubic_is_centered_cube(self, inverse_crime_setup):
        arrays, _ = inverse_crime_setup
        pt = _make_ptycho(_make_wrapper(arrays))
        vc = pt.volume_cropped
        cube = pt.volume_cropped_cubic
        n = min(vc.shape)
        assert cube.shape == (n, n, n)  # consistent cubic framing (WS4: z vacuum vs scan FOV)
        sl = tuple(slice((s - n) // 2, (s - n) // 2 + n) for s in vc.shape)
        np.testing.assert_array_equal(cube, vc[sl])

    def test_cubic_specimen_box_volume_cropped(self, inverse_crime_setup):
        """specimen_box_A=(T, L, L) with L == thickness crops to the specimen box with three
        ~equal (cubic) voxel sizes.

        Point-grid convention on all three axes (``round(L / s) + 1`` points spanning the
        box), so the cubic specimen box crops to a cubic (13, 13, 13) array.
        """
        arrays, _ = inverse_crime_setup
        wrapper = _make_wrapper(arrays)
        obj = ObjectVoxelTomo.from_uniform(
            thickness_A=THICKNESS_A, num_slices=NUM_SLICES, num_z_voxels=None, rng=0
        )
        pt = PtychoTomography.from_models(
            wrapper, obj, _make_probe(), DetectorPixelated(), rng=0, verbose=False
        )
        L = THICKNESS_A  # cubic specimen box: bz must equal the object's thickness_A
        pt.preprocess(specimen_box_A=(THICKNESS_A, L, L))
        samp = pt.sampling
        assert samp.shape == (3,)  # (z, y, x) voxel sizes, cubic by construction
        assert np.allclose(samp, samp[0], rtol=1e-3)
        crop = pt.obj_cropped
        n_box = round(L / float(samp[1]))
        assert crop.shape == (n_box + 1, n_box + 1, n_box + 1)  # cubic box -> cubic crop
        assert crop.shape[1] == crop.shape[2]
        assert pt.volume_cropped.shape == crop.shape

    def test_deprecated_padding_matches_explicit_set_geometry(self, inverse_crime_setup):
        """Old/new geometry equivalence: the deprecated preprocess(obj_padding_px=(PAD, PAD))
        maps to set_geometry with lateral_box_A = un-padded grid point-extent and
        box_margin_A = the (power-of-2-adjusted) padding in Å — identical box half-extents
        and a numerically identical object forward on the same payload."""
        arrays, _ = inverse_crime_setup
        pt = _make_ptycho(_make_wrapper(arrays))  # deprecated obj_padding_px=(PAD, PAD) inside
        obj_legacy = pt.obj_model
        assert isinstance(obj_legacy, ObjectVoxelTomo)
        s = float(pt.sampling[1])
        n_fov = pt.dset._obj_shape_full_2d((0, 0))  # un-padded grid points per axis
        pad_final = np.asarray(pt.obj_padding_px)  # PAD after the power-of-2 adjustment
        obj_new = ObjectVoxelTomo.from_uniform(
            thickness_A=THICKNESS_A, num_slices=NUM_SLICES, num_z_voxels=NUM_Z_VOX, rng=0
        )
        obj_new.set_geometry(
            lateral_box_A=((int(n_fov[0]) - 1) * s, (int(n_fov[1]) - 1) * s),
            sampling=(s, s),
            box_margin_A=(0.0, float(pad_final[0]) * s, float(pad_final[1]) * s),
        )
        assert obj_new._box_half_extents == obj_legacy._box_half_extents
        assert obj_new.volume_shape == obj_legacy.volume_shape
        # same payload + same volume -> identical transmission patches
        torch.manual_seed(0)
        vol = 0.01 * torch.rand(obj_legacy.volume_shape)
        obj_legacy.set_volume(vol, set_as_initial=False)
        obj_new.set_volume(vol, set_as_initial=False)
        patch_data, *_ = pt.dset.forward(torch.arange(8), pt.obj_padding_px)
        with torch.no_grad():
            out_legacy = obj_legacy.forward(patch_data)
            out_new = obj_new.forward(patch_data)  # pyright: ignore[reportArgumentType] -- payload seam: the tomo dset.forward emits PtychoTomoPatchData
        assert torch.allclose(out_legacy, out_new, atol=1e-6)

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
            PtychoTomography.from_models(wrapper, bad_obj, probe_model, DetectorPixelated())  # pyright: ignore[reportArgumentType] -- intentional wrong object type for the TypeError test


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
                    "tv_weight": 3e-3,
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
            store_snapshots_every=2,  # present at save -> exercises the snapshot-skip path
        )
        assert len(pt.snapshots) >= 1
        vol_before = pt.volume.copy()
        path = tmp_path / "ptycho_tomo.zip"
        pt.save(path, mode="o")  # raw tilt data excluded by default
        wrapper2 = _make_wrapper(arrays)  # rebuilt + preprocessed fresh, like the notebook
        wrapper2.preprocess(obj_padding_px=(PAD, PAD))
        loaded = PtychoTomography.from_file(path, dset=wrapper2)
        assert isinstance(loaded, PtychoTomography)
        assert loaded.dset.implicit_object is True  # re-synced for the fresh wrapper
        assert loaded.dset.num_tilts == len(TILTS)
        # snapshots (object state_dicts) survive the round trip and re-materialize after load
        assert len(loaded.snapshots) == len(pt.snapshots)
        assert "state_dict" in loaded.snapshots[-1]
        snp = loaded.get_snapshot_by_iter(loaded.snapshots[-1]["iteration"], cropped=True)
        assert snp["obj"].shape == tuple(loaded.obj_shape_crop)
        np.testing.assert_allclose(loaded.volume, vol_before, rtol=1e-5, atol=1e-6)
        # the loaded object must visualize (this exact path failed before the from_file fix)
        fig, _axs = loaded.visualize(return_fig=True)  # pyright: ignore[reportGeneralTypeIssues] -- return_fig=True always returns the (fig, axes) tuple
        assert fig is not None
        # continued reconstruction runs after reload
        loaded.reconstruct(
            num_iters=2,
            optimizer_params={"object": {"name": "adam", "lr": 1e-2}},
            batch_size=64,
        )
        assert loaded.num_iters == 5


class TestSnapshotsAndPadding:
    def test_snapshots_checkpoint_and_materialize(self, inverse_crime_setup):
        arrays, _ = inverse_crime_setup
        pt = _make_ptycho(_make_wrapper(arrays))
        pt.reconstruct(
            num_iters=4,
            optimizer_params={"object": {"name": "adam", "lr": 1e-2}},
            batch_size=64,
            store_snapshots_every=2,
        )
        assert len(pt.snapshots) >= 2
        # snapshots hold a lightweight object state_dict, not a materialized volume
        assert "state_dict" in pt.snapshots[-1]
        # materializing a snapshot returns the cropped specimen volume and restores live state
        vol_now = pt.volume.copy()
        last_iter = pt.snapshots[-1]["iteration"]
        snp = pt.get_snapshot_by_iter(last_iter, cropped=True)
        assert snp["obj"].shape == tuple(pt.obj_shape_crop)
        np.testing.assert_allclose(pt.volume, vol_now)
        # closest=True resolves to a stored iteration
        snp_closest = pt.get_snapshot_by_iter(last_iter - 1, closest=True)
        assert snp_closest["iteration"] in [s["iteration"] for s in pt.snapshots]

    def test_z_padding_preprocess_and_crop(self, inverse_crime_setup):
        arrays, _ = inverse_crime_setup
        wrapper = _make_wrapper(arrays)
        obj = ObjectVoxelTomo.from_uniform(
            thickness_A=THICKNESS_A, num_slices=NUM_SLICES, num_z_voxels=None, rng=0
        )
        pt = PtychoTomography.from_models(
            wrapper, obj, _make_probe(), DetectorPixelated(), rng=0, verbose=False
        )
        z_pad_px = 4
        pt.preprocess(obj_padding_px=(PAD, PAD), z_padding_px=z_pad_px)
        # pt.sampling is (z, y, x); the deprecated z_padding_px maps to Å via the lateral pixel
        s_lat = float(np.mean(pt.sampling[1:]))
        pad_A = z_pad_px * s_lat
        assert obj.box_thickness_A == pytest.approx(THICKNESS_A + 2 * pad_A)
        # cubic z point count covers the padded box; the crop removes the padding again
        assert obj.volume_shape[0] == round(obj.box_thickness_A / s_lat) + 1
        crop = pt.volume_cropped
        assert crop.shape[0] == round(THICKNESS_A / obj.z_voxel_A) + 1  # point convention
        assert crop.shape[0] < obj.volume_shape[0]
        # propagator spacing follows the padded slab thickness
        assert obj.slab_thickness_A == pytest.approx(obj.box_thickness_A / NUM_SLICES)
        # reconstruction runs with padding active
        pt.reconstruct(
            num_iters=2,
            optimizer_params={"object": {"name": "adam", "lr": 1e-2}},
            batch_size=64,
        )


class TestSlabWindow:
    """Per-scan-position slab-window mode (2026-07-16, plan-view campaign).

    The slab stack is decoupled from the box thickness (``set_slab_extent_A``) and displaced
    per scan position onto the tilted specimen slab (``set_slab_window``); the probe is
    Fresnel pre-propagated by the same offset. Classic mode with a stack long enough to cover
    everything is the physics reference — windowed forwards must reproduce it.
    """

    WIN_EXTENT_A = 36.0  # covers box/cos(30°) + 2·tan(30°)·r_patch ≈ 32.4 Å at ±30°
    WIN_SLICES = 18  # dz = 2 Å, same as the classic 12 Å / 6-slice default

    def _make_pt(self, arrays, window: bool) -> PtychoTomography:
        wrapper = _make_wrapper(arrays)
        obj = ObjectVoxelTomo.from_uniform(
            thickness_A=THICKNESS_A, num_slices=self.WIN_SLICES, num_z_voxels=NUM_Z_VOX, rng=0
        )
        obj.set_slab_extent_A(self.WIN_EXTENT_A)
        pt = PtychoTomography.from_models(
            dset=wrapper,
            obj_model=obj,
            probe_model=_make_probe(),
            detector_model=DetectorPixelated(),
            rng=0,
            verbose=False,
        )
        pt.preprocess(obj_padding_px=(PAD, PAD))
        if window:
            wrapper.set_slab_window(True)
        return pt

    def test_window_dz_matches_pure_tilt_formula(self, inverse_crime_setup):
        arrays, _gt = inverse_crime_setup
        pt = self._make_pt(arrays, window=True)
        dset = pt.dset
        assert isinstance(dset, PtychoTomoDatasetRaster)
        idx = torch.arange(dset.num_gpts)
        payload, _pos, _frac, _descan = dset.forward(idx, pt.obj_padding_px)
        dz = payload.window_dz_A
        assert dz is not None and dz.shape == (dset.num_gpts,)
        # expected: window center on the specimen mid-plane => dz = -tan(tilt) * y_scan_A
        center = dset._scan_center_px
        samp = dset.obj_sampling
        y_c = (dset.scan_positions_px[idx, 0] - center[0]) * float(samp[0])
        tilts = dset._tilt_angles_deg[dset.tilt_index_of(idx)]
        expected = -torch.tan(torch.deg2rad(tilts)) * y_c
        assert torch.allclose(dz, expected.to(dz.dtype), atol=1e-5)
        # the transient probe stash was set by forward and matches the payload
        assert dset._last_window_dz_A is dz

    def test_pre_propagate_probes_roundtrip_and_slab_consistency(self, inverse_crime_setup):
        arrays, _gt = inverse_crime_setup
        pt = self._make_pt(arrays, window=False)
        probes = pt.probe_model.forward(torch.zeros(3, 2))  # (P, 3, H, W)
        dz = torch.tensor([-4.0, 0.0, 4.0])
        out = pt._pre_propagate_probes(probes, dz)
        # dz = 0 is exactly the identity
        assert torch.allclose(out[:, 1], probes[:, 1], atol=1e-6)
        # propagating by +dz then -dz is the identity (unitary Fresnel factor)
        back = pt._pre_propagate_probes(out, -dz)
        assert torch.allclose(back, probes, atol=1e-5)
        # propagating by one slab thickness == the multislice inter-slice propagator
        obj = pt.obj_model
        assert isinstance(obj, ObjectVoxelTomo)
        t_slab = obj.slab_thickness_A
        stepped = pt._pre_propagate_probes(probes, torch.full((3,), t_slab))
        via_propagator = pt._propagate_array(probes, pt._propagators[0])
        assert torch.allclose(stepped, via_propagator, atol=1e-5)

    def test_windowed_forward_matches_classic_reference(self, inverse_crime_setup):
        """4a ≡ 4b: same object/probe/extent, window offsets on vs off, tilts 0 and ±30°.

        The 36 Å stack covers all material both ways at these tilts, so the windowed forward
        must reproduce the classic one up to quadrature/interpolation differences from the
        (continuously) shifted sample z's. At 0° the offsets vary only with scan row
        (tan(0) = 0 -> dz = 0 exactly), so the match is exact there.
        """
        arrays, gt = inverse_crime_setup
        pt_a = self._make_pt(arrays, window=False)
        pt_b = self._make_pt(arrays, window=True)
        for pt in (pt_a, pt_b):
            obj = pt.obj_model
            assert isinstance(obj, ObjectVoxelTomo)
            obj.set_volume(gt)
        preds_a = _forward_all(pt_a)
        preds_b = _forward_all(pt_b)
        n_scan = int(np.prod(SCAN_GPTS))
        for t_i, tilt in enumerate(TILTS):
            if abs(tilt) > 31.0:
                continue  # 36 Å window does not cover the patch corners beyond ±30°
            a = preds_a[t_i * n_scan : (t_i + 1) * n_scan]
            b = preds_b[t_i * n_scan : (t_i + 1) * n_scan]
            rel = float((a - b).norm() / a.norm())
            if tilt == 0.0:
                assert rel < 1e-5, f"tilt 0 must match exactly, rel={rel:.2e}"
            else:
                assert rel < 0.05, f"tilt {tilt}: windowed forward diverges, rel={rel:.2e}"

    def test_slab_extent_updates_slice_thicknesses_and_coverage_warns(self, inverse_crime_setup):
        arrays, _gt = inverse_crime_setup
        pt = self._make_pt(arrays, window=True)
        obj = pt.obj_model
        assert isinstance(obj, ObjectVoxelTomo)
        assert obj.slab_thickness_A == pytest.approx(self.WIN_EXTENT_A / self.WIN_SLICES)
        # shrink the stack below box/cos(60°) + patch spread -> the coverage check must warn
        obj.set_slab_extent_A(10.0)
        pt.compute_propagator_arrays()
        with pytest.warns(UserWarning, match="slab-window"):
            pt._check_slab_coverage()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
