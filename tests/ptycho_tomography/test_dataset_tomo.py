"""Tests for the multi-tilt ptycho-tomography dataset wrapper.

Uses small random Dataset4dstem stacks (the wrapper's mechanics don't need physical patterns);
the physically meaningful end-to-end checks live in test_ptycho_tomography.py.
"""

import numpy as np
import pytest
import torch

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import load as autoserialize_load
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.ptycho_tomography.dataset_models import PtychoTomoDatasetRaster
from quantem.ptycho_tomography.geometry import rot_beam_to_spec

ROI = 16
GPTS = (4, 5)
TILTS = [-35.0, 0.0, 35.0]
STEP = 1.5
Q_SAMP = 0.05


def _make_dset4d(seed: int, scale: float = 1.0) -> Dataset4dstem:
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0.5, 1.0, size=(*GPTS, ROI, ROI)).astype(np.float32) * scale
    return Dataset4dstem.from_array(
        array=arr,
        sampling=(STEP, STEP, Q_SAMP, Q_SAMP),
        units=("A", "A", "A^-1", "A^-1"),
    )


def _build_wrapper(preprocess: bool = True, free: bool = True) -> PtychoTomoDatasetRaster:
    dsets = [_make_dset4d(seed=i, scale=1.0 + 0.5 * i) for i in range(len(TILTS))]
    wrapper = PtychoTomoDatasetRaster.from_dataset4dstem_list(dsets, TILTS, verbose=0)
    if preprocess:
        wrapper.preprocess(obj_padding_px=(8, 8), free_per_tilt_arrays=free)
    return wrapper


class TestConstruction:
    def test_flat_index_space(self):
        w = _build_wrapper(preprocess=False)
        n_per = int(np.prod(GPTS))
        assert w.num_gpts == len(TILTS) * n_per
        assert w.num_tilts == len(TILTS)
        expected_offsets = torch.tensor([0, n_per, 2 * n_per, 3 * n_per])
        assert torch.equal(w.tilt_offsets.cpu(), expected_offsets)
        assert np.array_equal(w.gpts, GPTS)
        assert w.scan_positions_px.shape == (w.num_gpts, 2)
        assert not w.scan_positions_px.requires_grad  # frozen in v1
        assert not w.descan_shifts.requires_grad

    def test_angle_count_mismatch_raises(self):
        dsets = [_make_dset4d(i) for i in range(3)]
        with pytest.raises(ValueError, match="tilt angles"):
            PtychoTomoDatasetRaster.from_dataset4dstem_list(dsets, [0.0, 35.0])

    def test_geometry_mismatch_raises(self):
        d0 = _make_dset4d(0)
        rng = np.random.default_rng(1)
        d1 = Dataset4dstem.from_array(
            array=rng.uniform(size=(3, 5, ROI, ROI)).astype(np.float32),
            sampling=(STEP, STEP, Q_SAMP, Q_SAMP),
            units=("A", "A", "A^-1", "A^-1"),
        )
        with pytest.raises(ValueError, match="geometry"):
            PtychoTomoDatasetRaster.from_dataset4dstem_list([d0, d1], [0.0, 35.0])

    def test_tilt_index_of_boundaries(self):
        w = _build_wrapper(preprocess=False)
        n = int(np.prod(GPTS))
        idx = torch.tensor([0, n - 1, n, 2 * n - 1, 2 * n, 3 * n - 1])
        expected = torch.tensor([0, 0, 1, 1, 2, 2])
        assert torch.equal(w.tilt_index_of(idx), expected)

    def test_rotations_match_geometry(self):
        w = _build_wrapper(preprocess=False)
        R = w.rotations()
        assert R.shape == (3, 3, 3)
        expected = rot_beam_to_spec(0.0, torch.tensor(TILTS), 0.0)
        assert torch.allclose(R, expected, atol=1e-6)

    def test_patch_index_buffer_dropped(self):
        w = _build_wrapper(preprocess=False)
        assert w._patch_indices.numel() == 0
        w._set_patch_indices((8, 8))  # no-op
        assert w._patch_indices.numel() == 0
        assert w.patch_indices_need_update() is False


class TestPreprocess:
    def test_aggregated_shapes_and_targets(self):
        w = _build_wrapper()
        n = w.num_gpts
        assert w.amplitudes.shape == (n, ROI, ROI)
        assert w.centered_amplitudes.shape == (n, ROI, ROI)
        assert w.targets.shape == (n, ROI, ROI)
        assert w.preprocessed
        item = w[5]
        assert set(item) == {"index", "target"}
        assert item["target"].shape == (ROI, ROI)
        assert len(w) == n

    def test_global_mean_intensity_is_weighted_mean(self):
        w = _build_wrapper(free=False)
        per_tilt = [ds.mean_diffraction_intensity for ds in w.tilt_datasets]
        counts = [ds.num_gpts for ds in w.tilt_datasets]
        expected = float(np.average(per_tilt, weights=counts))
        assert w.mean_diffraction_intensity == pytest.approx(expected, rel=1e-6)
        # tilts were scaled differently, so this is a real average, not a constant
        assert max(per_tilt) > 1.2 * min(per_tilt)

    def test_free_per_tilt_arrays(self):
        w = _build_wrapper(free=True)
        for ds in w.tilt_datasets:
            assert ds._intensities_4d.size == 0
            assert ds._amplitudes.numel() == 0
            assert ds._patch_indices.numel() == 0
        # wrapper storage unaffected
        assert w.amplitudes.numel() == w.num_gpts * ROI * ROI
        # geometry metadata retained
        assert np.array_equal(w.gpts, GPTS)

    def test_keep_per_tilt_arrays(self):
        w = _build_wrapper(free=False)
        for ds in w.tilt_datasets:
            assert ds._amplitudes.numel() > 0

    def test_scan_positions_concatenated_per_tilt(self):
        w = _build_wrapper()
        n = int(np.prod(GPTS))
        pos = w.scan_positions_px.detach()
        # identical scan geometry -> per-tilt position blocks are identical
        assert torch.allclose(pos[:n], pos[n : 2 * n])
        assert torch.allclose(pos[:n], pos[2 * n :])


class TestForward:
    def test_requires_implicit_object(self):
        w = _build_wrapper()
        with pytest.raises(RuntimeError, match="implicit"):
            w.forward(torch.tensor([0, 1]), (8, 8))

    def test_payload_contract(self):
        w = _build_wrapper()
        w.implicit_object = True
        n = int(np.prod(GPTS))
        batch = torch.tensor([0, n - 1, n + 2, 2 * n + 3])  # spans all three tilts
        payload, pos, frac, descan = w.forward(batch, (8, 8))
        assert payload.coords_yx_A.shape == (4, ROI, ROI, 2)
        assert payload.rotations.shape == (4, 3, 3)
        assert torch.equal(payload.tilt_indices, torch.tensor([0, 0, 1, 2]))
        assert pos.shape == (4, 2)
        assert torch.equal(frac, torch.zeros_like(frac))
        assert descan is None
        expected_R = rot_beam_to_spec(
            0.0, torch.tensor([TILTS[0], TILTS[0], TILTS[1], TILTS[2]]), 0.0
        )
        assert torch.allclose(payload.rotations, expected_R, atol=1e-6)

    def test_scan_coords_match_single_tilt_dataset(self):
        """Tilt-0 block coordinates equal those of a standalone single-tilt dataset."""
        d0 = _make_dset4d(seed=0, scale=1.0)
        single = PtychographyDatasetRaster.from_dataset4dstem(
            d0, verbose=False, learn_descan=False, learn_scan_positions=False
        )
        single.preprocess(
            com_fit_function="constant",
            force_com_rotation=0.0,
            force_com_transpose=False,
            obj_padding_px=(8, 8),
            plot_rotation=False,
            plot_com=False,
        )
        w = _build_wrapper()
        w.implicit_object = True
        local = torch.tensor([0, 3, 7])
        coords_single = single._scan_coords(local, (8, 8))
        payload, *_ = w.forward(local, (8, 8))  # tilt-0 block: flat == local indices
        # the wrapper now emits physical Å (origin at the padded-grid center) while the base
        # _scan_coords stays normalized over the padded grid; they relate by
        # coords_A = coords_norm * h with h = (full2d - 1) / 2 * sampling per axis
        full2d = single._obj_shape_full_2d((8, 8))
        samp = single.obj_sampling
        h = torch.tensor(
            [
                (int(full2d[0]) - 1) / 2.0 * float(samp[0]),
                (int(full2d[1]) - 1) / 2.0 * float(samp[1]),
            ]
        )
        assert torch.allclose(payload.coords_yx_A, coords_single * h, atol=1e-5)

    def test_payload_coords_are_A_centered_on_scan_grid(self):
        """Payload coords are physical Å with the origin at the scan-grid center.

        For the tilt-0 block of a dataset preprocessed at padding (8, 8): the batch-mean of the
        per-position coordinate (ROI offset 0 == the scan position itself) sits near 0 (the scan
        grid is anchored at the coordinate origin, up to the pixel rounding of the grid
        placement), and each patch spans exactly one ROI extent in Å.
        """
        w = _build_wrapper()
        w.implicit_object = True
        samp = w.obj_sampling
        n = int(np.prod(GPTS))
        payload, *_ = w.forward(torch.arange(n), (8, 8))  # all tilt-0 scan positions
        coords = payload.coords_yx_A
        # ROI offset 0 is the (fractional) scan position itself -> grid centered near 0 Å
        centers = coords[:, 0, 0, :]
        assert centers.mean(0).abs().max().item() < STEP  # within one scan step of the origin
        # per-patch spread along each axis = (ROI - 1) * sampling in Å (fftfreq offsets)
        row_span = (coords[0, ..., 0].max() - coords[0, ..., 0].min()).item()
        col_span = (coords[0, ..., 1].max() - coords[0, ..., 1].min()).item()
        assert row_span == pytest.approx((ROI - 1) * float(samp[0]), rel=1e-6)
        assert col_span == pytest.approx((ROI - 1) * float(samp[1]), rel=1e-6)


class TestStateAndSerialization:
    def test_reset_zeros_pose_and_restores_positions(self):
        w = _build_wrapper()
        with torch.no_grad():
            w._pose_dtheta.add_(2.0)
            w._pose_shifts.add_(1.0)
            w.scan_positions_px.add_(3.0)
        w.reset()
        assert torch.equal(w._pose_dtheta, torch.zeros_like(w._pose_dtheta))
        assert torch.equal(w._pose_shifts, torch.zeros_like(w._pose_shifts))
        assert torch.allclose(w.scan_positions_px, w.initial_scan_positions_px)

    def test_to_cpu_roundtrip(self):
        w = _build_wrapper()
        w.to("cpu")
        assert w.targets.device.type == "cpu"

    def test_get_optimization_parameters_empty_v1(self):
        w = _build_wrapper()
        assert w.get_optimization_parameters() == {}

    def test_autoserialize_roundtrip(self, tmp_path):
        w = _build_wrapper()
        w.implicit_object = True
        path = tmp_path / "tomo_dset.zip"
        w.save(path, mode="o")
        loaded = autoserialize_load(path)
        assert loaded.num_gpts == w.num_gpts
        assert loaded.num_tilts == w.num_tilts
        assert torch.allclose(loaded.tilt_angles_deg, w.tilt_angles_deg)
        assert torch.equal(loaded.tilt_offsets, w.tilt_offsets)
        assert torch.allclose(loaded.scan_positions_px.detach(), w.scan_positions_px.detach())
        assert len(loaded.tilt_datasets) == w.num_tilts
        assert loaded.implicit_object is True
        # forward still works after reload
        payload, *_ = loaded.forward(torch.tensor([0, 25]), (8, 8))
        assert payload.rotations.shape == (2, 3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
