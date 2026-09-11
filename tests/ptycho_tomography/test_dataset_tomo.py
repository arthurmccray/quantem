"""Tests for the multi-tilt ptycho-tomography dataset wrapper.

Uses small random Dataset4dstem stacks (the wrapper's mechanics don't need physical patterns);
the physically meaningful end-to-end checks live in test_ptycho_tomography.py.
"""

import warnings

import numpy as np
import pytest
import torch

from quantem.core import config
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import load as autoserialize_load
from quantem.core.ml.optimizer_mixin import OptimizerParams
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


def _set_grad(w: PtychoTomoDatasetRaster, g: torch.Tensor) -> None:
    w._pose_shifts.grad = g  # pyright: ignore[reportAttributeAccessIssue] -- test-only fake gradient


def _pose_wrapper() -> PtychoTomoDatasetRaster:
    w = _build_wrapper()
    w.implicit_object = True
    return w


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

    def test_scan_grid_mismatch_is_allowed_detector_mismatch_raises(self):
        # 2026-09-09: per-tilt scan grids are allowed (plan-view step 02); the detector
        # geometry (roi_shape / detector_sampling) must still be shared by every tilt
        d0 = _make_dset4d(0)
        rng = np.random.default_rng(1)
        d1 = Dataset4dstem.from_array(
            array=rng.uniform(size=(3, 5, ROI, ROI)).astype(np.float32),
            sampling=(STEP, STEP, Q_SAMP, Q_SAMP),
            units=("A", "A", "A^-1", "A^-1"),
        )
        w = PtychoTomoDatasetRaster.from_dataset4dstem_list([d0, d1], [0.0, 35.0], verbose=0)
        assert w.num_gpts == 4 * 5 + 3 * 5
        d2 = Dataset4dstem.from_array(
            array=rng.uniform(size=(3, 5, ROI, ROI)).astype(np.float32),
            sampling=(STEP, STEP, 2 * Q_SAMP, 2 * Q_SAMP),
            units=("A", "A", "A^-1", "A^-1"),
        )
        with pytest.raises(ValueError, match="detector geometry"):
            PtychoTomoDatasetRaster.from_dataset4dstem_list([d0, d2], [0.0, 35.0])

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
            assert ds._amplitudes is None
            assert ds._patch_indices.numel() == 0
        # wrapper storage unaffected
        assert w.amplitudes.numel() == w.num_gpts * ROI * ROI
        # geometry metadata retained
        assert np.array_equal(w.gpts, GPTS)

    def test_keep_per_tilt_arrays(self):
        w = _build_wrapper(free=False)
        for ds in w.tilt_datasets:
            # amplitudes are lazy upstream; centered_amplitudes is the resident array
            assert ds._centered_amplitudes is not None
            assert ds._centered_amplitudes.numel() > 0

    def test_scan_positions_concatenated_per_tilt(self):
        w = _build_wrapper()
        n = int(np.prod(GPTS))
        pos = w.scan_positions_px.detach()
        # identical scan geometry -> per-tilt position blocks are identical
        assert torch.allclose(pos[:n], pos[n : 2 * n])
        assert torch.allclose(pos[:n], pos[2 * n :])

    def test_scan_center_is_the_position_centre(self):
        # 2026-09-09: the pivot is the exact midpoint of the reference tilt's positions, not
        # the (object grid - 1)/2 point, which sat up to half an object pixel short of it
        w = _build_wrapper()
        n = int(np.prod(GPTS))
        i0 = n * w.ref_tilt_index
        block = w.scan_positions_px.detach()[i0 : i0 + n]
        expected = (block.min(dim=0).values + block.max(dim=0).values) / 2.0
        assert torch.allclose(w._scan_center_px.cpu(), expected.to(w._scan_center_px.dtype))
        assert w.scan_center_mode == "positions"
        assert w.ref_tilt_index == 1  # tilts (-35, 0, 35): the 0 deg dataset

    def test_scan_center_legacy_grid_mode(self):
        dsets = [_make_dset4d(seed=i) for i in range(len(TILTS))]
        w = PtychoTomoDatasetRaster.from_dataset4dstem_list(dsets, TILTS, verbose=0)
        w.set_scan_center_mode("grid")
        w.preprocess(obj_padding_px=(8, 8))
        full2d = w._obj_shape_full_2d((8, 8))
        expected = torch.tensor([(int(full2d[0]) - 1) / 2.0, (int(full2d[1]) - 1) / 2.0])
        assert torch.allclose(w._scan_center_px.cpu(), expected.to(w._scan_center_px.dtype))
        with pytest.raises(ValueError, match="scan_center_mode"):
            w.set_scan_center_mode("bogus")


class TestUnequalScanGrids:
    """Per-tilt scan grids (2026-09-09): the scan across the layers widens with tilt."""

    GRIDS = [(7, 5), (4, 5), (9, 5)]  # rows = across the layers; tilt 0 (index 1) narrowest

    def _make(self, i: int, gpts: tuple[int, int], step_rows: float = STEP) -> Dataset4dstem:
        rng = np.random.default_rng(i)
        arr = rng.uniform(0.5, 1.0, size=(*gpts, ROI, ROI)).astype(np.float32)
        return Dataset4dstem.from_array(
            array=arr, sampling=(step_rows, STEP, Q_SAMP, Q_SAMP), units=("A", "A", "A^-1", "A^-1")
        )

    def _wrap(self, steps=(STEP, STEP, STEP)) -> PtychoTomoDatasetRaster:
        dsets = [self._make(i, g, st) for i, (g, st) in enumerate(zip(self.GRIDS, steps))]
        w = PtychoTomoDatasetRaster.from_dataset4dstem_list(dsets, TILTS, verbose=0)
        w.preprocess(obj_padding_px=(8, 8))
        return w

    def test_accepts_unequal_grids_and_counts(self):
        w = self._wrap()
        counts = [int(np.prod(g)) for g in self.GRIDS]
        assert w.num_gpts == sum(counts)
        assert [tuple(int(v) for v in g) for g in w.gpts_per_tilt] == self.GRIDS
        assert tuple(int(v) for v in w.gpts) == self.GRIDS[1]  # reference tilt = 0 deg
        assert torch.equal(w._tilt_offsets.cpu(), torch.tensor([0, 35, 55, 100]))

    def test_every_block_is_centred_on_the_pivot(self):
        w = self._wrap()
        pos = w.scan_positions_px.detach().cpu()
        off = w._tilt_offsets.cpu().tolist()
        centres = []
        for t in range(3):
            b = pos[off[t] : off[t + 1]]
            centres.append((b.min(dim=0).values + b.max(dim=0).values) / 2.0)
        for c in centres:
            assert torch.allclose(c, centres[1], atol=1e-5)
        assert torch.allclose(w._scan_center_px.cpu(), centres[1].to(w._scan_center_px.dtype))
        # the wide tilt's rows extend symmetrically past the reference tilt's rows
        wide = pos[off[2] : off[3]]
        ref = pos[off[1] : off[2]]
        extra = (self.GRIDS[2][0] - self.GRIDS[1][0]) / 2 * STEP / float(w.obj_sampling[0])
        assert abs(float(wide[:, 0].min() - ref[:, 0].min()) + extra) < 1e-4
        assert abs(float(wide[:, 0].max() - ref[:, 0].max()) - extra) < 1e-4

    def test_anisotropic_step_on_a_tilt(self):
        # variant (b): 30 positions across at a coarser step on the tilted datasets
        w = self._wrap(steps=(2.0 * STEP, STEP, 2.0 * STEP))
        pos = w.scan_positions_px.detach().cpu()
        off = w._tilt_offsets.cpu().tolist()
        b2 = pos[off[2] : off[3]]
        ext_rows = float(b2[:, 0].max() - b2[:, 0].min()) * float(w.obj_sampling[0])
        assert abs(ext_rows - (self.GRIDS[2][0] - 1) * 2.0 * STEP) < 1e-4
        c2 = (b2.min(dim=0).values + b2.max(dim=0).values) / 2.0
        assert torch.allclose(c2, w._scan_center_px.cpu().to(c2.dtype), atol=1e-5)

    def test_detector_mismatch_still_refused(self):
        dsets = [self._make(i, g) for i, g in enumerate(self.GRIDS)]
        bad = Dataset4dstem.from_array(
            array=dsets[2].array,
            sampling=(STEP, STEP, 2 * Q_SAMP, 2 * Q_SAMP),
            units=("A", "A", "A^-1", "A^-1"),
        )
        with pytest.raises(ValueError, match="detector geometry"):
            PtychoTomoDatasetRaster.from_dataset4dstem_list(
                [dsets[0], dsets[1], bad], TILTS, verbose=0
            )

    def test_beam_frame_coords_of_the_wide_tilt(self):
        w = self._wrap()
        w.implicit_object = True
        off = w._tilt_offsets.cpu().tolist()
        idx = torch.arange(off[2], off[3])
        payload, *_ = w.forward(idx, (8, 8))
        rows_A = payload.coords_yx_A[:, 0, 0, 0].cpu()  # centre pixel row coordinate, A
        half = (self.GRIDS[2][0] - 1) / 2 * STEP
        assert abs(float(rows_A.min()) + half) < 1e-4 and abs(float(rows_A.max()) - half) < 1e-4


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
        # the base _scan_coords is normalized about the padded-grid centre, so the identity
        # below holds for the LEGACY "grid" pivot; the default "positions" pivot (2026-09-09)
        # differs from it by the sub-pixel gap between the grid centre and the scan centre
        dsets = [_make_dset4d(seed=i, scale=1.0 + 0.5 * i) for i in range(len(TILTS))]
        w = PtychoTomoDatasetRaster.from_dataset4dstem_list(dsets, TILTS, verbose=0)
        w.set_scan_center_mode("grid")
        w.preprocess(obj_padding_px=(8, 8))
        w.implicit_object = True
        local = torch.tensor([0, 3, 7])
        coords_single = single._scan_coords(local, (8, 8))
        payload, *_ = w.forward(local, (8, 8))  # tilt-0 block: flat == local indices
        # the wrapper emits physical Å (origin at the padded-grid center in grid mode) while
        # the base _scan_coords stays normalized over the padded grid; they relate by
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
        # default mode: same coordinates shifted by (grid centre - position centre), a constant
        w2 = _build_wrapper()
        w2.implicit_object = True
        payload2, *_ = w2.forward(local, (8, 8))
        gap = (w2._scan_center_px.cpu() - w._scan_center_px.cpu()) * torch.as_tensor(
            samp, dtype=torch.float32
        )
        assert torch.allclose(payload2.coords_yx_A, payload.coords_yx_A - gap, atol=1e-5)
        assert float(gap.abs().max()) > 0  # the two definitions differ on this grid

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

    def test_get_optimization_parameters_empty_unless_learning_shifts(self):
        w = _build_wrapper()
        assert w.get_optimization_parameters() == {}
        w.set_learn_pose_shifts(True)
        assert list(w.get_optimization_parameters()) == ["pose_shifts"]
        w.set_learn_pose_shifts(False)
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
        assert payload.shifts_A is None  # nothing active on a fresh wrapper

    def test_autoserialize_roundtrip_carries_pose_state(self, tmp_path):
        w = _build_wrapper()
        w.implicit_object = True
        init = torch.tensor([[0.5, -0.25], [0.0, 0.0], [1.0, 2.0]])
        w.set_pose_shift_init(init)
        w.set_learn_pose_shifts(True)
        w.set_pose_accum(steps_per_iter=2, batches_per_epoch=10)
        path = tmp_path / "tomo_dset_pose.zip"
        w.save(path, mode="o")
        loaded = autoserialize_load(path)
        assert loaded.learn_pose_shifts is True
        assert loaded.reference_tilt_idx == 1
        assert loaded.pose_accum_steps == 5
        assert torch.allclose(loaded.pose_shifts_A, init)
        assert torch.allclose(loaded.pose_shifts_init_A, init)
        payload, *_ = loaded.forward(torch.tensor([0, 25]), (8, 8))
        assert payload.shifts_A is not None
        assert torch.allclose(payload.shifts_A, init[[0, 1]])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


REF = 1  # TILTS = [-35, 0, 35] -> argmin |tilt|
N_PER = int(np.prod(GPTS))


class TestPoseShiftPlumbing:
    def test_shifts_none_when_inactive(self):
        w = _pose_wrapper()
        assert w.reference_tilt_idx == REF
        assert not w.learn_pose_shifts
        payload, *_ = w.forward(torch.tensor([0, N_PER, 2 * N_PER]), (8, 8))
        assert payload.shifts_A is None

    def test_forward_gathers_per_tilt_shift(self):
        w = _pose_wrapper()
        init = torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, -4.0]])
        w.set_pose_shift_init(init)
        idx = torch.tensor([0, 3, N_PER, 2 * N_PER + 1, 2 * N_PER])
        payload, *_ = w.forward(idx, (8, 8))
        assert payload.shifts_A is not None and payload.shifts_A.shape == (5, 2)
        assert torch.allclose(payload.shifts_A, init[w.tilt_index_of(idx)])
        # a live (frozen) table keeps applying: it is the value, not the flag, that matters
        assert not w.learn_pose_shifts

    def test_init_validates_shape_and_zeroes_reference_row(self):
        w = _pose_wrapper()
        with pytest.raises(ValueError, match="shape"):
            w.set_pose_shift_init(np.zeros((2, 2)))
        with pytest.warns(UserWarning, match="gauge"):
            w.set_pose_shift_init([[1.0, 1.0], [0.7, -0.7], [1.0, 1.0]])
        assert torch.equal(w.pose_shifts_A[REF], torch.zeros(2))
        assert torch.equal(w.pose_shifts_init_A[REF], torch.zeros(2))
        with pytest.raises(ValueError, match="out of range"):
            w.reference_tilt_idx = 3

    def test_window_dz_sees_the_shift(self):
        w = _pose_wrapper()
        w.set_slab_window(True)
        idx = torch.arange(2 * N_PER, 3 * N_PER)  # the +35 deg tilt
        shift = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.3, -0.4]])
        rot = w.rotations()[w.tilt_index_of(idx)]
        dz0 = w._window_dz_A(idx, rot)
        w.set_pose_shift_init(shift)
        payload, *_ = w.forward(idx, (8, 8))
        dz1 = payload.window_dz_A
        assert dz1 is not None
        # dz = -tan(tilt) * (y_c - dy): the dx component is along the tilt axis and inert
        center = w._scan_center_px
        samp = w.obj_sampling
        y_c = (w.scan_positions_px[idx, 0] - center[0]) * float(samp[0])
        t = torch.deg2rad(torch.tensor(35.0))
        expected = -torch.tan(t) * (y_c - 1.3)
        assert torch.allclose(dz1, expected.to(dz1.dtype), atol=1e-5)
        assert not torch.allclose(dz0, dz1)


class TestPoseShiftOptimizer:
    @staticmethod
    def _learning(lr=0.5, opt="adam"):
        w = _pose_wrapper()
        w.set_learn_pose_shifts(True)
        spec = OptimizerParams.Adam(lr=lr) if opt == "adam" else OptimizerParams.SGD(lr=lr)
        w.set_optimizer(spec)
        return w

    def test_groups_follow_flag_and_angles_stay_frozen(self):
        w = _pose_wrapper()
        w.set_learn_pose_shifts(True)
        assert w._pose_shifts.requires_grad
        w.set_optimizer(OptimizerParams.Adam(lr=0.1))
        assert w.has_optimizer()
        assert list(w.get_optimization_parameters()) == ["pose_shifts"]
        for p in (w._pose_z1, w._pose_dtheta, w._pose_z3):
            assert not p.requires_grad
        w.set_learn_pose_shifts(False)
        assert not w._pose_shifts.requires_grad
        assert not w.has_optimizer()

    def test_single_spec_fans_out_and_pplr_passes_through(self):
        w = _pose_wrapper()
        w.set_learn_pose_shifts(True)
        w.optimizer_params = OptimizerParams.Adam(lr=0.3)
        assert list(w.optimizer_params) == ["pose_shifts"]
        w.optimizer_params = {"pose_shifts": {"name": "adam", "lr": 0.7}}
        w.set_optimizer()
        assert w.optimizer is not None
        assert w.optimizer.param_groups[0]["lr"] == pytest.approx(0.7)

    def test_nothing_learnable_warns_and_removes(self):
        w = _pose_wrapper()
        with pytest.warns(UserWarning, match="nothing is learnable"):
            w.set_optimizer(OptimizerParams.Adam(lr=0.1))
        assert not w.has_optimizer()

    def test_reference_tilt_pinned_across_a_step(self):
        w = self._learning(lr=0.5)
        _set_grad(w, torch.ones_like(w._pose_shifts))
        w.step_optimizer()
        shifts = w.pose_shifts_A
        assert torch.equal(shifts[REF], torch.zeros(2))
        others = shifts[[0, 2]]
        # Adam's first step is exactly -lr * sign(g) per component
        assert torch.allclose(others, torch.full_like(others, -0.5), atol=1e-6)
        assert w.pose_step_count == 1

    def test_reference_pinned_to_nonzero_baseline_after_reset(self):
        w = self._learning(lr=0.5)
        init = torch.tensor([[0.2, 0.1], [0.0, 0.0], [-0.3, 0.4]])
        w.set_pose_shift_init(init)
        _set_grad(w, torch.ones_like(w._pose_shifts))
        w.step_optimizer()
        assert not torch.allclose(w.pose_shifts_A, init)
        w.reset()
        assert torch.allclose(w.pose_shifts_A, init)
        assert w.pose_step_count == 0
        assert w.learn_pose_shifts  # flag survives reset

    def test_accumulation_equals_one_step_on_the_mean_gradient(self):
        torch.manual_seed(0)
        grads = [torch.randn(len(TILTS), 2) for _ in range(3)]
        w_acc = self._learning(lr=0.1, opt="sgd")
        w_acc.set_pose_accum(steps_per_iter=2, batches_per_epoch=6)  # M = 3
        assert w_acc.pose_accum_steps == 3
        for g in grads[:2]:
            _set_grad(w_acc, g.clone())
            w_acc.step_optimizer()
            assert torch.equal(w_acc.pose_shifts_A, torch.zeros(len(TILTS), 2))  # no step yet
        _set_grad(w_acc, grads[2].clone())
        w_acc.step_optimizer()
        assert w_acc.pose_step_count == 1
        w_one = self._learning(lr=0.1, opt="sgd")
        _set_grad(w_one, torch.stack(grads).mean(0))
        w_one.step_optimizer()
        assert torch.allclose(w_acc.pose_shifts_A, w_one.pose_shifts_A, atol=1e-7)
        assert torch.equal(w_acc.pose_shifts_A[REF], torch.zeros(2))

    def test_flush_takes_the_residual_step_and_warns_when_short(self):
        w = self._learning(lr=0.1, opt="sgd")
        w.set_pose_accum(steps_per_iter=1, batches_per_epoch=10)  # M = 10
        w.flush_pose_accum()  # nothing pending: no-op, no warning
        assert w.pose_step_count == 0
        for _ in range(2):
            _set_grad(w, torch.ones_like(w._pose_shifts))
            w.step_optimizer()
        with pytest.warns(UserWarning, match="does not match"):
            w.flush_pose_accum()
        assert w.pose_step_count == 1
        assert torch.allclose(w.pose_shifts_A[0], torch.full((2,), -0.1))

    def test_set_pose_accum_requires_batches_per_epoch(self):
        w = _pose_wrapper()
        with pytest.raises(ValueError, match="batches_per_epoch"):
            w.set_pose_accum(steps_per_iter=1)
        w.set_pose_accum(steps_per_iter=0)
        assert w.pose_accum_steps == 1
        w.set_pose_accum(steps_per_iter=4, batches_per_epoch=542)
        assert w.pose_accum_steps == 136  # ceil, so the epoch-end flush takes the last step

    def test_lr_hold_then_decay_to_floor(self):
        w = self._learning(lr=1.0)
        w.set_pose_lr_schedule(hold_steps=2, decay=0.5, floor=0.2)
        seen = []
        for _ in range(6):
            _set_grad(w, torch.ones_like(w._pose_shifts))
            w.step_optimizer()
            seen.append(w.get_current_lr())
        assert seen == pytest.approx([1.0, 1.0, 0.5, 0.25, 0.2, 0.2])
        with pytest.raises(ValueError):
            w.set_pose_lr_schedule(decay=1.5)

    def test_lr_warmup_ramps_then_holds_then_decays(self):
        w = self._learning(lr=1.0)
        w.set_pose_lr_schedule(hold_steps=1, decay=0.5, floor=0.1, warmup_steps=4)
        seen, moved = [], []
        for _ in range(8):
            before = w.pose_shifts_A[0].clone()
            _set_grad(w, torch.ones_like(w._pose_shifts))
            w.step_optimizer()
            seen.append(w.get_current_lr())
            moved.append(float((w.pose_shifts_A[0] - before).abs().max()))
        # LR used AT each step: 0.25, 0.5, 0.75, 1.0 (ramp), 1.0 (hold), then x0.5 -> floor
        assert moved[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0], abs=1e-5)
        assert seen[4:] == pytest.approx([1.0, 0.5, 0.25, 0.125])
        w.reset()
        assert w._pose_lr_base is None  # a reset re-arms the warm-up


# ---------------------------------------------------------------- phase 2: tilt-axis angles
def _set_angle_grad(w: PtychoTomoDatasetRaster, g1: torch.Tensor, g3: torch.Tensor) -> None:
    w._pose_z1.grad = g1  # pyright: ignore[reportAttributeAccessIssue] -- test-only fake gradient
    w._pose_z3.grad = g3  # pyright: ignore[reportAttributeAccessIssue] -- test-only fake gradient


class TestPoseAngles:
    @staticmethod
    def _learning(lr=0.5, opt="adam", shifts=False):
        w = _pose_wrapper()
        if shifts:
            w.set_learn_pose_shifts(True)
        w.set_learn_pose_angles(True)
        spec = OptimizerParams.Adam(lr=lr) if opt == "adam" else OptimizerParams.SGD(lr=lr)
        w.set_optimizer(spec)
        return w

    def test_rotations_follow_the_angle_init(self):
        w = _pose_wrapper()
        z1 = torch.tensor([10.0, 0.0, -25.0])
        z3 = torch.tensor([-5.0, 0.0, 15.0])
        w.set_pose_angle_init(z1, z3)
        assert torch.allclose(w.pose_z1_deg, z1) and torch.allclose(w.pose_z3_deg, z3)
        R = w.rotations()
        expected = rot_beam_to_spec(z1, torch.tensor(TILTS), z3)
        assert torch.allclose(R, expected, atol=1e-6)
        payload, *_ = w.forward(torch.tensor([0, N_PER, 2 * N_PER]), (8, 8))
        assert torch.allclose(payload.rotations, expected, atol=1e-6)
        assert payload.shifts_A is None  # angles alone never touch the shift path
        # a constant z1 is a global rotation of the specimen frame about the beam axis
        with pytest.warns(UserWarning, match="gauge"):
            w.set_pose_angle_init(torch.full((3,), 7.0), torch.zeros(3))
        Rz = rot_beam_to_spec(7.0, 0.0, 0.0)[0]
        R0 = rot_beam_to_spec(0.0, torch.tensor(TILTS), 0.0)
        for t in (0, 2):
            assert torch.allclose(w.rotations()[t], Rz @ R0[t], atol=1e-6)
        assert torch.allclose(w.rotations()[REF], R0[REF], atol=1e-6)  # reference row zeroed

    def test_angle_init_validates_shape_and_zeroes_reference(self):
        w = _pose_wrapper()
        with pytest.raises(ValueError, match="shape"):
            w.set_pose_angle_init(np.zeros(2), np.zeros(3))
        with pytest.raises(ValueError, match="shape"):
            w.set_pose_angle_init(np.zeros(3), np.zeros((3, 1)))
        with pytest.warns(UserWarning, match="gauge"):
            w.set_pose_angle_init([1.0, 0.5, 1.0], [-1.0, 0.0, -1.0])
        assert w.pose_z1_deg[REF].item() == 0.0 and w.pose_z1_init_deg[REF].item() == 0.0
        assert w.pose_z3_deg[REF].item() == 0.0 and w.pose_z3_init_deg[REF].item() == 0.0
        assert w.pose_z1_deg[0].item() == pytest.approx(1.0)

    def test_groups_follow_flags_independently(self):
        w = _pose_wrapper()
        w.set_learn_pose_shifts(True)
        w.set_learn_pose_angles(True)
        assert w._pose_z1.requires_grad and w._pose_z3.requires_grad
        assert not w._pose_dtheta.requires_grad
        w.set_optimizer(OptimizerParams.Adam(lr=0.1))
        assert list(w.get_optimization_parameters()) == ["pose_shifts", "pose_angles"]
        assert len(w.optimizer.param_groups) == 2  # pyright: ignore[reportOptionalMemberAccess]
        w.set_learn_pose_shifts(False)  # angles still learned: optimizer rebuilt, not removed
        assert w.has_optimizer()
        assert list(w.get_optimization_parameters()) == ["pose_angles"]
        assert len(w.optimizer.param_groups) == 1  # pyright: ignore[reportOptionalMemberAccess]
        assert w._pose_z1.requires_grad and not w._pose_shifts.requires_grad
        w.set_learn_pose_angles(False)
        assert not w.has_optimizer()
        assert not w._pose_z1.requires_grad and not w._pose_z3.requires_grad
        # toggling the shift flag never re-freezes the angles
        w.set_learn_pose_angles(True)
        w.set_learn_pose_shifts(True)
        w.set_learn_pose_shifts(False)
        assert w._pose_z1.requires_grad

    def test_single_spec_fans_out_to_both_and_pplr_sets_independent_lrs(self):
        w = _pose_wrapper()
        w.set_learn_pose_shifts(True)
        w.set_learn_pose_angles(True)
        w.optimizer_params = OptimizerParams.Adam(lr=0.3)
        assert list(w.optimizer_params) == ["pose_shifts", "pose_angles"]
        w.set_optimizer(
            {
                "pose_shifts": {"name": "adam", "lr": 0.7},
                "pose_angles": {"name": "adam", "lr": 0.05},
            }
        )
        assert w.pose_group_lr("pose_shifts") == pytest.approx(0.7)
        assert w.pose_group_lr("pose_angles") == pytest.approx(0.05)
        assert w.get_current_lr() == pytest.approx(0.7)  # param_groups[0] stays the shifts

    def test_enabling_a_group_without_a_spec_warns_and_removes(self):
        w = _pose_wrapper()
        w.set_learn_pose_shifts(True)
        w.set_optimizer({"pose_shifts": {"name": "adam", "lr": 0.7}})
        with pytest.warns(UserWarning, match="no optimizer spec"):
            w.set_learn_pose_angles(True)
        assert not w.has_optimizer()
        assert w.learn_pose_angles and w._pose_z1.requires_grad

    def test_reference_rows_pinned_across_a_step(self):
        w = self._learning(lr=0.5)
        _set_angle_grad(w, torch.ones(3), -torch.ones(3))
        w.step_optimizer()
        assert w.pose_z1_deg[REF].item() == 0.0 and w.pose_z3_deg[REF].item() == 0.0
        assert torch.allclose(w.pose_z1_deg[[0, 2]], torch.full((2,), -0.5), atol=1e-6)
        assert torch.allclose(w.pose_z3_deg[[0, 2]], torch.full((2,), 0.5), atol=1e-6)
        assert w.pose_step_count == 1
        assert torch.equal(w.pose_shifts_A, torch.zeros(3, 2))
        rms, mx = w.pose_last_angle_grad_stats
        assert rms == pytest.approx(1.0) and mx == pytest.approx(1.0)

    def test_accumulation_equals_one_step_on_the_mean_gradient(self):
        torch.manual_seed(1)
        g1s = [torch.randn(3) for _ in range(3)]
        g3s = [torch.randn(3) for _ in range(3)]
        w_acc = self._learning(lr=0.1, opt="sgd")
        w_acc.set_pose_accum(steps_per_iter=2, batches_per_epoch=6)  # M = 3
        for g1, g3 in zip(g1s[:2], g3s[:2]):
            _set_angle_grad(w_acc, g1.clone(), g3.clone())
            w_acc.step_optimizer()
            assert torch.equal(w_acc.pose_z1_deg, torch.zeros(3))
        _set_angle_grad(w_acc, g1s[2].clone(), g3s[2].clone())
        w_acc.step_optimizer()
        assert w_acc.pose_step_count == 1
        w_one = self._learning(lr=0.1, opt="sgd")
        _set_angle_grad(w_one, torch.stack(g1s).mean(0), torch.stack(g3s).mean(0))
        w_one.step_optimizer()
        assert torch.allclose(w_acc.pose_z1_deg, w_one.pose_z1_deg, atol=1e-7)
        assert torch.allclose(w_acc.pose_z3_deg, w_one.pose_z3_deg, atol=1e-7)
        assert w_acc.pose_z1_deg[REF].item() == 0.0 and w_acc.pose_z3_deg[REF].item() == 0.0
        # the residual flush steps the angles too
        w = self._learning(lr=0.1, opt="sgd")
        w.set_pose_accum(steps_per_iter=1, batches_per_epoch=10)
        for _ in range(2):
            _set_angle_grad(w, torch.ones(3), torch.ones(3))
            w.step_optimizer()
        with pytest.warns(UserWarning, match="does not match"):
            w.flush_pose_accum()
        assert w.pose_z1_deg[0].item() == pytest.approx(-0.1)

    def test_reset_returns_angles_to_init(self):
        w = self._learning(lr=0.5)
        z1 = torch.tensor([1.0, 0.0, -2.0])
        z3 = torch.tensor([-1.0, 0.0, 2.0])
        w.set_pose_angle_init(z1, z3)
        _set_angle_grad(w, torch.ones(3), torch.ones(3))
        w.step_optimizer()
        assert not torch.allclose(w.pose_z1_deg, z1)
        w.reset()
        assert torch.allclose(w.pose_z1_deg, z1) and torch.allclose(w.pose_z3_deg, z3)
        assert torch.equal(w._pose_dtheta, torch.zeros(3))
        assert w.pose_step_count == 0 and w.learn_pose_angles

    def test_independent_lrs_warmup_and_floors_per_group(self):
        w = self._learning(shifts=True)
        w.set_optimizer(
            {
                "pose_shifts": {"name": "adam", "lr": 1.0},
                "pose_angles": {"name": "adam", "lr": 0.2},
            }
        )
        w.set_pose_lr_schedule(
            hold_steps=1, decay=0.5, floor=0.3, warmup_steps=4, angle_floor=0.02
        )
        moved_s, moved_a, lr_a = [], [], []
        for _ in range(8):
            s0, a0 = w.pose_shifts_A[0].clone(), w.pose_z1_deg[0].clone()
            _set_grad(w, torch.ones(3, 2))
            _set_angle_grad(w, torch.ones(3), torch.ones(3))
            w.step_optimizer()
            moved_s.append(float((w.pose_shifts_A[0] - s0).abs().max()))
            moved_a.append(float((w.pose_z1_deg[0] - a0).abs()))
            lr_a.append(w.pose_group_lr("pose_angles"))
        assert moved_s[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0], abs=1e-5)
        assert moved_a[:4] == pytest.approx([0.05, 0.1, 0.15, 0.2], abs=1e-5)
        assert lr_a[4:] == pytest.approx([0.2, 0.1, 0.05, 0.025])
        assert w.pose_group_lr("pose_shifts") == pytest.approx(0.3)  # shift floor, separately
        for _ in range(2):
            _set_grad(w, torch.ones(3, 2))
            _set_angle_grad(w, torch.ones(3), torch.ones(3))
            w.step_optimizer()
        assert w.pose_group_lr("pose_angles") == pytest.approx(0.02)  # angle floor

    def test_shifts_only_schedule_unchanged_with_angle_code_present(self):
        """The phase-1 warm-up/hold/decay numbers, re-asserted with the angle flag explicitly
        off: the angle machinery must be a no-op on the certified shift path."""
        w = _pose_wrapper()
        w.set_learn_pose_angles(False)
        w.set_learn_pose_shifts(True)
        w.set_optimizer(OptimizerParams.Adam(lr=1.0))
        w.set_pose_lr_schedule(hold_steps=1, decay=0.5, floor=0.1, warmup_steps=4)
        seen, moved = [], []
        for _ in range(8):
            before = w.pose_shifts_A[0].clone()
            _set_grad(w, torch.ones_like(w._pose_shifts))
            w.step_optimizer()
            seen.append(w.get_current_lr())
            moved.append(float((w.pose_shifts_A[0] - before).abs().max()))
        assert moved[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0], abs=1e-5)
        assert seen[4:] == pytest.approx([1.0, 0.5, 0.25, 0.125])
        assert torch.equal(w.pose_z1_deg, torch.zeros(3)) and torch.equal(
            w.pose_z3_deg, torch.zeros(3)
        )
        assert not w._pose_z1.requires_grad

    def test_autoserialize_roundtrip_carries_angles(self, tmp_path):
        w = _pose_wrapper()
        w.set_pose_angle_init([1.0, 0.0, -1.0], [-2.0, 0.0, 2.0])
        w.set_learn_pose_angles(True)
        path = tmp_path / "tomo_angles.zip"
        w.save(path, mode="o")
        w2 = autoserialize_load(path)
        assert isinstance(w2, PtychoTomoDatasetRaster)
        assert w2.learn_pose_angles
        assert torch.allclose(w2.pose_z1_deg, torch.tensor([1.0, 0.0, -1.0]))
        assert torch.allclose(w2.pose_z3_init_deg, torch.tensor([-2.0, 0.0, 2.0]))

    def test_phase1_wrapper_without_angle_buffers_still_works(self):
        """A wrapper deserialized from a phase-1 cache has no angle buffers: every angle entry
        point materialises them (zeros) instead of failing."""
        w = _pose_wrapper()
        for name in ("_pose_z1_init", "_pose_z3_init", "_pose_z1_accum", "_pose_z3_accum"):
            del w._buffers[name]
        del w._learn_pose_angles
        assert not w.learn_pose_angles
        w.set_learn_pose_shifts(True)
        w.set_optimizer(OptimizerParams.SGD(lr=0.1))
        _set_grad(w, torch.ones(3, 2))
        w.step_optimizer()  # shift-only path never touches the missing buffers
        w.reset()
        assert torch.equal(w.pose_z1_init_deg, torch.zeros(3))
        with pytest.warns(UserWarning, match="no optimizer spec"):
            w.set_learn_pose_angles(True)
        assert list(w.get_optimization_parameters()) == ["pose_shifts", "pose_angles"]


def _set_defocus_grad(w: PtychoTomoDatasetRaster, g: torch.Tensor) -> None:
    w._defocus_offset_A.grad = g  # pyright: ignore[reportAttributeAccessIssue] -- test-only fake gradient


class TestPoseDefocus:
    """Task 2: per-dataset probe defocus offset learned inline (the twin of the angle slot)."""

    @staticmethod
    def _learning(lr=0.5, opt="adam", shifts=False, angles=False):
        w = _pose_wrapper()
        if shifts:
            w.set_learn_pose_shifts(True)
        if angles:
            w.set_learn_pose_angles(True)
        w.set_learn_defocus(True)
        spec = OptimizerParams.Adam(lr=lr) if opt == "adam" else OptimizerParams.SGD(lr=lr)
        w.set_optimizer(spec)
        return w

    def test_parameter_shape_dtype_and_frozen_at_construction(self):
        w = _pose_wrapper()
        p = w._defocus_offset_A
        assert isinstance(p, torch.nn.Parameter)
        assert p.shape == (len(TILTS),)  # one row per constituent dataset
        assert p.dtype == getattr(torch, config.get("dtype_real"))
        assert p.device == w._pose_shifts.device
        assert not p.requires_grad and not w.learn_defocus
        assert torch.equal(w.defocus_offset_A, torch.zeros(3))
        assert torch.equal(w.defocus_offset_init_A, torch.zeros(3))
        assert w.get_optimization_parameters() == {}
        w.forward(torch.tensor([0, N_PER, 2 * N_PER]), (8, 8))
        assert w._last_probe_dz_A is None  # nothing active: no probe pre-propagation

    def test_init_broadcasts_validates_and_keeps_reference(self):
        """No gauge on the defocus slot (2026-09-07): the reference row's init is kept, not
        zeroed, and moving the reference does not warn about a nonzero defocus baseline."""
        w = _pose_wrapper()
        with pytest.raises(ValueError, match="shape"):
            w.set_defocus_init(np.zeros(2))
        with pytest.raises(ValueError, match="shape"):
            w.set_defocus_init(np.zeros((3, 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w.set_defocus_init(5.0)  # a scalar broadcasts to EVERY row, the reference included
        assert torch.equal(w.defocus_offset_A, torch.tensor([5.0, 5.0, 5.0]))
        assert torch.equal(w.defocus_offset_init_A, torch.tensor([5.0, 5.0, 5.0]))
        with torch.no_grad():
            w._defocus_offset_accum.fill_(1.0)
            w._pose_shifts_accum.fill_(1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w.set_defocus_init([1.0, 2.0, -3.0])
        assert torch.equal(w.defocus_offset_A, torch.tensor([1.0, 2.0, -3.0]))
        assert torch.equal(w._defocus_offset_accum, torch.zeros(3))
        assert torch.equal(w._pose_shifts_accum, torch.zeros(3, 2))
        assert w._defocus_active and not w.learn_defocus  # nonzero table applies even frozen
        w.set_defocus_init(0.0)
        assert not w._defocus_active
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w.set_defocus_init([0.0, 0.0, 4.0])
            w.reference_tilt_idx = 2  # no "defocus baseline" warning: nothing is pinned
        assert torch.equal(w.defocus_offset_A, torch.tensor([0.0, 0.0, 4.0]))
        # the shift gauge is untouched by the defocus rule: its reference row is still zeroed
        with pytest.warns(UserWarning, match="gauge"):
            w.set_pose_shift_init(np.ones((3, 2)))
        assert torch.equal(w.pose_shifts_A[2], torch.zeros(2))

    def test_reset_returns_to_baseline_and_keeps_flag(self):
        w = self._learning(lr=0.5)
        init = torch.tensor([1.0, 0.0, -2.0])
        w.set_defocus_init(init)
        _set_defocus_grad(w, torch.ones(3))
        w.step_optimizer()
        assert not torch.allclose(w.defocus_offset_A, init)
        w.reset()
        assert torch.allclose(w.defocus_offset_A, init)
        assert w.learn_defocus and w._defocus_offset_A.requires_grad and w._defocus_active
        assert w.pose_step_count == 0
        assert torch.equal(w.pose_shifts_A, torch.zeros(3, 2))
        assert torch.equal(w.pose_z1_deg, torch.zeros(3))

    def test_groups_follow_flags_and_keep_a_stable_order(self):
        w = _pose_wrapper()
        w.set_learn_defocus(True)
        assert w._defocus_offset_A.requires_grad and w._defocus_active
        assert list(w.get_optimization_parameters()) == ["pose_defocus"]
        w.set_learn_pose_shifts(True)
        w.set_learn_pose_angles(True)
        w.set_optimizer(OptimizerParams.Adam(lr=0.1))
        assert list(w.get_optimization_parameters()) == [
            "pose_shifts",
            "pose_angles",
            "pose_defocus",
        ]
        assert len(w.optimizer.param_groups) == 3  # pyright: ignore[reportOptionalMemberAccess]
        w.set_learn_pose_shifts(False)  # rebuilt, not removed
        assert w.has_optimizer()
        assert list(w.get_optimization_parameters()) == ["pose_angles", "pose_defocus"]
        assert len(w.optimizer.param_groups) == 2  # pyright: ignore[reportOptionalMemberAccess]
        w.set_learn_pose_angles(False)
        assert list(w.get_optimization_parameters()) == ["pose_defocus"]
        assert w._defocus_offset_A.requires_grad and not w._pose_z1.requires_grad
        w.set_learn_defocus(False)
        assert not w.has_optimizer()
        assert not w._defocus_offset_A.requires_grad and not w._defocus_active
        assert not w._pose_dtheta.requires_grad

    def test_single_spec_fans_out_to_all_three_and_pplr_sets_independent_lrs(self):
        w = _pose_wrapper()
        w.set_learn_pose_shifts(True)
        w.set_learn_pose_angles(True)
        w.set_learn_defocus(True)
        w.optimizer_params = OptimizerParams.Adam(lr=0.3)
        assert list(w.optimizer_params) == ["pose_shifts", "pose_angles", "pose_defocus"]
        w.set_optimizer(
            {
                "pose_shifts": {"name": "adam", "lr": 0.7},
                "pose_angles": {"name": "adam", "lr": 0.05},
                "pose_defocus": {"name": "adam", "lr": 2.0},
            }
        )
        assert w.pose_group_lr("pose_shifts") == pytest.approx(0.7)
        assert w.pose_group_lr("pose_angles") == pytest.approx(0.05)
        assert w.pose_group_lr("pose_defocus") == pytest.approx(2.0)
        assert w.get_current_lr() == pytest.approx(0.7)

    def test_enabling_defocus_without_a_spec_warns_and_removes(self):
        w = _pose_wrapper()
        w.set_learn_pose_shifts(True)
        w.set_optimizer({"pose_shifts": {"name": "adam", "lr": 0.7}})
        with pytest.warns(UserWarning, match="no optimizer spec"):
            w.set_learn_defocus(True)
        assert not w.has_optimizer()
        assert w.learn_defocus and w._defocus_offset_A.requires_grad

    def test_reference_row_moves_like_any_other_across_a_step_and_grad_stats(self):
        """The reference row is NOT pinned for the defocus slot: it steps with the others,
        and the gradient stats are taken over all rows."""
        w = self._learning(lr=0.5)
        _set_defocus_grad(w, torch.tensor([1.0, 1.0, -1.0]))
        w.step_optimizer()
        assert torch.allclose(w.defocus_offset_A, torch.tensor([-0.5, -0.5, 0.5]), atol=1e-6)
        assert w.pose_step_count == 1
        rms, mx = w.pose_last_defocus_grad_stats
        assert rms == pytest.approx(1.0) and mx == pytest.approx(1.0)
        # a nonzero reference init is kept and the row keeps moving from there
        w2 = self._learning(lr=0.5)
        w2.set_defocus_init([0.0, 3.0, 0.0])
        _set_defocus_grad(w2, torch.tensor([0.0, 1.0, 0.0]))
        w2.step_optimizer()
        assert w2.defocus_offset_A[REF].item() == pytest.approx(2.5, abs=1e-6)
        w2.reset()
        assert w2.defocus_offset_A[REF].item() == 3.0
        assert torch.equal(w.pose_shifts_A, torch.zeros(3, 2))  # other slots untouched
        assert torch.equal(w.pose_z1_deg, torch.zeros(3))

    def test_accumulation_equals_one_step_on_the_mean_gradient(self):
        torch.manual_seed(2)
        gs = [torch.randn(3) for _ in range(3)]
        w_acc = self._learning(lr=0.1, opt="sgd")
        w_acc.set_pose_accum(steps_per_iter=2, batches_per_epoch=6)  # M = 3
        for g in gs[:2]:
            _set_defocus_grad(w_acc, g.clone())
            w_acc.step_optimizer()
            assert torch.equal(w_acc.defocus_offset_A, torch.zeros(3))  # not stepped yet
        _set_defocus_grad(w_acc, gs[2].clone())
        w_acc.step_optimizer()
        assert w_acc.pose_step_count == 1
        w_one = self._learning(lr=0.1, opt="sgd")
        _set_defocus_grad(w_one, torch.stack(gs).mean(0))
        w_one.step_optimizer()
        assert torch.allclose(w_acc.defocus_offset_A, w_one.defocus_offset_A, atol=1e-7)
        assert w_acc.defocus_offset_A[REF].item() != 0.0  # the reference row steps too
        assert torch.equal(w_acc._defocus_offset_accum, torch.zeros(3))
        # the residual flush steps the defocus too
        w = self._learning(lr=0.1, opt="sgd")
        w.set_pose_accum(steps_per_iter=1, batches_per_epoch=10)
        for _ in range(2):
            _set_defocus_grad(w, torch.ones(3))
            w.step_optimizer()
        with pytest.warns(UserWarning, match="does not match"):
            w.flush_pose_accum()
        assert w.defocus_offset_A[0].item() == pytest.approx(-0.1)

    def test_independent_lr_floor_warmup_and_decay_for_the_defocus_group(self):
        w = self._learning(shifts=True, angles=True)
        w.set_optimizer(
            {
                "pose_shifts": {"name": "adam", "lr": 1.0},
                "pose_angles": {"name": "adam", "lr": 0.2},
                "pose_defocus": {"name": "adam", "lr": 0.4},
            }
        )
        w.set_pose_lr_schedule(
            hold_steps=1, decay=0.5, floor=0.3, warmup_steps=4, angle_floor=0.02, defocus_floor=0.1
        )
        moved_d, lr_d = [], []
        for _ in range(8):
            d0 = w.defocus_offset_A[0].clone()
            _set_grad(w, torch.ones(3, 2))
            _set_angle_grad(w, torch.ones(3), torch.ones(3))
            _set_defocus_grad(w, torch.ones(3))
            w.step_optimizer()
            moved_d.append(float((w.defocus_offset_A[0] - d0).abs()))
            lr_d.append(w.pose_group_lr("pose_defocus"))
        assert moved_d[:4] == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=1e-5)  # warm-up ramp
        assert lr_d[4:] == pytest.approx([0.4, 0.2, 0.1, 0.1])  # hold, decay, defocus floor
        assert w.pose_group_lr("pose_shifts") == pytest.approx(0.3)  # the other floors, separately
        assert w.pose_group_lr("pose_angles") == pytest.approx(0.025)

    def test_shift_and_angle_paths_unchanged_with_defocus_code_present(self):
        """The certified shift / angle machinery must be a no-op on the defocus slot: the
        phase-1 schedule numbers re-asserted with the defocus flag explicitly off."""
        w = _pose_wrapper()
        w.set_learn_defocus(False)
        w.set_learn_pose_shifts(True)
        w.set_learn_pose_angles(True)
        w.set_optimizer(OptimizerParams.Adam(lr=1.0))
        w.set_pose_lr_schedule(hold_steps=1, decay=0.5, floor=0.1, warmup_steps=4)
        seen, moved = [], []
        for _ in range(8):
            before = w.pose_shifts_A[0].clone()
            _set_grad(w, torch.ones_like(w._pose_shifts))
            _set_angle_grad(w, torch.ones(3), torch.ones(3))
            w.step_optimizer()
            seen.append(w.get_current_lr())
            moved.append(float((w.pose_shifts_A[0] - before).abs().max()))
        assert moved[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0], abs=1e-5)
        assert seen[4:] == pytest.approx([1.0, 0.5, 0.25, 0.125])
        assert torch.equal(w.defocus_offset_A, torch.zeros(3))
        assert not w._defocus_offset_A.requires_grad and not w._defocus_active
        assert list(w.get_optimization_parameters()) == ["pose_shifts", "pose_angles"]
        w.forward(torch.tensor([0, N_PER, 2 * N_PER]), (8, 8))
        assert w._last_probe_dz_A is None

    def test_autoserialize_roundtrip_carries_defocus(self, tmp_path):
        w = _pose_wrapper()
        w.set_defocus_init([1.0, 0.0, -2.0])
        w.set_learn_defocus(True)
        path = tmp_path / "tomo_defocus.zip"
        w.save(path, mode="o")
        w2 = autoserialize_load(path)
        assert isinstance(w2, PtychoTomoDatasetRaster)
        assert w2.learn_defocus and w2._defocus_active
        assert torch.allclose(w2.defocus_offset_A, torch.tensor([1.0, 0.0, -2.0]))
        assert torch.allclose(w2.defocus_offset_init_A, torch.tensor([1.0, 0.0, -2.0]))
        idx = torch.tensor([0, N_PER, 2 * N_PER + 3])
        w2.forward(idx, (8, 8))
        assert w2._last_probe_dz_A is not None
        assert torch.allclose(w2._last_probe_dz_A, torch.tensor([-1.0, 0.0, 2.0]))

    def test_pre_defocus_wrapper_materialises_zeros(self):
        """A wrapper deserialized from a phase-1/2 cache has no defocus parameter, buffers or
        flags: the shift / angle path never touches them and every defocus entry point
        materialises zeros instead of failing."""
        w = _pose_wrapper()
        del w._parameters["_defocus_offset_A"]
        for name in ("_defocus_offset_init_A", "_defocus_offset_accum"):
            del w._buffers[name]
        del w._learn_defocus
        del w._defocus_active
        assert not w.learn_defocus and not w._defocus_active  # class-level defaults
        w.forward(torch.tensor([0, N_PER]), (8, 8))
        assert w._last_probe_dz_A is None
        w.set_learn_pose_shifts(True)
        w.set_optimizer(OptimizerParams.SGD(lr=0.1))
        _set_grad(w, torch.ones(3, 2))
        w.step_optimizer()  # shift-only path never touches the missing state
        assert not w._has_defocus_state()
        w.reset()
        assert w._has_defocus_state()
        assert torch.equal(w.defocus_offset_A, torch.zeros(3))
        assert torch.equal(w.defocus_offset_init_A, torch.zeros(3))
        with pytest.warns(UserWarning, match="no optimizer spec"):
            w.set_learn_defocus(True)
        assert list(w.get_optimization_parameters()) == ["pose_shifts", "pose_defocus"]

    def test_forward_stash_is_the_probe_offset_and_never_the_object_query(self):
        w = _pose_wrapper()
        idx = torch.tensor([0, 3, N_PER, 2 * N_PER + 1, 2 * N_PER])
        table = torch.tensor([1.5, 0.0, -4.0])
        w.set_defocus_init(table)
        payload, *_ = w.forward(idx, (8, 8))
        dz = w._last_probe_dz_A
        assert dz is not None and dz.shape == (5,)
        # no slab window: the probe is propagated by MINUS the offset (offset = +d raises the
        # effective defocus by d; propagation lowers it), gathered per position
        assert torch.allclose(dz, -table[w.tilt_index_of(idx)])
        assert payload.window_dz_A is None and payload.shifts_A is None
        # a live (frozen) table keeps applying: the value, not the flag, decides
        assert not w.learn_defocus
        # learning with a zero table is still "active" (the gather must build the graph)
        w.set_defocus_init(0.0)
        w.set_learn_defocus(True)
        w.forward(idx, (8, 8))
        assert w._last_probe_dz_A is not None and torch.equal(w._last_probe_dz_A, torch.zeros(5))
        w.set_learn_defocus(False)
        # slab window on: the stash is window_dz - offset while the payload's window offset
        # (the object query) does not see the defocus at all
        w.set_slab_window(True)
        w.set_defocus_init(table)
        idx = torch.arange(2 * N_PER, 3 * N_PER)  # the +35 deg tilt
        rot = w.rotations()[w.tilt_index_of(idx)]
        window = w._window_dz_A(idx, rot)
        payload, *_ = w.forward(idx, (8, 8))
        assert payload.window_dz_A is not None
        assert torch.allclose(payload.window_dz_A, window)
        assert w._last_probe_dz_A is not None
        assert torch.allclose(w._last_probe_dz_A, window + 4.0)


class TestScanCenterVerify:
    """A4 (2026-09-09): a cached wrapper's stored pivot is checked against the pivot its
    requested mode implies for the stored positions -- an old cache (no mode attribute, grid
    pivot) must be refused in 'positions' mode and accepted in 'grid' mode."""

    def test_for_mode_matches_set_scan_center(self):
        w = _build_wrapper()
        assert torch.allclose(
            w.scan_center_px_for_mode("positions"), w._scan_center_px.cpu().to(torch.float64)
        )
        grid = w.scan_center_px_for_mode("grid")
        full2d = w._obj_shape_full_2d((8, 8))
        expected = [(int(full2d[0]) - 1) / 2.0, (int(full2d[1]) - 1) / 2.0]
        assert torch.allclose(grid, torch.tensor(expected, dtype=torch.float64))
        assert (grid - w.scan_center_px_for_mode("positions")).abs().max() > 1e-3
        w.verify_scan_center("positions")
        with pytest.raises(ValueError, match="pivot"):
            w.verify_scan_center("grid")
        with pytest.raises(ValueError, match="scan_center_mode"):
            w.scan_center_px_for_mode("bogus")

    def test_old_cache_is_caught(self, tmp_path):
        # an old cache: preprocessed with the grid pivot, and the mode attribute never written
        dsets = [_make_dset4d(seed=i) for i in range(len(TILTS))]
        w = PtychoTomoDatasetRaster.from_dataset4dstem_list(dsets, TILTS, verbose=0)
        w.set_scan_center_mode("grid")
        w.preprocess(obj_padding_px=(8, 8))
        path = tmp_path / "old_cache.zip"
        w.save(path, mode="o")
        loaded = autoserialize_load(path)
        del loaded.__dict__["_scan_center_mode"]
        assert loaded.scan_center_mode == "positions"  # the class default LIES about the cache
        with pytest.raises(ValueError, match="mismatch"):
            loaded.verify_scan_center("positions")
        loaded.verify_scan_center("grid")  # the pivot it stores IS the grid point
        # and a cache that was built in positions mode passes as itself
        w2 = _build_wrapper()
        path2 = tmp_path / "new_cache.zip"
        w2.save(path2, mode="o")
        autoserialize_load(path2).verify_scan_center("positions")
