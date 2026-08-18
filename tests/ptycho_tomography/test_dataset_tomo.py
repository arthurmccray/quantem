"""Tests for the multi-tilt ptycho-tomography dataset wrapper.

Uses small random Dataset4dstem stacks (the wrapper's mechanics don't need physical patterns);
the physically meaningful end-to-end checks live in test_ptycho_tomography.py.
"""

import json

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


def _to_legacy(obj: torch.nn.Module) -> None:
    """Rewrite the CoM-transpose flag to its pre-merge attribute name: exactly the state a cache
    written before the 2026-08-17 merge restores (deserialization bypasses ``__init__``)."""
    value = obj.com_transpose  # pyright: ignore[reportAttributeAccessIssue] -- ptycho datasets only
    delattr(obj, "_transpose")
    setattr(obj, "_com_transpose", value)


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


class TestPreMergeCacheBackCompat:
    """Wraps written before the 2026-08-17 diffractive_imaging merge stored the CoM-transpose
    flag as ``_com_transpose``; deserialization bypasses ``__init__``, so without the shim the
    getter finds neither that name nor the current ``_transpose``."""

    def test_legacy_attribute_name_is_normalized(self):
        w = _build_wrapper()
        _to_legacy(w)
        assert w.com_transpose is False
        assert "_com_transpose" not in w.__dict__  # renamed in place, so a re-save is clean

    def test_legacy_true_value_is_preserved(self):
        w = _build_wrapper()
        delattr(w, "_transpose")
        setattr(w, "_com_transpose", True)
        assert w.com_transpose is True

    def test_legacy_per_tilt_datasets_are_normalized(self):
        w = _build_wrapper()
        _to_legacy(w)
        for ds in w.tilt_datasets:
            _to_legacy(ds)
        _ = w.com_transpose
        for ds in w.tilt_datasets:
            assert ds.com_transpose is False

    def test_missing_flag_defaults_to_false(self):
        """Neither name present (an even older save) -> the __init__ default, not a crash."""
        w = _build_wrapper()
        delattr(w, "_transpose")
        assert w.com_transpose is False

    def test_reload_of_rewritten_store(self, tmp_path):
        """The real failure: rewrite the persisted key, reload, and use the wrap."""
        w = _build_wrapper()
        w.implicit_object = True
        path = tmp_path / "tomo_dset_legacy"
        w.save(path, mode="o")
        meta_path = path / "zarr.json"
        meta = json.loads(meta_path.read_text())
        attrs = meta["attributes"]
        assert "_transpose" in attrs, "current code should persist _transpose"
        attrs["_com_transpose"] = attrs.pop("_transpose")  # pre-merge on-disk layout
        meta_path.write_text(json.dumps(meta))

        loaded = autoserialize_load(path)
        assert loaded.com_transpose is False
        # the geometry chain that raised the misleading _obj_shape_rot_2d AttributeError
        assert tuple(loaded._obj_shape_crop_2d) == tuple(w._obj_shape_crop_2d)
        payload, *_ = loaded.forward(torch.tensor([0, 25]), (8, 8))
        assert payload.rotations.shape == (2, 3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Pose refinement (task 1) and per-tilt defocus (task 2)
# ---------------------------------------------------------------------------
PAD = (8, 8)
REF_TILT = 1  # argmin |tilt| for TILTS = [-35, 0, 35]


def _pose_wrapper() -> PtychoTomoDatasetRaster:
    w = _build_wrapper()
    w.implicit_object = True
    return w


def _matched_object(w: PtychoTomoDatasetRaster, num_slices: int = 3):
    """A smooth, low-frequency, asymmetric voxel object on the wrapper's own padded grid.

    Used as a differentiable probe of the pose parameters: the object query is the only place
    the shifts/angles enter the forward, so d(query)/d(pose) is what the reconstruction sees.
    The density is deliberately smooth on the VOXEL scale — the backend is trilinear, so a
    sharp feature makes the query piecewise-linear with kinks every voxel and a central finite
    difference then averages two different slopes.
    """
    from quantem.ptycho_tomography.object_models import ObjectVoxelTomo

    full2d = w._obj_shape_full_2d(PAD)
    h, ww = int(full2d[0]), int(full2d[1])
    nz = 9
    thickness = 9.0
    zz = torch.linspace(-1.0, 1.0, nz)[:, None, None]
    yy = torch.linspace(-1.0, 1.0, h)[None, :, None]
    xx = torch.linspace(-1.0, 1.0, ww)[None, None, :]
    vol = 0.02 * (
        1.0
        + 0.5 * torch.sin(2.0 * yy + 0.3)
        + 0.4 * torch.cos(1.7 * xx - 0.2)
        + 0.3 * zz
        + 0.25 * yy * xx
    )
    obj = ObjectVoxelTomo.from_array(vol, thickness_A=thickness, num_slices=num_slices)
    samp = w.obj_sampling
    obj._initialize_obj((num_slices, h, ww), sampling=(float(samp[0]), float(samp[1])))
    return obj


def _fd_batch(w: PtychoTomoDatasetRaster) -> torch.Tensor:
    """One scan position per tilt: keeps the summed functional small enough that a float32
    central difference is not swamped by cancellation."""
    n = w.num_gpts // w.num_tilts
    return torch.tensor([2, n + 7, 2 * n + 11])


def _pose_scalar(w: PtychoTomoDatasetRaster, obj, batch: torch.Tensor) -> torch.Tensor:
    """Smooth scalar functional of the pose-dependent object query."""
    payload, *_ = w.forward(batch, PAD)
    return obj.forward(payload).imag.sum()


def _finite_difference(w, obj, batch, param: torch.Tensor, index, h: float) -> float:
    base = param.data.clone()
    with torch.no_grad():
        param.data[index] = base[index] + h
    plus = float(_pose_scalar(w, obj, batch).detach())
    with torch.no_grad():
        param.data[index] = base[index] - h
    minus = float(_pose_scalar(w, obj, batch).detach())
    with torch.no_grad():
        param.data.copy_(base)
    return (plus - minus) / (2 * h)


class TestPoseGradients:
    """Gradients of the object query w.r.t. the pose, against central finite differences."""

    @pytest.mark.parametrize(
        "slot, step",
        [("_pose_shifts", 0.02), ("_pose_z1", 0.05), ("_pose_z3", 0.05), ("_pose_dtheta", 0.05)],
    )
    def test_pose_gradient_matches_finite_difference(self, slot, step):
        w = _pose_wrapper()
        obj = _matched_object(w)
        w.set_learn_pose(shifts=True, z1=True, z3=True, dtheta=True)
        batch = _fd_batch(w)
        param = getattr(w, slot)
        _pose_scalar(w, obj, batch).backward()
        grad = param.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().max() > 0, f"{slot}: no gradient reached the pose parameter"
        if param.ndim == 2:
            indices = [(t, c) for t in range(w.num_tilts) for c in range(int(param.shape[1]))]
        else:
            indices = [t for t in range(w.num_tilts)]
        checked = 0
        for index in indices:
            analytic = float(grad[index])
            numeric = _finite_difference(w, obj, batch, param, index, step)
            scale = max(abs(analytic), abs(numeric))
            if scale < 1e-4:
                continue  # below the float32 finite-difference noise floor
            checked += 1
            assert analytic * numeric > 0, f"{slot}{index}: sign flip ({analytic} vs {numeric})"
            assert abs(analytic - numeric) < 0.05 * scale, (
                f"{slot}{index}: {analytic} vs finite difference {numeric}"
            )
        assert checked >= 3, f"{slot}: finite-difference check was vacuous ({checked} entries)"

    def test_shift_gradient_sign_matches_a_shifted_query(self):
        """A +dy shift must move the queried object the same way -dy of scan position does."""
        w = _pose_wrapper()
        obj = _matched_object(w)
        batch = _fd_batch(w)
        with torch.no_grad():
            w.set_pose_init(shifts=[0.0, 0.0])
        payload_a, *_ = w.forward(batch, PAD)
        coords_shifted = payload_a.coords_yx_A.clone()
        coords_shifted[..., 0] -= 0.7  # move the coordinates by -0.7 Å in row
        from quantem.ptycho_tomography.geometry import PtychoTomoPatchData

        ref = obj.forward(
            PtychoTomoPatchData(
                coords_yx_A=coords_shifted,
                rotations=payload_a.rotations,
                tilt_indices=payload_a.tilt_indices,
            )
        )
        w.set_pose_init(shifts=[0.7, 0.0])  # subtracted from the coordinates => same thing
        payload_b, *_ = w.forward(batch, PAD)
        assert torch.allclose(obj.forward(payload_b), ref, atol=1e-6)


class TestPoseGaugeAndOptimizer:
    def _learning_wrapper(self, **kwargs) -> PtychoTomoDatasetRaster:
        w = _pose_wrapper()
        w.set_learn_pose(shifts=True, z1=True, z3=True, dtheta=True)
        w.set_learn_defocus(True)
        return w

    def test_groups_appear_and_disappear_with_learn_flags(self):
        w = _pose_wrapper()
        assert w.get_optimization_parameters() == {}
        w.set_learn_pose(shifts=True, z1=False, z3=False)
        assert set(w.get_optimization_parameters()) == {"pose_shifts"}
        w.set_learn_pose(shifts=False, z1=True, z3=True)
        groups = w.get_optimization_parameters()
        assert set(groups) == {"pose_angles"} and len(groups["pose_angles"]) == 2
        w.set_learn_pose(shifts=True, z1=True, z3=True, dtheta=True)
        groups = w.get_optimization_parameters()
        assert set(groups) == {"pose_shifts", "pose_angles"}
        assert len(groups["pose_angles"]) == 3  # z1 + z3 + dtheta
        w.set_learn_defocus(True)
        assert set(w.get_optimization_parameters()) == {
            "pose_shifts",
            "pose_angles",
            "defocus",
        }
        w.set_learn_pose(shifts=False, z1=False, z3=False)
        w.set_learn_defocus(False)
        assert w.get_optimization_parameters() == {}

    def test_requires_grad_follows_learn_flags(self):
        w = _pose_wrapper()
        for p in (w._pose_shifts, w._pose_z1, w._pose_z3, w._pose_dtheta, w._defocus_offset_A):
            assert not p.requires_grad
        w.set_learn_pose(shifts=True, z1=True, z3=False, dtheta=True)
        w.set_learn_defocus(True)
        assert w._pose_shifts.requires_grad and w._pose_z1.requires_grad
        assert w._pose_dtheta.requires_grad and w._defocus_offset_A.requires_grad
        assert not w._pose_z3.requires_grad

    def test_single_spec_fans_out_to_enabled_groups(self):
        from quantem.core.ml.optimizer_mixin import OptimizerParams

        w = self._learning_wrapper()
        w.set_optimizer(OptimizerParams.Adam(lr=1e-3))
        assert w.optimizer is not None
        assert len(w.optimizer.param_groups) == 3  # shifts, angles, defocus
        assert set(w.optimizer_params) == {"pose_shifts", "pose_angles", "defocus"}

    def test_explicit_pplr_dict_passes_through(self):
        from quantem.core.ml.optimizer_mixin import OptimizerParams

        w = self._learning_wrapper()
        w.set_optimizer(
            {
                "pose_shifts": OptimizerParams.Adam(lr=1e-3),
                "pose_angles": OptimizerParams.Adam(lr=1e-2),
                "defocus": OptimizerParams.Adam(lr=1e-1),
            }
        )
        assert w.optimizer is not None
        lrs = sorted(g["lr"] for g in w.optimizer.param_groups)
        assert lrs == pytest.approx([1e-3, 1e-2, 1e-1])

    def test_nothing_learnable_warns_and_removes_optimizer(self):
        from quantem.core.ml.optimizer_mixin import OptimizerParams

        w = _pose_wrapper()
        with pytest.warns(UserWarning, match="nothing is learnable"):
            w.set_optimizer(OptimizerParams.Adam(lr=1e-3))
        assert not w.has_optimizer()

    def test_reference_tilt_pinned_across_a_step(self):
        from quantem.core.ml.optimizer_mixin import OptimizerParams

        w = self._learning_wrapper()
        w.set_optimizer(OptimizerParams.Adam(lr=1e-2))
        assert w.reference_tilt_idx == REF_TILT
        for param, _init in w._reference_pose_slots():
            param.grad = torch.ones_like(param)
        w.step_optimizer()
        for param, init in w._reference_pose_slots():
            assert torch.equal(param.data[REF_TILT], init[REF_TILT]), "reference tilt moved"
            moved = [t for t in range(w.num_tilts) if t != REF_TILT]
            assert (param.data[moved] != init[moved]).all(), "non-reference tilts did not move"

    def test_reference_tilt_pinned_to_a_nonzero_baseline(self):
        """set_tilt_axis_pose / the rotation-center wrapper set nonzero baselines; the gauge
        holds the reference tilt THERE, it does not hard-zero it."""
        from quantem.core.ml.optimizer_mixin import OptimizerParams

        w = self._learning_wrapper()
        w.set_tilt_axis_pose(-90.0, 90.0)
        w.set_pose_init(shifts=[0.4, -0.2])
        w.set_optimizer(OptimizerParams.Adam(lr=1e-2))
        for param, _init in w._reference_pose_slots():
            param.grad = torch.ones_like(param)
        w.step_optimizer()
        assert float(w._pose_z1.detach()[REF_TILT]) == pytest.approx(-90.0)
        assert float(w._pose_z3.detach()[REF_TILT]) == pytest.approx(90.0)
        assert torch.allclose(w._pose_shifts.detach()[REF_TILT], torch.tensor([0.4, -0.2]))

    def test_reference_tilt_gradients_are_zeroed(self):
        w = self._learning_wrapper()
        for param, _init in w._reference_pose_slots():
            param.grad = torch.ones_like(param)
        w.zero_reference_pose_grads()
        for param, _init in w._reference_pose_slots():
            assert param.grad is not None
            assert float(param.grad[REF_TILT].abs().sum()) == 0.0
            assert float(param.grad.abs().sum()) > 0.0

    def test_reference_tilt_idx_is_settable_and_validated(self):
        w = _pose_wrapper()
        w.reference_tilt_idx = 2
        assert w.reference_tilt_idx == 2
        with pytest.raises(ValueError, match="reference_tilt_idx"):
            w.reference_tilt_idx = 7


class TestPoseInitAndReset:
    def test_set_pose_init_scalar_and_per_tilt(self):
        w = _pose_wrapper()
        w.set_pose_init(z1=1.5)
        assert torch.allclose(w._pose_z1_init, torch.full((w.num_tilts,), 1.5))
        w.set_pose_init(
            z3=[0.0, 1.0, 2.0], dtheta=0.25, shifts=[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]
        )
        assert torch.allclose(w._pose_z3, torch.tensor([0.0, 1.0, 2.0]))
        assert torch.allclose(w._pose_dtheta, torch.full((w.num_tilts,), 0.25))
        assert torch.allclose(w._pose_shifts[2], torch.tensor([3.0, 4.0]))
        w.set_pose_init(shifts=[0.5, -0.5])  # a single (dy, dx) pair broadcasts to all tilts
        assert torch.allclose(w._pose_shifts[0], torch.tensor([0.5, -0.5]))
        with pytest.raises(ValueError, match="must broadcast"):
            w.set_pose_init(z1=[1.0, 2.0])

    def test_reset_returns_to_init_not_to_zero(self):
        w = _pose_wrapper()
        w.set_pose_init(shifts=[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]], z1=1.0, z3=-2.0, dtheta=0.5)
        w.set_defocus_init([0.0, 5.0, -5.0])
        with torch.no_grad():
            for p in (w._pose_shifts, w._pose_z1, w._pose_z3, w._pose_dtheta, w._defocus_offset_A):
                p.add_(9.0)
        w.reset()
        assert torch.allclose(w._pose_shifts, w._pose_shifts_init)
        assert torch.allclose(w._pose_z1, w._pose_z1_init)
        assert torch.allclose(w._pose_z3, w._pose_z3_init)
        assert torch.allclose(w._pose_dtheta, w._pose_dtheta_init)
        assert torch.allclose(w._defocus_offset_A, w._defocus_offset_init_A)

    def test_learn_flags_survive_reset(self):
        w = _pose_wrapper()
        w.set_learn_pose(shifts=True, z1=False, z3=True, dtheta=True)
        w.set_learn_defocus(True)
        w.reset()
        assert w.learn_pose_shifts and w.learn_pose_z3 and w.learn_pose_dtheta
        assert w.learn_defocus and not w.learn_pose_z1
        assert set(w.get_optimization_parameters()) == {"pose_shifts", "pose_angles", "defocus"}


class TestPosePlumbing:
    def test_payload_carries_pose_shifts(self):
        w = _pose_wrapper()
        w.set_pose_init(shifts=[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        n = int(np.prod(GPTS))
        batch = torch.tensor([0, n + 1, 2 * n + 2])
        payload, *_ = w.forward(batch, PAD)
        assert payload.shifts_A is not None
        expected = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(payload.shifts_A, expected)

    def test_window_dz_subtracts_the_pose_shift(self):
        w = _pose_wrapper()
        w.set_slab_window(True)
        batch = torch.arange(w.num_gpts)
        payload_a, *_ = w.forward(batch, PAD)
        dz_a = payload_a.window_dz_A
        shift = torch.tensor([0.6, -0.4])
        w.set_pose_init(shifts=shift.tolist())
        payload_b, *_ = w.forward(batch, PAD)
        dz_b = payload_b.window_dz_A
        assert dz_a is not None and dz_b is not None
        # reference: the same formula with the beam center displaced by -shift
        center = w._scan_center_px
        samp = w.obj_sampling
        pos = w.scan_positions_px[batch]
        y_c = (pos[:, 0] - center[0]) * float(samp[0]) - shift[0]
        x_c = (pos[:, 1] - center[1]) * float(samp[1]) - shift[1]
        r = w.rotations()[w.tilt_index_of(batch)]
        expected = -(r[:, 0, 1] * y_c + r[:, 0, 2] * x_c) / r[:, 0, 0]
        assert torch.allclose(dz_b, expected, atol=1e-5)
        assert not torch.allclose(dz_a, dz_b)  # the shift really moved the window


class TestLegacyRotationCenterShim:
    def _legacy_coords(self, w, batch):
        """The pre-pose coordinate path: (grid - center) * sampling - _rot_axis_offset_A."""
        center = w._scan_center_px
        samp = w.obj_sampling
        pos = w.scan_positions_px[batch]
        hroi, wroi = int(w.roi_shape[0]), int(w.roi_shape[1])
        r_ind = torch.fft.fftfreq(hroi, d=1 / hroi)
        c_ind = torch.fft.fftfreq(wroi, d=1 / wroi)
        rows = pos[:, 0][:, None, None] + r_ind[None, :, None]
        cols = pos[:, 1][:, None, None] + c_ind[None, None, :]
        rows = (rows.expand(-1, -1, wroi) - center[0]) * float(samp[0])
        cols = (cols.expand(-1, hroi, -1) - center[1]) * float(samp[1])
        return torch.stack([rows, cols], dim=-1)

    def test_loaded_legacy_buffer_reproduces_the_old_coordinates(self):
        w = _pose_wrapper()
        off = torch.tensor([0.99, -0.49])
        # emulate a cache written before the pose work: only the legacy buffer is set
        w._ensure_rot_offset_buffer()
        with torch.no_grad():
            w._rot_axis_offset_A.copy_(off)
        batch = torch.arange(w.num_gpts)
        legacy = self._legacy_coords(w, batch) - off  # what the old code emitted
        payload, *_ = w.forward(batch, PAD)  # first use -> folds the buffer
        assert payload.shifts_A is not None
        effective = payload.coords_yx_A - payload.shifts_A[:, None, None, :]
        assert torch.allclose(effective, legacy, atol=1e-6)
        # folded exactly once, into the baseline, and the legacy buffer is retired
        assert torch.allclose(w._pose_shifts_init, off.expand(w.num_tilts, 2))
        assert torch.equal(w._rot_axis_offset_A, torch.zeros(2))
        w.forward(batch, PAD)
        assert torch.allclose(w._pose_shifts_init, off.expand(w.num_tilts, 2))

    def test_legacy_buffer_reproduces_the_old_window_dz(self):
        w = _pose_wrapper()
        off = torch.tensor([0.99, -0.49])
        w.set_slab_window(True)
        w._ensure_rot_offset_buffer()
        with torch.no_grad():
            w._rot_axis_offset_A.copy_(off)
        batch = torch.arange(w.num_gpts)
        center, samp = w._scan_center_px, w.obj_sampling
        pos = w.scan_positions_px[batch]
        y_c = (pos[:, 0] - center[0]) * float(samp[0]) - off[0]
        x_c = (pos[:, 1] - center[1]) * float(samp[1]) - off[1]
        r = w.rotations()[w.tilt_index_of(batch)]
        legacy_dz = -(r[:, 0, 1] * y_c + r[:, 0, 2] * x_c) / r[:, 0, 0]
        payload, *_ = w.forward(batch, PAD)
        assert payload.window_dz_A is not None
        assert torch.allclose(payload.window_dz_A, legacy_dz, atol=1e-6)

    def test_deprecated_wrapper_writes_the_pose_baseline(self):
        w = _pose_wrapper()
        with pytest.deprecated_call():
            w.set_rotation_center_offset_A(0.99, -0.49)
        expected = torch.tensor([0.99, -0.49]).expand(w.num_tilts, 2)
        assert torch.allclose(w._pose_shifts_init, expected)
        assert torch.allclose(w._pose_shifts, expected)
        w.reset()
        assert torch.allclose(w._pose_shifts, expected)  # survives reset, as the hack did


class TestDefocusOffset:
    def test_defocus_does_not_move_the_object_query(self):
        w = _pose_wrapper()
        batch = torch.arange(w.num_gpts)
        payload_a, *_ = w.forward(batch, PAD)
        w.set_defocus_init([0.0, 10.0, -25.0])
        payload_b, *_ = w.forward(batch, PAD)
        assert torch.equal(payload_a.coords_yx_A, payload_b.coords_yx_A)
        assert payload_b.window_dz_A is None  # slab window off: object stack untouched
        assert w._last_window_dz_A is None

    def test_probe_dz_is_minus_the_offset(self):
        """Sign rule: effective defocus = probe defocus + offset, so the probe is propagated
        by -offset (propagating forward DECREASES defocus)."""
        w = _pose_wrapper()
        offsets = [0.0, 10.0, -25.0]
        w.set_defocus_init(offsets)
        n = int(np.prod(GPTS))
        batch = torch.tensor([0, n, 2 * n])
        w.forward(batch, PAD)
        assert w._last_probe_dz_A is not None
        assert torch.allclose(w._last_probe_dz_A, -torch.tensor(offsets))

    def test_probe_dz_adds_to_the_window_offset(self):
        w = _pose_wrapper()
        w.set_slab_window(True)
        batch = torch.arange(w.num_gpts)
        w.forward(batch, PAD)
        window_only = w._last_window_dz_A
        assert window_only is not None
        window_only = window_only.clone()
        w.set_defocus_init([0.0, 10.0, -25.0])
        payload, *_ = w.forward(batch, PAD)
        assert payload.window_dz_A is not None
        # the object still sees the pure window offset ...
        assert torch.allclose(payload.window_dz_A, window_only, atol=1e-6)
        # ... while the probe sees window - offset
        offs = w._defocus_offset_A[w.tilt_index_of(batch)]
        assert w._last_probe_dz_A is not None
        assert torch.allclose(w._last_probe_dz_A, window_only - offs, atol=1e-6)

    def test_no_probe_stash_when_defocus_is_off(self):
        """A zero, frozen defocus offset must not switch the probe pre-propagation on (it
        would cost two FFTs per batch for an identity operation)."""
        w = _pose_wrapper()
        w.forward(torch.arange(w.num_gpts), PAD)
        assert w._last_probe_dz_A is None
        w.set_learn_defocus(True)
        w.forward(torch.arange(w.num_gpts), PAD)
        assert w._last_probe_dz_A is not None  # learnable: on even while still zero

    def test_defocus_gradient_flows_to_the_parameter(self):
        w = _pose_wrapper()
        w.set_learn_defocus(True)
        w.forward(torch.arange(w.num_gpts), PAD)
        dz = w._last_probe_dz_A
        assert dz is not None and dz.requires_grad
        dz.sum().backward()
        assert w._defocus_offset_A.grad is not None
        assert torch.allclose(
            w._defocus_offset_A.grad, -torch.tensor([float(np.prod(GPTS))] * w.num_tilts)
        )


class TestPoseSerialization:
    def test_autoserialize_roundtrip_keeps_pose_state(self, tmp_path):
        w = _pose_wrapper()
        w.set_pose_init(shifts=[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]], z1=1.0, z3=-2.0, dtheta=0.5)
        w.set_defocus_init([0.0, 5.0, -5.0])
        w.set_learn_pose(shifts=True, z1=False, z3=True, dtheta=True)
        w.set_learn_defocus(True)
        w.reference_tilt_idx = 1
        path = tmp_path / "pose_dset.zip"
        w.save(path, mode="o")
        loaded = autoserialize_load(path)
        for name in (
            "_pose_shifts",
            "_pose_z1",
            "_pose_z3",
            "_pose_dtheta",
            "_defocus_offset_A",
            "_pose_shifts_init",
            "_pose_z1_init",
            "_pose_z3_init",
            "_pose_dtheta_init",
            "_defocus_offset_init_A",
        ):
            assert torch.allclose(getattr(loaded, name).detach(), getattr(w, name).detach()), name
        assert loaded.learn_pose_shifts and loaded.learn_pose_z3 and loaded.learn_pose_dtheta
        assert loaded.learn_defocus and not loaded.learn_pose_z1
        assert loaded.reference_tilt_idx == 1
        assert set(loaded.get_optimization_parameters()) == {
            "pose_shifts",
            "pose_angles",
            "defocus",
        }
        assert loaded._pose_shifts.requires_grad and not loaded._pose_z1.requires_grad

    def test_missing_slots_are_materialized_on_load(self, tmp_path):
        """A cache written before these slots existed must still load and behave (gotcha #23)."""
        w = _pose_wrapper()
        path = tmp_path / "legacy_dset.zip"
        w.save(path, mode="o")
        loaded = autoserialize_load(path)
        # emulate the pre-task-1/2 on-disk state
        del loaded._parameters["_defocus_offset_A"]
        for name in ("_pose_shifts_init", "_pose_dtheta_init", "_defocus_offset_init_A"):
            del loaded._buffers[name]
        for flag in PtychoTomoDatasetRaster._POSE_LEARN_FLAGS + (
            "_reference_tilt_idx",
            "_defocus_active",
        ):
            loaded.__dict__.pop(flag, None)
        loaded.implicit_object = True
        payload, *_ = loaded.forward(torch.tensor([0, 20]), PAD)
        assert payload.shifts_A is not None
        assert loaded.reference_tilt_idx == REF_TILT
        assert loaded.get_optimization_parameters() == {}
        assert loaded._defocus_offset_A.shape == (loaded.num_tilts,)
        loaded.reset()
