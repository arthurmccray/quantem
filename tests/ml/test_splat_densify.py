"""Tests for ``quantem.core.ml.splat.densify``: optimizer-state remapping and merging."""

import numpy as np
import pytest
import torch

from quantem.core.ml.optimizer_mixin import OptimizerParams
from quantem.core.ml.splat.densify import merge_close_atoms
from quantem.tomography.object_models import ObjectGaussianSplat


def _obj():
    pos = np.array([[0.0, 0.0, 0.0], [2.9, 0.0, 0.0], [0.0, 2.9, 0.0], [2.9, 2.9, 0.0]])
    return ObjectGaussianSplat.from_positions(pos, 1.0, 0.4, fov_A=10.0, view_sampling=0.5)


def test_prune_and_add_remap_adam_state():
    obj = _obj()
    obj.set_optimizer(OptimizerParams.Adam(lr=1e-2))
    # one step so that Adam state exists, then stamp recognisable rows into exp_avg
    obj.forward(torch.eye(3)[None], torch.zeros(1, 2), (16, 16), 0.5).sum().backward()
    obj.step_optimizer()
    state = obj.optimizer.state[obj._positions]
    state["exp_avg"] = torch.arange(12.0).reshape(4, 3)
    removed = obj.prune(torch.tensor([True, False, True, True]))
    assert removed == 1 and obj.num_atoms == 3
    new_state = obj.optimizer.state[obj._positions]
    assert torch.equal(
        new_state["exp_avg"], torch.tensor([[0.0, 1, 2], [6.0, 7, 8], [9.0, 10, 11]])
    )
    added = obj.add_atoms(np.array([[1.0, 1.0, 1.0]]))
    assert added == 1 and obj.num_atoms == 4
    new_state = obj.optimizer.state[obj._positions]
    assert new_state["exp_avg"].shape == (4, 3) and torch.all(new_state["exp_avg"][3] == 0)
    assert obj.optimizer.param_groups[0]["params"][0] is obj._positions
    # the optimizer still steps the rebuilt parameters
    obj.zero_optimizer_grad()
    obj.forward(torch.eye(3)[None], torch.zeros(1, 2), (16, 16), 0.5).sum().backward()
    before = obj._positions.detach().clone()
    obj.step_optimizer()
    assert not torch.equal(before, obj._positions.detach())


def test_merge_close_atoms_sums_amplitude_and_weights_position():
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [5.0, 0.0, 0.0]])
    amp = torch.tensor([3.0, 1.0, 2.0])
    sig = torch.tensor([0.4, 0.5, 0.6])
    keep, p, a, s = merge_close_atoms(pos, amp, sig, min_distance=1.0)
    assert keep.tolist() == [0, 2]
    assert a.tolist() == [4.0, 2.0]
    assert torch.allclose(p[0], torch.tensor([0.1, 0.0, 0.0]))
    assert torch.allclose(s, torch.tensor([0.4, 0.6]))


def test_merge_max_mode_keeps_the_brighter_amplitude():
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    keep, p, a, s = merge_close_atoms(pos, torch.tensor([3.0, 1.0]), torch.tensor([0.4, 0.5]), 1.0, "max")
    assert keep.tolist() == [0] and a.tolist() == [3.0]
    with pytest.raises(ValueError):
        merge_close_atoms(pos, torch.tensor([3.0, 1.0]), torch.tensor([0.4, 0.5]), 1.0, "mean")


def test_atom_support_is_positive_for_needed_atoms_and_negative_for_spurious_ones():
    """Two true atoms render the target; the model adds a third, spurious atom far from them. The true
    atoms' support equals the chi^2 of their own stamps (sum g^2 / var); the spurious atom's support is
    minus that (removing it lowers the loss); pixel_variance follows Var(v) = v / k with the count floor."""
    from quantem.core.ml.splat.densify import atom_support, pixel_variance
    from quantem.core.ml.splat.renderer import render_projections, window_radius_for
    from quantem.tomography.utils import pose_to_rotations

    tilts = torch.tensor([-30.0, 0.0, 30.0])
    rot = pose_to_rotations(tilts, torch.zeros(3), torch.zeros(3))
    shifts = torch.zeros(3, 2)
    s, sig = 0.5, torch.full((3,), 0.5)
    pos = torch.tensor([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    amp = torch.tensor([1.0, 1.0, 0.5])
    r = window_radius_for(0.5, s)
    target = render_projections(pos[:2], amp[:2], sig[:2], rot, shifts, s, (32, 32), r)
    pred = render_projections(pos, amp, sig, rot, shifts, s, (32, 32), r)
    scale, dose = 2.0, 1e4
    var = pixel_variance(pred, scale, dose, s, min_counts=1.0)
    k = scale * dose * s * s
    assert torch.allclose(var, pred / k + 1.0 / k**2)
    # round 3 options: the measured image where it exceeds the model (the vacuum atom's pixels keep the model), and the
    # relative model-error floor
    var_max = pixel_variance(pred, scale, dose, s, min_counts=1.0, target=target)
    assert torch.allclose(var_max, torch.maximum(pred, target) / k + 1.0 / k**2)
    assert torch.equal(var_max, var)  # target <= pred everywhere here: nothing changes
    under = pixel_variance(target, scale, dose, s, min_counts=1.0, target=pred)  # model = target lacks the third atom
    assert bool((under > pixel_variance(target, scale, dose, s, min_counts=1.0)).any())
    assert torch.allclose(pixel_variance(pred, scale, dose, s, min_counts=1.0, model_error_frac=0.1), var + (0.1 * pred) ** 2)
    # smoothed target: a single bright noise pixel no longer raises the variance to its own level
    spike = torch.zeros(1, 32, 32)
    spike[0, 16, 16] = 1.0
    flat = torch.zeros(1, 32, 32)
    v_raw = pixel_variance(flat, scale, dose, s, min_counts=1.0, target=spike)
    v_s = pixel_variance(flat, scale, dose, s, min_counts=1.0, target=spike, target_blur_px=1.5)
    assert float(v_s.max()) < 0.2 * float(v_raw.max()) and torch.allclose(v_s.sum(), v_raw.sum(), rtol=0.05)
    sup = atom_support(pos, amp, sig, rot, shifts, s, (32, 32), r, target - pred, var, tilt_chunk=2)
    own = [(render_projections(pos[i : i + 1], amp[i : i + 1], sig[i : i + 1], rot, shifts, s, (32, 32), r) ** 2 / var).sum() for i in range(3)]
    assert torch.allclose(sup[:2], torch.stack(own[:2]), rtol=1e-4)
    assert torch.allclose(sup[2], -own[2], rtol=1e-4)
