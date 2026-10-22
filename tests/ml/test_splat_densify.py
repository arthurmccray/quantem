"""Tests for ``quantem.core.ml.splat.densify``: optimizer-state remapping and merging."""

import numpy as np
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
