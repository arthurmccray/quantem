"""Tests for ``quantem.tomography.tomography_splat``: the Gaussian-splat reconstruction loop,
the prior hook and schedule, and the lite wrapper. All CPU and tiny."""

import numpy as np
import pytest
import torch

from quantem.core.ml.splat import render_projections, window_radius_for
from quantem.tomography.tomography_lite import TomographyLiteSplat
from quantem.tomography.tomography_splat import DensifyParams, PriorSchedule
from quantem.tomography.utils import pose_to_rotations

SAMPLING = 0.5
SIGMA = 0.5


@pytest.fixture(scope="module")
def five_atom_series():
    gt = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.88, 0.0, 0.0],
            [0.0, 2.88, 0.0],
            [1.44, 1.44, 2.04],
            [-1.44, 1.44, -2.04],
        ]
    )
    tilts = np.arange(-60.0, 61.0, 20.0)
    rot = pose_to_rotations(
        torch.tensor(tilts, dtype=torch.float32), torch.zeros(len(tilts)), torch.zeros(len(tilts))
    )
    imgs = render_projections(
        gt,
        torch.full((5,), 1.7e-3),
        torch.full((5,), SIGMA),
        rot,
        torch.zeros(len(tilts), 2),
        SAMPLING,
        (24, 24),
        window_radius_for(SIGMA, SAMPLING),
    ).numpy()
    return gt, tilts, imgs


class _CountingPrior:
    def __init__(self):
        self.calls = []

    def clean_estimate(self, x, t=None):
        self.calls.append(t)
        return x + 0.05  # a constant pull so the prior term is non-zero


def test_reconstruct_reduces_loss_moves_atom_and_calls_prior_on_schedule(five_atom_series):
    gt, tilts, imgs = five_atom_series
    init = gt.numpy().copy()
    init[3] += [0.15, -0.1, 0.1]
    rec = TomographyLiteSplat.from_dataset(
        imgs, tilts, SAMPLING, init=init, sigma_init=SIGMA, device="cpu", rng=0
    )
    prior = _CountingPrior()
    err0 = float(np.linalg.norm(rec.obj_model.positions[3] - gt[3].numpy()))
    rec.reconstruct(
        num_iter=4,
        learn_pose=False,
        prior=prior,
        prior_schedule=PriorSchedule(
            every=2, start=1, t=(0.3, 0.9), lam=0.1, lam_mode="relative", anneal_frac=1.0
        ),
    )
    err1 = float(np.linalg.norm(rec.obj_model.positions[3] - gt[3].numpy()))
    assert rec.epoch_losses[-1] < rec.epoch_losses[0]
    assert err1 < err0
    assert prior.calls == [pytest.approx(0.3 + 0.6 * 1 / 4), pytest.approx(0.3 + 0.6 * 3 / 4)]
    assert rec.obj_model.soft_constraint_loss["prior"] > 0
    assert rec.prior_history[-1]["lam"] > 0 and rec.dset.optimizer is None  # learn_pose=False
    assert len(rec.num_atoms_history) == 4


def test_schedule_tuple_anneals_then_holds_and_sigma_uses_prior_mapping():
    sched = PriorSchedule(t=(0.2, 0.8), lam=(1.0, 0.5), anneal_frac=0.5)
    assert sched.at(0, 10) == (0.2, 1.0)
    assert sched.at(5, 10) == (pytest.approx(0.8), pytest.approx(0.5))
    assert sched.at(9, 10) == (pytest.approx(0.8), pytest.approx(0.5))

    class P:
        def t_of_sigma(self, s):
            return 1.0 - s / 2.0

    assert PriorSchedule(sigma=0.5, lam=1.0).at(0, 10, P()) == (0.75, 1.0)


def test_lite_backproject_init_and_densify_prune(five_atom_series):
    gt, tilts, imgs = five_atom_series
    rec = TomographyLiteSplat.from_dataset(
        imgs,
        tilts,
        SAMPLING,
        init="backproject",
        sigma_init=SIGMA,
        device="cpu",
        rng=0,
        min_distance=2.0,
        threshold_fraction=0.99,
    )
    assert rec.obj_model.num_atoms >= 1
    # add a spurious weak atom far from everything, then let densification prune it
    rec.obj_model.add_atoms(np.array([[4.0, -4.0, 4.0]]), amplitude=1e-7)
    n0 = rec.obj_model.num_atoms
    rec.reconstruct(
        num_iter=2,
        learn_pose=True,
        densify=DensifyParams(every=1, start=0, prune_amplitude_frac=0.5),
    )
    assert rec.obj_model.num_atoms == n0 - 1
    assert rec.dset.optimizer is not None and rec.obj_model.optimizer is not None
