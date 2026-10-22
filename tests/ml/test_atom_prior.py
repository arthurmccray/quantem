"""Tests for ``quantem.core.ml.splat.atom_prior.AtomPrior`` with stub velocity models (the real
graph builders come from tricor, so the tests are skipped where it is not installed)."""

import pytest
import torch

tricor = pytest.importorskip("tricor")

from quantem.core.ml.splat.atom_prior import AtomPrior  # noqa: E402


def _positions():
    return torch.tensor(
        [[0.0, 0.0, 0.0], [2.88, 0.0, 0.0], [0.0, 2.88, 0.0], [1.44, 1.44, 2.04]],
        dtype=torch.float64,
    )


def test_zero_velocity_returns_input_and_constant_velocity_adds_step():
    pos = _positions()
    zero = AtomPrior(
        lambda z, ei, ea, t: torch.zeros(z.shape[0], 3), sigma_max=1.5, cutoff=5.0, device="cpu"
    )
    out = zero.clean_estimate(pos, t=0.7, noise_scale=0.0)
    assert out.dtype == pos.dtype and torch.allclose(out, pos)
    const = AtomPrior(
        lambda z, ei, ea, t: torch.full((z.shape[0], 3), 0.5),
        sigma_max=1.5,
        cutoff=5.0,
        device="cpu",
    )
    out = const.clean_estimate(pos, t=0.7, noise_scale=0.0)
    assert torch.allclose(out, pos + 0.3 * 0.5)
    assert const.t_of_sigma(0.3) == pytest.approx(0.8) and const.sigma_of_t(0.8) == pytest.approx(
        0.3
    )
    assert const.t_of_sigma(5.0) == 0.02  # clamped


def test_graph_is_free_space_and_velocity_sees_neighbours():
    """A velocity model that returns the mean edge vector per atom: the padded box must give the
    same graph as free space (two atoms 2.88 A apart -> one neighbour each, opposite vectors), and
    the 2-atom case must trigger the 15 A box clamp without error."""

    def mean_edge(z, ei, ea, t):
        out = torch.zeros(z.shape[0], 3)
        out.index_add_(0, ei[0], ea[:, :3])
        return out

    prior = AtomPrior(mean_edge, sigma_max=1.0, cutoff=5.0, device="cpu")
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.88, 0.0, 0.0]])
    v = prior.velocity(pos, t=0.5)
    assert torch.allclose(v[0], torch.tensor([2.88, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(v[1], -v[0], atol=1e-5)
    # test-time rotation averaging leaves a rotation-equivariant model unchanged
    x_hat = prior.clean_estimate(
        pos, t=0.5, noise_scale=0.0, n_rot=3, generator=torch.Generator().manual_seed(0)
    )
    assert torch.allclose(x_hat, pos + 0.5 * v, atol=1e-4)
