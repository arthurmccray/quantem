"""Tests for ``quantem.tomography.candidate_sites``: the lattice-continuation candidate generator on small ase
gold particles (CPU; one GPU parity test gated by ``requires_gpu``)."""

import numpy as np
import pytest
import torch

from quantem.tomography.candidate_sites import (
    CandidateParams,
    empty_interior_sites,
    generate_candidate_sites,
)

from .conftest import requires_gpu

A0 = 4.078  # Angstrom, gold
D1 = A0 / np.sqrt(2.0)  # ideal first-shell distance


@pytest.fixture(scope="module", autouse=True)
def _few_threads():
    # the brute-force distance passes are tiny here; 4 threads avoid the login node's oversubscription
    n = torch.get_num_threads()
    torch.set_num_threads(min(4, n))
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def octahedron() -> np.ndarray:
    from ase.cluster import Octahedron

    return Octahedron(
        "Au", 9, cutoff=3, latticeconstant=A0
    ).positions  # 405 atoms, 115 with 12 neighbours


def _coordination(x: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return (d < 1.2 * D1).sum(1)


def _remove(x: np.ndarray, frac: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Drop a fixed random ``frac`` of the atoms (bulk and surface mixed); returns (kept, removed)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), int(round(frac * len(x))), replace=False)
    keep = np.ones(len(x), bool)
    keep[idx] = False
    return x[keep], x[idx]


def _recovered(removed: np.ndarray, cand: np.ndarray, tol: float) -> int:
    return int((np.linalg.norm(removed[:, None] - cand[None], axis=-1).min(1) < tol).sum())


def test_recall_of_removed_sites_on_octahedron(octahedron):
    x, removed = _remove(octahedron, 0.05)
    cn = _coordination(octahedron)[
        np.linalg.norm(octahedron[:, None] - removed[None], axis=-1).argmin(0)
    ]
    assert 0 < (cn == 12).sum() < len(removed), "the draw must remove bulk and surface atoms"
    c = generate_candidate_sites(x)
    assert isinstance(c.positions, np.ndarray), "numpy in, numpy out"
    assert abs(c.d1 - D1) < 1e-6, f"measured d1 {c.d1} vs ideal {D1}"
    assert c.c_max == 12, f"measured full coordination {c.c_max}"
    assert c.n_under == int((c.atom_coordination < 12).sum()) > 0
    n_rec = _recovered(removed, c.positions, 0.5)
    assert n_rec == len(removed), (
        f"{n_rec} of {len(removed)} removed sites within 0.5 A of a candidate"
    )
    assert len(c) <= 0.9 * len(x), f"{len(c)} candidates for {len(x)} atoms (bound 0.9 N)"
    assert len(c) >= len(removed)


def test_jitter_recall_and_cluster_invariants(octahedron):
    rng = np.random.default_rng(1)
    jittered = octahedron + rng.normal(0.0, 0.15, octahedron.shape)
    x, removed = _remove(jittered, 0.05)
    c = generate_candidate_sites(x, params=CandidateParams(min_support=3))
    n_rec = _recovered(removed, c.positions, 0.7)
    assert n_rec == len(removed), f"{n_rec} of {len(removed)} jittered removed sites within 0.7 A"
    d_atom = np.linalg.norm(c.positions[:, None] - x[None], axis=-1).min(1)
    assert d_atom.min() > 0.8 * c.d1, (
        f"a candidate is {d_atom.min():.3f} A from an atom (bar {0.8 * c.d1:.3f})"
    )
    d_cc = np.linalg.norm(c.positions[:, None] - c.positions[None], axis=-1)
    np.fill_diagonal(d_cc, np.inf)
    assert d_cc.min() >= 1.0, f"two candidates {d_cc.min():.3f} A apart after clustering"
    assert c.support.min() >= 3, f"support below min_support: {c.support.min()}"
    assert c.n_clusters > len(c), (
        "the jittered structure must produce clusters that the support cut removes"
    )
    expect = (np.linalg.norm(c.positions[:, None] - x[None], axis=-1) < c.r_nn).sum(1)
    assert np.array_equal(c.coordination, expect)


def test_d1_override_under_heavy_jitter(octahedron):
    """At 0.3 A jitter per coordinate the measured first-shell distance is biased low (the minimum over 12 noisy
    neighbour distances) and the measured full coordination collapses; ``CandidateParams.d1`` restores both and
    the recall of the removed sites, and on a clean structure it is a pure pass-through."""
    x_clean, removed = _remove(octahedron, 0.05, seed=2)
    c_meas = generate_candidate_sites(x_clean)
    c_given = generate_candidate_sites(x_clean, d1=D1)
    assert c_given.d1 == D1 and len(c_given) == len(c_meas), (
        "override on a clean structure changed the result"
    )
    assert np.allclose(
        np.sort(c_given.positions, axis=0), np.sort(c_meas.positions, axis=0), atol=1e-6
    )
    rng = np.random.default_rng(2)
    x = x_clean + rng.normal(0.0, 0.3, x_clean.shape)
    c_meas = generate_candidate_sites(x)
    assert c_meas.d1 < 0.9 * D1, (
        f"measured d1 {c_meas.d1:.3f} A is not biased low under 0.3 A jitter"
    )
    assert c_meas.c_max < 12, f"measured c_max {c_meas.c_max} did not collapse under 0.3 A jitter"
    c_given = generate_candidate_sites(x, d1=D1)
    assert c_given.c_max >= 12, f"c_max {c_given.c_max} with the true d1"
    n_rec = _recovered(removed, c_given.positions, 0.7)
    assert n_rec >= 0.85 * len(removed), (
        f"{n_rec} of {len(removed)} removed sites within 0.7 A with the true d1"
    )


def test_support_cap_keeps_highest_support_first(octahedron):
    x, removed = _remove(octahedron, 0.05)
    c = generate_candidate_sites(x, max_frac=0.1)
    assert len(c) == int(0.1 * c.n_under) < 0.9 * len(x) / 2, f"{len(c)} candidates under the cap"
    assert np.all(np.diff(c.support) <= 0), "candidates must come highest support first"
    n_rec = _recovered(removed, c.positions, 0.5)
    assert n_rec == len(removed), (
        f"{n_rec} of {len(removed)} removed sites survive the cap (they rank highest)"
    )


def test_recall_on_decahedron_with_twins():
    from ase.cluster import Decahedron

    deca = Decahedron("Au", 4, 4, 1, latticeconstant=A0).positions  # 389 atoms, 5 twin boundaries
    x, removed = _remove(deca, 0.05)
    c = generate_candidate_sites(x)
    assert c.c_max == 12, f"decahedron: full coordination measured as {c.c_max}"
    n_rec = _recovered(removed, c.positions, 0.7)
    assert len(removed) == 19
    assert n_rec >= 18, (
        f"decahedron: {n_rec} of {len(removed)} removed sites within 0.7 A (measured 19 of 19)"
    )
    assert len(c) <= 0.9 * len(x), f"decahedron: {len(c)} candidates for {len(x)} atoms"


def test_from_residual_merge(octahedron):
    cn = _coordination(octahedron)
    site = octahedron[np.where(cn == 12)[0][0]]
    x = np.delete(octahedron, np.where(cn == 12)[0][0], axis=0)
    base = generate_candidate_sites(x)
    far = np.array([[80.0, 80.0, 80.0]])
    c = generate_candidate_sites(x, extra=np.concatenate([site[None] + 0.3, far]))
    assert len(c) == len(base) + 1, (
        "the near extra must join its cluster, the far one become a candidate"
    )
    near = np.linalg.norm(c.positions - site, axis=-1) < 0.5
    assert near.sum() == 1 and c.from_residual[near].all() and c.support[near] > 0
    assert not c.from_residual[~near][:-1].any(), "no other cluster gets the flag"
    assert c.from_residual[-1] and c.support[-1] == 0 and c.coordination[-1] == 0
    assert np.allclose(c.positions[-1], far[0])


def test_empty_interior_sites(octahedron):
    cn = _coordination(octahedron)
    bulk = np.where(cn == 12)[0][[0, 40, 80]]
    x = np.delete(octahedron, bulk, axis=0)
    e = empty_interior_sites(x, min_coordination=10)
    assert len(e) == 3, f"{len(e)} empty interior sites for 3 removed bulk atoms"
    d = np.linalg.norm(octahedron[bulk][:, None] - e.positions[None], axis=-1).min(1)
    assert d.max() < 0.5, f"empty interior sites {d} A from the removed bulk atoms"
    assert e.coordination.min() >= 10 and e.support.min() >= 2
    assert isinstance(e.positions, np.ndarray)


@requires_gpu
def test_gpu_matches_cpu(octahedron):
    rng = np.random.default_rng(2)
    x, _ = _remove(octahedron + rng.normal(0.0, 0.1, octahedron.shape), 0.05)
    extra = x[:2] + 2.2  # two residual-style extras
    ref = generate_candidate_sites(
        torch.tensor(x, dtype=torch.float32), torch.tensor(extra, dtype=torch.float32)
    )
    gpu = generate_candidate_sites(
        torch.tensor(x, dtype=torch.float32, device="cuda:0"),
        torch.tensor(extra, dtype=torch.float32, device="cuda:0"),
    )
    assert gpu.positions.device.type == "cuda"
    assert len(gpu) == len(ref) and gpu.c_max == ref.c_max and abs(gpu.d1 - ref.d1) < 1e-5
    d = torch.cdist(gpu.positions.cpu().double(), ref.positions.double())
    nearest = d.argmin(dim=1)
    assert torch.equal(nearest.sort().values, torch.arange(len(ref))), "one-to-one candidate match"
    assert d.min(dim=1).values.max() < 1e-4, (
        f"max candidate offset GPU vs CPU {d.min(dim=1).values.max():.2e} A"
    )
    assert torch.equal(gpu.support.cpu(), ref.support[nearest])
    assert torch.equal(gpu.coordination.cpu(), ref.coordination[nearest])
    assert torch.equal(gpu.from_residual.cpu(), ref.from_residual[nearest])
    assert gpu.from_residual.sum() >= 1
