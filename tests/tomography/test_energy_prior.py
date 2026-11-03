"""Tests for ``quantem.tomography.energy_prior``: the MACE energy prior (energies, forces, neighbour list,
FIRE clean estimate) on a 13-atom gold cuboctahedron, CPU, fp32. Needs the ``mace`` package and the project's
``mace-mp-0`` medium model file; otherwise the module is skipped."""

import os

import numpy as np
import pytest
import torch

mace = pytest.importorskip("mace")
from quantem.tomography.energy_prior import EnergyPrior  # noqa: E402

MODEL = os.path.expandvars("$SCRATCH/splat_tomo/models/mace/2023-12-03-mace-128-L1_epoch-199.model")
if not os.path.exists(MODEL):
    pytest.skip(f"MACE model file not found: {MODEL}", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def _few_threads():
    # the MACE CPU call takes 0.2 s with 4 threads and >5 s with 244 oversubscribed ones (login node)
    n = torch.get_num_threads()
    torch.set_num_threads(min(4, n))
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def prior():
    return EnergyPrior(MODEL, device="cpu", dtype=torch.float32, fire_steps=20, fire_fmax=0.02)


def _aligned_rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    """RMSD after the best rigid alignment (Kabsch): the energy is invariant to rigid motion, so FIRE never
    removes the net translation / rotation of a random jitter."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = torch.linalg.svd(a.T @ b)
    d = torch.diag(torch.tensor([1.0, 1.0, float(torch.sign(torch.det(u @ vt)))]))
    return float(((a @ (u @ d @ vt)) - b).norm(dim=-1).pow(2).mean().sqrt())


@pytest.fixture(scope="module")
def cluster13():
    from ase.cluster import Octahedron

    atoms = Octahedron("Au", 3, cutoff=1)  # 13-atom cuboctahedron
    assert len(atoms) == 13
    return atoms


@pytest.fixture(scope="module")
def relaxed13(prior, cluster13):
    x0 = torch.tensor(cluster13.positions, dtype=torch.float32)
    x, hist = prior.relax(x0, steps=300, fmax=0.02)
    assert hist["converged"], f"FIRE did not reach fmax 0.02 in 300 steps (fmax {hist['fmax'][-1]:.3f})"
    return x


def test_energy_forces_and_edges_match_mace_calculator(prior, cluster13):
    from mace.calculators import MACECalculator
    from matscipy.neighbours import neighbour_list

    atoms = cluster13.copy()
    atoms.positions += np.random.default_rng(0).normal(0.0, 0.1, atoms.positions.shape)  # off the minimum
    atoms.pbc = False
    atoms.calc = MACECalculator(model_paths=MODEL, device="cpu", default_dtype="float32")
    e_ref = float(atoms.get_potential_energy())
    f_ref = np.asarray(atoms.get_forces(), np.float64)
    pe_ref = np.asarray(atoms.get_potential_energies(), np.float64)

    x = torch.tensor(atoms.positions, dtype=torch.float32)
    e, pe, f = prior.energy_and_forces(x)
    n = len(atoms)
    assert abs(float(e) - e_ref) / n <= 1e-4, f"total energy {float(e)} vs ASE {e_ref}"
    assert np.abs(pe.numpy() - pe_ref).max() <= 1e-4
    assert np.abs(f.numpy() - f_ref).max() <= 1e-3
    assert np.abs(f_ref).max() > 0.5, "the jittered cluster must carry real forces for the check to mean anything"
    # the torch neighbour list is the matscipy one (same cutoff, no self edges)
    big = 10.0 * prior.cutoff
    i, j = neighbour_list("ij", positions=atoms.positions + big / 2, cell=np.eye(3) * big, pbc=np.zeros(3, bool),
                          cutoff=prior.cutoff)
    ref = set(zip(i.tolist(), j.tolist()))
    got = set(zip(*prior.edges(x, force=True).tolist()))
    assert got == ref
    # the forward-only path gives the same energies as the forward+backward one
    e2, pe2 = prior.energy(x)
    assert float(e2) == float(e) and torch.equal(pe2, pe)


def test_forces_are_minus_gradient_of_energy(prior, cluster13):
    x = torch.tensor(cluster13.positions, dtype=torch.float32)
    x[0, 0] += 0.2  # push one atom off its site so dE/dx is of order 1 eV/A
    f = prior.forces(x)
    h = 5e-3
    xp, xm = x.clone(), x.clone()
    xp[0, 0] += h
    xm[0, 0] -= h
    fd = -(float(prior.energy(xp)[0]) - float(prior.energy(xm)[0])) / (2 * h)
    assert abs(float(f[0, 0])) > 0.3, "the displaced atom must carry a real force"
    assert abs(float(f[0, 0]) - fd) <= 1e-2 * max(1.0, abs(fd)), f"autograd {float(f[0, 0])} vs FD {fd}"


def test_neighbour_list_is_reused_every_k_calls_and_rebuilt_when_atoms_change(prior, cluster13):
    x = torch.tensor(cluster13.positions, dtype=torch.float32)
    prior.nl_every = 2
    try:
        prior.n_calls = 0
        prior.energy(x)
        first = prior._edge_cache[1]
        prior.energy(x + 0.01)  # n_calls = 1 -> reused
        assert prior._edge_cache[1] is first
        prior.energy(x[:12])  # atom count changed -> rebuilt although it is not a rebuild call
        assert prior._edge_cache[0] == 12 and prior._edge_cache[1] is not first
    finally:
        prior.nl_every = 1


def test_fire_clean_estimate_lowers_energy_and_rmsd_to_the_relaxed_cluster(prior, relaxed13):
    g = torch.Generator().manual_seed(0)
    x = relaxed13 + 0.1 * torch.randn(relaxed13.shape, generator=g)
    e0 = float(prior.energy(x)[0])
    rmsd0 = _aligned_rmsd(x, relaxed13)
    x_hat = prior.clean_estimate(x, t=0.5)  # t is ignored; 20 FIRE steps, 0.1 A per atom per step
    e1 = float(prior.energy(x_hat)[0])
    rmsd1 = _aligned_rmsd(x_hat, relaxed13)
    assert e1 < e0 - 1e-3, f"FIRE must lower the energy: {e0} -> {e1}"
    # measured 2026-10-29 (seed 0): 0.135 -> 0.086 A in 20 steps (0.096 in 10); a broken FIRE leaves the ratio at 1
    assert rmsd1 < 0.75 * rmsd0, f"FIRE must move toward the relaxed cluster: RMSD {rmsd0:.3f} -> {rmsd1:.3f}"
    assert float((x_hat - x).norm(dim=-1).max()) <= prior.fire_steps * prior.fire_max_step + 1e-6
    assert x_hat.dtype == x.dtype and x_hat.shape == x.shape
