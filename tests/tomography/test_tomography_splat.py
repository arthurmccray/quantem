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


def test_schedule_tuple_anneals_then_holds_and_sigma_uses_prior_mapping(five_atom_series):
    sched = PriorSchedule(t=(0.2, 0.8), lam=(1.0, 0.5), anneal_frac=0.5)
    assert sched.at(0, 10) == (0.2, 1.0)
    assert sched.at(5, 10) == (pytest.approx(0.8), pytest.approx(0.5))
    assert sched.at(9, 10) == (pytest.approx(0.8), pytest.approx(0.5))

    class P:
        def t_of_sigma(self, s):
            return 1.0 - s / 2.0

    assert PriorSchedule(sigma=0.5, lam=1.0).at(0, 10, P()) == (0.75, 1.0)

    # late fade (round 5): lam held at its first value until lam_fade_start, then linear to the end value at the
    # end of the run; t keeps annealing over anneal_frac (unchanged)
    late = PriorSchedule(t=(0.2, 0.8), lam=(1.0, 0.0), anneal_frac=0.5, lam_fade_start=2.0 / 3.0)
    assert late.at(0, 30) == (0.2, 1.0)
    assert late.at(20, 30) == (pytest.approx(0.8), 1.0)
    assert late.at(25, 30) == (pytest.approx(0.8), pytest.approx(0.5))
    assert late.at(30, 30) == (pytest.approx(0.8), 0.0)
    assert PriorSchedule(lam=2.0, lam_fade_start=0.5).at(9, 10) == (0.5, 2.0)  # a scalar lam is not faded

    # sigma from the measured error (round 6 step 4): sigma_mode="pull" sets sigma_t = clamp(c * previous call's
    # per-coordinate RMS pull, min(sigma), max(sigma)); the first call (no previous pull) uses max(sigma); the clock
    # mode ignores the pull. A scalar sigma has no clamp range and is refused.
    pull_sched = PriorSchedule(sigma=(0.4, 0.15), lam=1.0, sigma_mode="pull", sigma_pull_scale=2.0)
    assert pull_sched.sigma_at(0, 10) == 0.4 and pull_sched.at(0, 10, P()) == (pytest.approx(0.8), 1.0)
    assert pull_sched.sigma_at(5, 10, last_pull_rms_coord=0.12) == pytest.approx(0.24)  # inside the range: c * e
    assert pull_sched.sigma_at(5, 10, last_pull_rms_coord=0.02) == 0.15  # clamped at the floor
    assert pull_sched.sigma_at(5, 10, last_pull_rms_coord=0.5) == 0.4  # clamped at the ceiling
    assert pull_sched.at(5, 10, P(), last_pull_rms_coord=0.12) == (pytest.approx(1.0 - 0.12), 1.0)
    clock = PriorSchedule(sigma=(0.4, 0.15), lam=1.0, anneal_frac=0.5)
    assert clock.sigma_at(5, 10, last_pull_rms_coord=0.02) == pytest.approx(0.15) and clock.sigma_at(0, 10, 0.02) == 0.4
    with pytest.raises(ValueError):
        PriorSchedule(sigma=0.3, sigma_mode="pull").sigma_at(1, 10, 0.1)

    # in a run: the second call's sigma_t equals clamp(c * pull_rms_coord of the first call) and the history records
    # sigma and pull_rms_coord at every call (a fake prior with a known constant pull of 0.05 A per coordinate)
    gt, tilts, imgs = five_atom_series

    class _ShiftPrior(_CountingPrior):  # x_hat - x = 0.05 on every coordinate: pull norm 0.05 * sqrt(3), rms/coord 0.05
        sigma_max, t_range = 1.0, (0.0, 1.0)

        def t_of_sigma(self, s):
            return 1.0 - s / self.sigma_max

        def clean_estimate(self, x, t=None, noise_scale=1.0):
            self.calls.append(t)
            return x + 0.05

    for c, s_min, expected in ((2.0, 0.05, 0.10), (1.0, 0.15, 0.15)):  # (inside the range, clamped at the floor)
        rec = TomographyLiteSplat.from_dataset(
            imgs, tilts, SAMPLING, init=gt.numpy() + 0.1, sigma_init=SIGMA, device="cpu", rng=0
        )
        prior = _ShiftPrior()
        rec.reconstruct(
            num_iter=3, learn_pose=False, prior=prior,
            prior_schedule=PriorSchedule(every=1, start=0, sigma=(0.4, s_min), lam=0.1, lam_mode="absolute",
                                         sigma_mode="pull", sigma_pull_scale=c),
        )
        h = rec.prior_history
        assert [e["sigma"] for e in h] == [0.4, pytest.approx(expected), pytest.approx(expected)], f"c={c} s_min={s_min}"
        assert prior.calls == [pytest.approx(1.0 - 0.4), pytest.approx(1.0 - expected), pytest.approx(1.0 - expected)]
        assert [e["pull_rms_coord"] for e in h] == [pytest.approx(0.05, rel=1e-4)] * 3, f"c={c}"
        assert h[0]["pull_mean"] == pytest.approx(0.05 * 3**0.5, rel=1e-4)
    # the clock mode records sigma and pull_rms_coord too and keeps its schedule (0.4 -> 0.15 over anneal_frac)
    rec = TomographyLiteSplat.from_dataset(imgs, tilts, SAMPLING, init=gt.numpy() + 0.1, sigma_init=SIGMA, device="cpu", rng=0)
    rec.reconstruct(num_iter=3, learn_pose=False, prior=_ShiftPrior(),
                    prior_schedule=PriorSchedule(every=1, start=0, sigma=(0.4, 0.15), lam=0.1, lam_mode="absolute", anneal_frac=1.0))
    assert [e["sigma"] for e in rec.prior_history] == [0.4, pytest.approx(0.4 - 0.25 / 3), pytest.approx(0.4 - 0.5 / 3)]
    assert rec.prior_history[1]["pull_rms_coord"] == pytest.approx(0.05, rel=1e-4)


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


def test_densify_splits_a_fat_atom_between_two_sites(five_atom_series):
    gt, tilts, imgs = five_atom_series
    # atoms 0 and 1 (2.88 A apart along x) collapsed onto one double-amplitude Gaussian at their midpoint
    init = np.concatenate([np.array([[1.44, 0.0, 0.0]]), gt[2:].numpy()])
    rec = TomographyLiteSplat.from_dataset(
        imgs, tilts, SAMPLING, init=init, sigma_init=SIGMA, device="cpu", rng=0
    )
    with torch.no_grad():
        rec.obj_model._log_amplitudes[0] += float(np.log(2.0))
    rec.reconstruct(
        num_iter=1,
        learn_pose=False,
        densify=DensifyParams(
            every=1,
            start=0,
            prune_amplitude_frac=0.0,
            merge_distance=1.5,
            add_from_residual=True,
            add_min_distance=2.0,
            split_from_residual=True,
            voxel_size=SAMPLING,
        ),
    )
    pos = np.asarray(rec.obj_model.positions)
    assert rec.obj_model.num_atoms == 5, (
        f"expected the fat atom split into two, got {rec.obj_model.num_atoms}"
    )
    for site in (gt[0].numpy(), gt[1].numpy()):
        d = float(np.linalg.norm(pos - site, axis=1).min())
        assert d < 0.75, f"no atom near site {site.tolist()} after the split (nearest {d:.2f} A)"
    d_mid = float(np.linalg.norm(pos - np.array([1.44, 0.0, 0.0]), axis=1).min())
    assert d_mid > 0.7, "the midpoint atom survived"


def test_densify_add_cap_and_monotone_prune_reference(five_atom_series):
    gt, tilts, imgs = five_atom_series
    # start with 2 of the 5 atoms: the residual offers 3 peaks, the cap allows one per pass
    rec = TomographyLiteSplat.from_dataset(
        imgs, tilts, SAMPLING, init=gt[:2].numpy(), sigma_init=SIGMA, device="cpu", rng=0
    )
    rec.reconstruct(
        num_iter=1,
        learn_pose=False,
        densify=DensifyParams(
            every=1,
            start=0,
            prune_amplitude_frac=0.0,
            add_from_residual=True,
            max_add_frac=0.5,
            voxel_size=SAMPLING,
        ),
    )
    assert rec.obj_model.num_atoms == 3, (
        f"cap of 50 % of 2 atoms should add 1, got {rec.obj_model.num_atoms - 2}"
    )
    # monotone reference: a flood of weak atoms lowers the median but not the bar
    with torch.no_grad():
        rec.obj_model._log_amplitudes.fill_(float(np.log(1.7e-3)))
    rec.reconstruct(
        num_iter=1,
        learn_pose=False,
        densify=DensifyParams(
            every=1, start=0, prune_amplitude_frac=0.5, prune_reference="max_median"
        ),
    )
    assert rec._prune_amp_ref == pytest.approx(1.7e-3, rel=0.2)
    rec.obj_model.add_atoms(np.array([[4.0, -4.0, 4.0]] * 6), amplitude=1e-5)
    rec.reconstruct(
        num_iter=1,
        learn_pose=False,
        densify=DensifyParams(
            every=1, start=0, prune_amplitude_frac=0.5, prune_reference="max_median"
        ),
    )
    assert rec.obj_model.num_atoms == 3, (
        "the six weak atoms must be pruned against the remembered reference"
    )


def test_densify_noise_aware_add_threshold_rejects_shot_noise(five_atom_series):
    gt, tilts, imgs = five_atom_series
    noisy = imgs + np.random.default_rng(0).normal(0.0, 0.02 * imgs.max(), imgs.shape).astype(
        imgs.dtype
    )
    added = {}
    for sig in (0.0, 4.0):
        rec = TomographyLiteSplat.from_dataset(
            noisy, tilts, SAMPLING, init=gt.numpy(), sigma_init=SIGMA, device="cpu", rng=0
        )
        n0 = rec.obj_model.num_atoms
        rec.reconstruct(
            num_iter=1,
            learn_pose=False,
            densify=DensifyParams(
                every=1,
                start=0,
                prune_amplitude_frac=0.0,
                add_from_residual=True,
                add_min_rel_peak=0.01,  # the single-atom-relative bar alone lets noise peaks in
                add_noise_sigmas=sig,
                voxel_size=SAMPLING,
            ),
        )
        added[sig] = rec.obj_model.num_atoms - n0
        if sig > 0:
            assert rec._last_add_noise > 0
    assert added[0.0] >= 1, "the relative bar alone should admit noise peaks in this setup"
    assert added[4.0] == 0, f"the noise-aware bar must reject them, added {added[4.0]}"


def test_prior_lam_relative_live_is_remeasured_each_application(five_atom_series):
    gt, tilts, imgs = five_atom_series
    lams = {}
    for mode in ("relative", "relative_live"):
        rec = TomographyLiteSplat.from_dataset(
            imgs, tilts, SAMPLING, init=gt.numpy() + 0.2, sigma_init=SIGMA, device="cpu", rng=0
        )
        rec.reconstruct(
            num_iter=4,
            learn_pose=False,
            prior=_CountingPrior(),
            prior_schedule=PriorSchedule(every=1, t=0.5, lam=1.0, lam_mode=mode),
        )
        lams[mode] = [h["lam"] for h in rec.prior_history]
    assert len(set(lams["relative"])) == 1, "fixed mode must keep the first-application scale"
    assert len(set(lams["relative_live"])) > 1, (
        "live mode must re-measure the scale as the fit moves"
    )


def test_prior_stop_proximal_mode_and_uniform_reset(five_atom_series):
    gt, tilts, imgs = five_atom_series
    # (a) stop: no prior calls at or after ``stop``
    rec = TomographyLiteSplat.from_dataset(
        imgs, tilts, SAMPLING, init=gt.numpy() + 0.2, sigma_init=SIGMA, device="cpu", rng=0
    )
    prior = _CountingPrior()
    rec.reconstruct(
        num_iter=6,
        learn_pose=False,
        prior=prior,
        prior_schedule=PriorSchedule(every=1, start=1, stop=4, t=0.5, lam=1.0),
    )
    assert [h["iteration"] for h in rec.prior_history] == [1, 2, 3], (
        "prior must run only in [start, stop)"
    )
    assert rec.obj_model._prior_target is None, "the target must be cleared when the prior stops"
    # (b) proximal mode moves the atoms alpha of the way to x_hat (= x + 0.05) on top of the data step:
    # compare with an identical fit without the prior (CPU runs are deterministic)
    pos = {}
    for use_prior in (False, True):
        rec = TomographyLiteSplat.from_dataset(
            imgs, tilts, SAMPLING, init=gt.numpy(), sigma_init=SIGMA, device="cpu", rng=0
        )
        rec.reconstruct(
            num_iter=1,
            learn_pose=False,
            prior=_CountingPrior() if use_prior else None,
            prior_schedule=PriorSchedule(
                every=1, start=0, t=0.5, lam=1.0, mode="proximal", alpha=0.4
            )
            if use_prior
            else None,
        )
        pos[use_prior] = rec.obj_model.positions.copy()
        if use_prior:
            assert rec.prior_history[0]["lam"] == 0.0
    shift = pos[True] - pos[False]
    assert np.allclose(shift, 0.4 * 0.05, atol=1e-5), (
        f"proximal step should add 0.02 to every coordinate, got {shift[0]}"
    )
    # (c) uniform reset: every amplitude and width equal after each iteration
    rec = TomographyLiteSplat.from_dataset(
        imgs, tilts, SAMPLING, init=gt.numpy() + 0.1, sigma_init=SIGMA, device="cpu", rng=0
    )
    with torch.no_grad():
        rec.obj_model._log_amplitudes += torch.linspace(-0.3, 0.3, rec.obj_model.num_atoms)
    rec.reconstruct(num_iter=2, learn_pose=False, uniform_every=1)
    amps = rec.obj_model.amplitudes_np
    assert np.ptp(amps) < 1e-6 * amps.mean(), (
        "uniform_every=1 must leave one amplitude for all atoms"
    )
    assert np.ptp(rec.obj_model.sigmas_np) < 1e-6
    # (d) windowed reset: an atom far below the median keeps its own (decaying) amplitude
    with torch.no_grad():
        rec.obj_model._log_amplitudes[0] -= float(np.log(10.0))
    rec.reconstruct(num_iter=1, learn_pose=False, uniform_every=1, uniform_window=(0.5, 2.0))
    amps = rec.obj_model.amplitudes_np
    assert amps[0] < 0.2 * np.median(amps), "the outlier must not be pulled back to the median"
    assert np.ptp(amps[1:]) < 1e-6 * amps[1:].mean()
    # (e) hard species reset every 2 iterations overrides the window (the outlier is reset too) and does not
    # fire in between
    rec.reconstruct(num_iter=1, learn_pose=False, uniform_every=1, uniform_window=(0.5, 2.0), species_reset_every=2)
    amps = rec.obj_model.amplitudes_np  # global iteration 4 (it + 1 = 4): hard reset
    assert np.ptp(amps) < 1e-6 * amps.mean(), "the hard reset must reset every atom, outliers included"
    with torch.no_grad():
        rec.obj_model._log_amplitudes[0] -= float(np.log(10.0))
    rec.reconstruct(num_iter=1, learn_pose=False, species_reset_every=2)
    amps = rec.obj_model.amplitudes_np  # it + 1 = 5: no reset
    assert amps[0] < 0.2 * np.median(amps), "the hard reset must only fire every species_reset_every iterations"
    # (f) energy mode (round 5): the prior is an energy; lam * dE/dx is stored as a constant gradient on the
    # positions at every refresh and lam * E is logged as the "prior" soft loss. Quadratic stand-in energy
    # E = sum |x - gt|^2 (forces -2 (x - gt)), checked at the first application (after the step of iteration 0).
    class _Quadratic:
        def __init__(self, c):
            self.c = c

        def energy_and_forces(self, x):
            d = x - self.c.to(x)
            return (d * d).sum(), (d * d).sum(-1), -2.0 * d

    rec = TomographyLiteSplat.from_dataset(
        imgs, tilts, SAMPLING, init=gt.numpy() + 0.2, sigma_init=SIGMA, device="cpu", rng=0
    )
    rec.reconstruct(
        num_iter=2,
        learn_pose=False,
        prior=_Quadratic(gt),
        prior_schedule=PriorSchedule(every=1, lam=0.5, lam_mode="absolute", mode="energy"),
    )
    x1 = torch.as_tensor(rec.obj_model.positions)  # positions after iteration 1 = where the last refresh was
    h = rec.prior_history[-1]
    assert h["lam"] == 0.5 and h["t"] == -1.0 and "pull_mean" not in h
    assert h["energy"] == pytest.approx(float(((x1 - gt) ** 2).sum()), rel=1e-5)
    assert torch.allclose(rec.obj_model._prior_grad, 0.5 * 2.0 * (x1 - gt), atol=1e-6), (
        "the stored gradient must be lam * dE/dx = -lam * F at the refresh positions"
    )
    assert rec.obj_model._prior_target is None, "energy mode must not set a quadratic target"
    assert rec.obj_model.soft_constraint_loss["prior"] == pytest.approx(
        0.5 * rec.prior_history[-2]["energy"], rel=1e-5
    ), "the logged prior soft loss of iteration 1 is lam * E from the refresh at iteration 0"
    # relative lam: fixed once so that ||lam * dE/dx|| = lam_rel * ||grad L_data||
    rec = TomographyLiteSplat.from_dataset(
        imgs, tilts, SAMPLING, init=gt.numpy() + 0.2, sigma_init=SIGMA, device="cpu", rng=0
    )
    rec.reconstruct(
        num_iter=2,
        learn_pose=False,
        prior=_Quadratic(gt),
        prior_schedule=PriorSchedule(every=1, lam=0.3, lam_mode="relative", mode="energy"),
    )
    lams = [h["lam"] for h in rec.prior_history]
    assert lams[0] == lams[1] != 0.3 and lams[0] == pytest.approx(0.3 * rec._lam_scale)
    x1 = torch.as_tensor(rec.obj_model.positions)
    assert rec.prior_history[-1]["grad_norm_prior"] == pytest.approx(
        lams[-1] * float((2.0 * (x1 - gt)).norm()), rel=1e-5
    )
    with pytest.raises(TypeError):  # a clean-estimate prior has no energy
        rec.reconstruct(
            num_iter=1, learn_pose=False, prior=_CountingPrior(),
            prior_schedule=PriorSchedule(every=1, lam=0.5, lam_mode="absolute", mode="energy"),
        )
    # (g) round 5 step 4: the energy term beside the learned prior (energy_prior / energy_schedule). Both terms
    # are set at every refresh (quadratic target kept, gradient = lam_e * dE/dx), each has its own history, the
    # energy's soft loss is logged as "energy", and each stop clears only its own term.
    rec = TomographyLiteSplat.from_dataset(
        imgs, tilts, SAMPLING, init=gt.numpy() + 0.2, sigma_init=SIGMA, device="cpu", rng=0
    )
    den = _CountingPrior()
    rec.reconstruct(
        num_iter=3,
        learn_pose=False,
        prior=den,
        prior_schedule=PriorSchedule(every=1, t=0.5, lam=0.2, lam_mode="absolute", stop=2),
        energy_prior=_Quadratic(gt),
        energy_schedule=PriorSchedule(every=1, lam=0.5, lam_mode="absolute", mode="energy", stop=10),
        schedule_total=3,
    )
    om = rec.obj_model
    assert len(den.calls) == 2 and len(rec.prior_history) == 2, "the denoiser stops at iteration 2"
    assert len(rec.energy_term_history) == 3 and "energy" in rec.energy_term_history[-1]
    assert all("pull_mean" in h for h in rec.prior_history), "the denoiser's records stay in prior_history"
    x2 = torch.as_tensor(om.positions)
    assert torch.allclose(om._prior_grad, 0.5 * 2.0 * (x2 - gt), atol=1e-6), (
        "the energy gradient must be refreshed at the last iteration although the denoiser has stopped"
    )
    assert om._prior_target is None, "the denoiser's stop must clear its target ..."
    assert om.soft_constraint_loss["energy"] == pytest.approx(0.5 * rec.energy_term_history[-2]["energy"], rel=1e-5), (
        "... and while both ran, the energy's lam * E was logged as 'energy' (the quadratic term keeps 'prior')"
    )
    rec2 = TomographyLiteSplat.from_dataset(
        imgs, tilts, SAMPLING, init=gt.numpy() + 0.2, sigma_init=SIGMA, device="cpu", rng=0
    )
    rec2.reconstruct(
        num_iter=2,
        learn_pose=False,
        prior=_CountingPrior(),
        prior_schedule=PriorSchedule(every=1, t=0.5, lam=0.2, lam_mode="absolute"),
        energy_prior=_Quadratic(gt),
        energy_schedule=PriorSchedule(every=1, lam=0.5, lam_mode="absolute", mode="energy", stop=1),
        schedule_total=2,
    )
    assert rec2.obj_model._prior_grad is None and rec2.obj_model._prior_target is not None, (
        "the energy stop clears only the gradient; the denoiser target set at iteration 1 stays"
    )
    assert len(rec2.energy_term_history) == 1 and len(rec2.prior_history) == 2
    rec3 = TomographyLiteSplat.from_dataset(
        imgs, tilts, SAMPLING, init=gt.numpy() + 0.2, sigma_init=SIGMA, device="cpu", rng=0
    )
    rec3.reconstruct(
        num_iter=2,
        learn_pose=False,
        prior=_CountingPrior(),
        prior_schedule=PriorSchedule(every=1, t=0.5, lam=0.2, lam_mode="absolute"),
        energy_prior=_Quadratic(gt),
        energy_schedule=PriorSchedule(every=1, lam=0.3, lam_mode="relative", mode="energy"),
        schedule_total=2,
    )
    lams = [h["lam"] for h in rec3.energy_term_history]
    assert rec3._lam_scale is None and lams[0] == lams[1] == pytest.approx(0.3 * rec3._lam_scale_energy), (
        "a relative lam of the second term is fixed at its first application on its OWN scale, not the denoiser's"
    )


def test_densify_support_and_whitened_rules_remove_a_spurious_atom_and_gate_additions(five_atom_series):
    """Chapter 06 rules: with the true atoms plus one spurious atom in vacuum, the data-support rule
    (SNR below the bar) and the whitened-residual rule (model too bright in every view) each remove
    exactly the spurious atom; the whitened add signal with the chi^2 gate adds nothing to a model
    that already explains the data; the max-remove cap limits a pass."""
    gt, tilts, imgs = five_atom_series
    init = np.vstack([gt.numpy(), [[4.0, -4.0, 4.0]]])
    for rule in ({"remove_snr_min": 3.0, "add_signal": "residual"}, {"remove_whitened_sigmas": 3.0}):
        rec = TomographyLiteSplat.from_dataset(
            imgs, tilts, SAMPLING, init=init, sigma_init=SIGMA, device="cpu", rng=0
        )
        rec.dset._dose_e_per_A2 = 1e5  # the rules need the shot-noise level
        rec.obj_model._log_amplitudes.data[-1] = rec.obj_model._log_amplitudes.data[0] - 1.0
        rec.reconstruct(
            num_iter=1,
            learn_pose=False,
            densify=DensifyParams(every=1, start=0, prune_amplitude_frac=0.0, add_from_residual=True,
                                  **{"add_signal": "whitened", "add_chi2_min": 1.5, "add_noise_sigmas": 4.0, **rule}),
        )  # fmt: skip
        assert rec.obj_model.num_atoms == 5, rule
        assert np.linalg.norm(rec.obj_model.positions[:, None] - gt.numpy()[None], axis=-1).min(1).max() < 0.2, rule
    rec = TomographyLiteSplat.from_dataset(
        imgs, tilts, SAMPLING, init=np.vstack([init, [[-4.0, 4.0, -4.0]]]), sigma_init=SIGMA, device="cpu", rng=0
    )
    rec.dset._dose_e_per_A2 = 1e5
    rec.obj_model._log_amplitudes.data[-2:] = rec.obj_model._log_amplitudes.data[0] - 1.0
    rec.reconstruct(num_iter=1, learn_pose=False,
                    densify=DensifyParams(every=1, start=0, prune_amplitude_frac=0.0, remove_snr_min=3.0, max_remove_frac=0.2))  # fmt: skip
    assert rec.obj_model.num_atoms == 6  # 7 atoms, cap 20 % -> one removal per pass

    # --- round 5 step 3: the per-atom energy as the remove signal (chapter 07; a fake energy, no MACE) ---
    class _FakeEnergy:
        """Per-atom 'energy' = minus the number of neighbours within 3.5 A: an isolated atom reads 0, a cluster atom <= -2."""

        def __init__(self):
            self.calls = []

        def per_atom_energy(self, x):
            self.calls.append(int(x.shape[0]))
            d = torch.cdist(x, x)
            return -((d < 3.5) & (d > 0)).sum(1).to(torch.float64)

    two_vac = np.vstack([init, [[-4.0, 4.0, -4.0]]])  # the 5 true atoms + 2 isolated splats in vacuum
    for case, low_amp, require, max_frac, expect in (
        ("AND, both splats weak", True, True, None, 5),
        ("AND, splats at the median amplitude -> kept", False, True, None, 7),
        ("energy alone removes them whatever the amplitude", False, False, None, 5),
        ("cap 20 % of 7 atoms -> one removal per pass", True, True, 0.2, 6),
    ):
        fake = _FakeEnergy()
        rec = TomographyLiteSplat.from_dataset(imgs, tilts, SAMPLING, init=two_vac, sigma_init=SIGMA, device="cpu", rng=0)
        if low_amp:
            rec.obj_model._log_amplitudes.data[-2:] = rec.obj_model._log_amplitudes.data[0] - 1.0  # 0.37 x the median
        rec.reconstruct(num_iter=1, learn_pose=False,
                        densify=DensifyParams(every=1, start=0, prune_amplitude_frac=0.0, energy_prior=fake, energy_prune_above=-0.5,
                                              energy_prune_require_low_amp=require, energy_prune_amp_frac=0.9, energy_prune_max_frac=max_frac))  # fmt: skip
        assert rec.obj_model.num_atoms == expect, case
        assert fake.calls == [7], case  # one energy call per pass, at the current atoms
        h = rec.energy_history
        assert len(h) == 1 and h[0]["kind"] == "prune" and h[0]["n_above"] == 2, case
        assert h[0]["n_removed"] == 7 - expect and h[0]["capped"] == (max_frac is not None), case
        assert all(e > -0.5 for e in h[0]["energies_removed_eV"]), case
        assert np.linalg.norm(rec.obj_model.positions[:5, None] - gt.numpy()[None], axis=-1).min(1).max() < 0.2, case
        assert rec._densify_history[-1]["removed_energy"] == 7 - expect, case

    # --- round 5 step 3: add vetting (the residual add from 3 of the 5 atoms proposes the two missing sites) ---
    class _AllUnbound(_FakeEnergy):
        def per_atom_energy(self, x):
            self.calls.append(int(x.shape[0]))
            return torch.ones(x.shape[0], dtype=torch.float64)  # every candidate reads +1 eV -> rejected

    for case, fake, expect_added in (("candidates on real sites pass the vet", _FakeEnergy(), 2), ("unbound candidates are rejected", _AllUnbound(), 0)):
        rec = TomographyLiteSplat.from_dataset(imgs, tilts, SAMPLING, init=gt[:3].numpy(), sigma_init=SIGMA, device="cpu", rng=0)
        rec.reconstruct(num_iter=1, learn_pose=False,
                        densify=DensifyParams(every=1, start=0, prune_amplitude_frac=0.0, add_from_residual=True, voxel_size=SAMPLING,
                                              energy_prior=fake, energy_prune_above=-0.5))  # fmt: skip
        assert rec.obj_model.num_atoms == 3 + expect_added, case
        vet = [h for h in rec.energy_history if h["kind"] == "vet"]
        assert len(vet) == 1 and vet[0]["n_candidates"] == 2 and vet[0]["n_rejected"] == 2 - expect_added, case
        assert fake.calls == [3, 3 + 2], case  # the prune call on the 3 atoms, then one in-place call with the 2 candidates inserted
        assert rec._densify_history[-1]["added"] == expect_added and rec._densify_history[-1]["rejected_energy"] == 2 - expect_added, case
    # the vet can be switched off: both candidates are added although they read +1 eV
    rec = TomographyLiteSplat.from_dataset(imgs, tilts, SAMPLING, init=gt[:3].numpy(), sigma_init=SIGMA, device="cpu", rng=0)
    rec.reconstruct(num_iter=1, learn_pose=False,
                    densify=DensifyParams(every=1, start=0, prune_amplitude_frac=0.0, add_from_residual=True, voxel_size=SAMPLING,
                                          energy_prior=_AllUnbound(), energy_prune_above=-0.5, energy_prune_max_frac=None, energy_vet_add=False))  # fmt: skip
    assert rec.obj_model.num_atoms == 5, "with the vet off both +1 eV candidates must be added"
    assert not [h for h in rec.energy_history if h["kind"] == "vet"] and rec._densify_history[-1]["rejected_energy"] == 0
    # (the 3 start atoms also read +1 eV but sit at the median amplitude, so the AND rule keeps them: n_removed 0)
    assert rec.energy_history[0]["n_above"] == 3 and rec.energy_history[0]["n_removed"] == 0
    # energy_start: a pass before it runs as plain v1 (no energy call, the two vacuum splats stay)
    fake = _FakeEnergy()
    rec = TomographyLiteSplat.from_dataset(imgs, tilts, SAMPLING, init=two_vac, sigma_init=SIGMA, device="cpu", rng=0)
    rec.reconstruct(num_iter=1, learn_pose=False,
                    densify=DensifyParams(every=1, start=0, prune_amplitude_frac=0.0, energy_prior=fake, energy_prune_above=-0.5,
                                          energy_prune_require_low_amp=False, energy_start=5))  # fmt: skip
    assert rec.obj_model.num_atoms == 7 and fake.calls == [] and rec.energy_history == []
    assert "removed_energy" not in rec._densify_history[-1]


def test_densify_params_defaults_are_v2_and_v1_is_reachable(five_atom_series):
    """Gate 06: the plain constructor (and ``v2()``) carry the chapter 06 rules, ``v1()`` the chapter 03
    values; a dataset without a dose cannot whiten and falls back to v1 with one warning."""
    d, v2, v1 = DensifyParams(every=5), DensifyParams.v2(every=5, prune_amplitude_frac=0.3), DensifyParams.v1(every=5)
    assert (d.add_signal, d.remove_snr_min, d.max_remove_frac) == ("whitened", 2.0, 0.2)
    assert (v2.add_signal, v2.remove_snr_min, v2.max_remove_frac, v2.prune_amplitude_frac) == ("whitened", 2.0, 0.2, 0.3)
    assert (v1.add_signal, v1.remove_snr_min, v1.max_remove_frac) == ("residual", 0.0, None)
    assert DensifyParams.v2(add_signal="residual").add_signal == "residual"  # an explicit field wins
    # round 3: every new rule is off in v1 and v2 (track A re-baselines with them)
    for name, p in (("v1", v1), ("v2", v2), ("default", d)):
        assert p.clone_grad_k == 0.0, name
    gt, tilts, imgs = five_atom_series
    rec = TomographyLiteSplat.from_dataset(imgs, tilts, SAMPLING, init=gt.numpy(), sigma_init=SIGMA, device="cpu", rng=0)
    assert rec.dset.dose_e_per_A2 is None
    with pytest.warns(UserWarning, match="falling back"):
        rec.reconstruct(num_iter=2, learn_pose=False, densify=DensifyParams(every=1, start=0, add_from_residual=True))
    assert rec.obj_model.num_atoms == 5


def test_densify_gradient_clone_splits_the_high_gradient_atom_along_its_descent_direction(five_atom_series):
    """Round 3 add rule (3DGS-style clone): atom 3 starts 0.4 A off its site, so over the densify interval its
    positional gradient is far above the population median; it is cloned once, the clone sits ``clone_offset``
    from it toward the true site, both halves carry half the amplitude (total mass unchanged), no residual add."""
    gt, tilts, imgs = five_atom_series
    init = gt.numpy().copy()
    init[3] += [0.0, 0.0, 0.4]
    rec = TomographyLiteSplat.from_dataset(imgs, tilts, SAMPLING, init=init, sigma_init=SIGMA, device="cpu", rng=0)
    params = DensifyParams(every=2, start=0, prune_amplitude_frac=0.0, add_from_residual=True, clone_grad_k=5.0,
                           clone_offset=0.8, add_from_residual_when_cloning=False, add_signal="residual", remove_snr_min=0.0)
    rec.reconstruct(num_iter=1, learn_pose=False, densify=params)
    p3 = rec.obj_model.positions[3].copy()
    a3 = float(rec.obj_model.amplitudes_np[3])
    mass = float(rec.obj_model.amplitudes_np.sum())
    rec.reconstruct(num_iter=1, learn_pose=False, densify=params)  # it + 1 = 2: the densify pass
    obj = rec.obj_model
    assert obj.num_atoms == 6, f"exactly the displaced atom must be cloned, got {obj.num_atoms - 5} clones"
    clone = obj.positions[5]
    assert np.linalg.norm(clone - obj.positions[3]) == pytest.approx(0.8, abs=0.05)
    u = (clone - obj.positions[3]) / np.linalg.norm(clone - obj.positions[3])
    assert float(u @ (gt[3].numpy() - p3)) / float(np.linalg.norm(gt[3].numpy() - p3)) > 0.9, "clone not on the descent side"
    amps = obj.amplitudes_np
    assert amps[5] == pytest.approx(amps[3], rel=1e-5) and amps[3] == pytest.approx(a3 / 2, rel=0.05)
    assert float(amps.sum()) == pytest.approx(mass, rel=0.05)


def test_pose_l1_penalty_holds_the_learned_shift_at_zero(five_atom_series):
    """Round 3: an L1 penalty on the pose corrections; tilt 0 is rendered 1 px off, a strong penalty keeps the
    learned shift near zero while the unpenalised fit follows the data."""
    gt, tilts, imgs = five_atom_series
    rot = pose_to_rotations(torch.tensor(tilts, dtype=torch.float32), torch.zeros(len(tilts)), torch.zeros(len(tilts)))
    sh = torch.zeros(len(tilts), 2)
    sh[0, 0] = 1.0
    shifted = render_projections(gt, torch.full((5,), 1.7e-3), torch.full((5,), SIGMA), rot, sh, SAMPLING, (24, 24),
                                 window_radius_for(SIGMA, SAMPLING)).numpy()  # fmt: skip
    learned = {}
    for w in (0.0, 1.0):
        rec = TomographyLiteSplat.from_dataset(shifted, tilts, SAMPLING, init=gt.numpy(), sigma_init=SIGMA, device="cpu", rng=0)
        rec.reconstruct(num_iter=25, learn_pose=True, position_lr=0.0, pose_l1=(w, 0.0))
        learned[w] = float(rec.dset.forward(None)[0][0, 0].detach())
    assert learned[0.0] > 0.5, f"without the penalty the shift must follow the data, got {learned[0.0]:.3f} px"
    assert abs(learned[1.0]) < 0.3 * learned[0.0], f"the L1 penalty must hold the shift near 0, got {learned[1.0]:.3f} px"


# --- chapter 10 step 4: the candidate classifiers in the densify pass (stub models, no checkpoints) ---
_A_FCC = 4.0729  # first-shell distance 2.88 A


@pytest.fixture(scope="module")
def cubocta_series():
    """A 13-atom cuboctahedron (a centre and its 12 first-shell neighbours), 7 tilts of 32 x 32 px."""
    h = _A_FCC / 2
    shell = [(sx * h, sy * h, 0.0) for sx in (-1, 1) for sy in (-1, 1)]
    shell += [(sx * h, 0.0, sz * h) for sx in (-1, 1) for sz in (-1, 1)]
    shell += [(0.0, sy * h, sz * h) for sy in (-1, 1) for sz in (-1, 1)]
    gt = torch.tensor([(0.0, 0.0, 0.0)] + shell)
    tilts = np.arange(-60.0, 61.0, 20.0)
    rot = pose_to_rotations(
        torch.tensor(tilts, dtype=torch.float32), torch.zeros(len(tilts)), torch.zeros(len(tilts))
    )
    imgs = render_projections(gt, torch.full((13,), 1.7e-3), torch.full((13,), SIGMA), rot,
                              torch.zeros(len(tilts), 2), SAMPLING, (32, 32),
                              window_radius_for(SIGMA, SAMPLING)).numpy()  # fmt: skip
    return gt, tilts, imgs


class _StubAddModel:
    """Add classifier stand-in: p = 1 for candidates within 0.5 A of ``sites``, 0 elsewhere."""

    d1 = 2.88

    def __init__(self, sites):
        self.sites = torch.as_tensor(np.asarray(sites), dtype=torch.float32).reshape(-1, 3)
        self.calls = []

    def score_candidates(self, atoms, candidates, d1=None):
        self.calls.append(int(candidates.shape[0]))
        d = torch.cdist(candidates.float(), self.sites.to(candidates.device)).min(dim=1).values
        return (d < 0.5).float()


class _StubRemoveModel:
    """Remove classifier stand-in: p_real = ``p`` for atoms within 0.3 A of ``sites[k]``, 0.9 elsewhere."""

    d1 = 2.88

    def __init__(self, sites, p):
        self.sites = torch.as_tensor(np.asarray(sites), dtype=torch.float32).reshape(-1, 3)
        self.p = torch.as_tensor(p, dtype=torch.float32)

    def score_atoms(self, atoms, d1=None):
        d, k = torch.cdist(atoms.float(), self.sites.to(atoms.device)).min(dim=1)
        return torch.where(d < 0.3, self.p.to(atoms.device)[k], torch.full_like(d, 0.9))


class _NeverCalled:
    d1 = 2.88

    def score_candidates(self, *a, **k):
        raise AssertionError("the add model must not be called")

    def score_atoms(self, *a, **k):
        raise AssertionError("the remove model must not be called")


def _cls_params(**kw):
    base = dict(every=1, start=0, prune_amplitude_frac=0.0, voxel_size=SAMPLING, cand_start=0)
    return DensifyParams.v1(**{**base, **kw})


def test_classifier_add_inserts_a_weak_tagged_atom_only_where_the_model_says(cubocta_series):
    gt, tilts, imgs = cubocta_series
    site = gt[1].numpy()  # a shell atom, removed
    rec = TomographyLiteSplat.from_dataset(imgs, tilts, SAMPLING, init=np.delete(gt.numpy(), 1, 0), sigma_init=SIGMA,
                                           device="cpu", rng=0)  # fmt: skip
    stub = _StubAddModel(site)
    rec.reconstruct(num_iter=1, learn_pose=False,
                    densify=_cls_params(add_from_residual=True, add_method="classifier", cand_add_model=stub))  # fmt: skip
    obj = rec.obj_model
    assert len(stub.calls) == 1 and stub.calls[0] >= 10, f"the generator must propose more than the one site: {stub.calls}"
    assert obj.num_atoms == 13, "exactly the one site with p = 1 must be added"
    assert np.linalg.norm(obj.positions[-1] - site) < 0.5
    amps, sig = obj.amplitudes.detach(), obj.sigmas.detach()
    assert float(amps[-1]) == pytest.approx(0.3 * float(amps[:12].median()), rel=1e-5)
    assert float(sig[-1]) == pytest.approx(float(sig[:12].median()), rel=1e-5)
    assert obj._atom_tags.tolist() == [-1] * 12 + [0], "the added atom is tagged with its add iteration"
    h, c = rec._densify_history[-1], rec.cand_history[-1]
    assert h["added"] == 0 and h["added_classifier"] == 1, "the residual add is replaced, not run beside"
    assert c["n_candidates"] == stub.calls[0] and c["n_accepted"] == 1 and c["n_added"] == 1
    # below the amplitude-prune bar before it is judged: the amplitude prune leaves it to the data confirmation
    with torch.no_grad():
        obj._log_amplitudes[-1] -= 5.0
    rec.reconstruct(num_iter=1, learn_pose=False,
                    densify=_cls_params(add_from_residual=True, add_method="classifier", cand_add_model=stub,
                                        prune_amplitude_frac=0.2))  # fmt: skip
    c = rec.cand_history[-1]
    assert obj.num_atoms == 13 and obj._atom_tags.tolist() == [-1] * 12 + [0], "an unjudged add must not be pruned"
    assert (c["n_prune_exempt_unjudged"], c["n_pruned_unjudged"], c["n_added"]) == (1, 0, 0)


def test_data_confirmation_judges_added_atoms_and_the_uniform_reset_leaves_them_alone(cubocta_series):
    gt, tilts, imgs = cubocta_series
    site_a, site_b = gt[1].numpy(), gt[4].numpy()  # opposite shell atoms, both removed
    rec = TomographyLiteSplat.from_dataset(imgs, tilts, SAMPLING, init=np.delete(gt.numpy(), [1, 4], 0), sigma_init=SIGMA,
                                           device="cpu", rng=0)  # fmt: skip
    params = _cls_params(add_from_residual=True, add_method="classifier", cand_add_model=_StubAddModel([site_a, site_b]),
                         cand_confirm_iters=2)  # fmt: skip
    rec.reconstruct(num_iter=1, learn_pose=False, densify=params)
    obj = rec.obj_model
    assert obj.num_atoms == 13 and obj._atom_tags.tolist() == [-1] * 11 + [0, 0]
    ia = 11 + int(np.linalg.norm(obj.positions[11:] - site_a, axis=1).argmin())
    with torch.no_grad():  # site a: inside the uniform window (0.5-2) but below the bar 0.6; site b: above the bar
        med = float(obj._log_amplitudes[:11].median())
        obj._log_amplitudes[ia] = med + np.log(0.55)
        obj._log_amplitudes[23 - ia] = med + np.log(0.9)
    rec.reconstruct(num_iter=1, learn_pose=False, densify=params, uniform_every=1, uniform_window=(0.5, 2.0))
    amps = obj.amplitudes.detach()
    rel_a = float(amps[ia] / amps[:11].median())
    assert 0.5 < rel_a < 0.6, f"the uniform reset must not lift an unconfirmed atom (rel amplitude {rel_a:.3f})"
    assert rec.cand_history[-1]["n_confirmed"] == 0 and obj.num_atoms == 13, "not judged before cand_confirm_iters"
    rec.reconstruct(num_iter=1, learn_pose=False, densify=params, uniform_every=1, uniform_window=(0.5, 2.0))
    c, pos = rec.cand_history[-1], obj.positions
    assert (c["n_confirmed"], c["n_rejected_data"], c["added_at_judged"]) == (1, 1, [0])
    assert obj.num_atoms == 12 and obj._atom_tags.tolist() == [-1] * 12
    assert np.linalg.norm(pos - site_a, axis=1).min() > 1.0, "the weak atom is rejected by the data"
    assert np.linalg.norm(pos - site_b, axis=1).min() < 0.5, "the strong atom is confirmed"
    assert c["n_blocked"] >= 1 and c["n_added"] == 0, "the rejected site must not be proposed again"
    amps = obj.amplitudes_np
    assert np.ptp(amps) < 1e-6 * amps.mean(), "a confirmed atom takes part in the uniform reset again"


def test_classifier_remove_needs_low_p_and_low_amplitude_and_respects_the_cap(cubocta_series):
    gt, tilts, imgs = cubocta_series
    s1, s2 = np.array([0.0, 0.0, 5.0]), np.array([5.0, 0.0, 0.0])  # two off-lattice splats
    init = np.vstack([gt.numpy(), s1, s2])  # 15 atoms; atom 1 (real, p_real 0.9) is weak in every case
    stub = _StubRemoveModel([s1, s2], [0.05, 0.2])
    for case, s2_weak, max_frac, s2_tag, kept in (
        ("both spurious atoms weak: both removed", True, None, False, [False, False]),
        ("s2 at the median amplitude: kept", False, None, False, [False, True]),
        ("cap 10 % of 15 = 1 removal, lowest p first", True, 0.1, False, [False, True]),
        ("an unconfirmed classifier add is left to the data", True, None, True, [False, True]),
    ):
        rec = TomographyLiteSplat.from_dataset(imgs, tilts, SAMPLING, init=init, sigma_init=SIGMA, device="cpu", rng=0)
        obj = rec.obj_model
        with torch.no_grad():
            for i in [1, 13] + ([14] if s2_weak else []):
                obj._log_amplitudes[i] -= 1.0  # 0.37 x the median
        if s2_tag:
            obj._atom_tags = torch.tensor([-1] * 14 + [0])
        rec.reconstruct(num_iter=1, learn_pose=False,
                        densify=_cls_params(prune_method="classifier", cand_rm_model=stub, cand_rm_max_frac=max_frac))  # fmt: skip
        pos = obj.positions
        got = [bool(np.linalg.norm(pos - s, axis=1).min() < 0.3) for s in (s1, s2)]
        assert got == kept, case
        assert obj.num_atoms == 13 + sum(kept), case
        assert np.linalg.norm(pos - gt.numpy()[1], axis=1).min() < 0.3, f"{case}: a weak real atom is kept"
        c = rec.cand_history[-1]
        assert c["n_removed_classifier"] == 2 - sum(kept) and c["rm_capped"] == (max_frac is not None), case


def test_classifier_options_at_their_defaults_reproduce_the_residual_bookkeeping(cubocta_series):
    """Defaults (and the classifier methods before cand_start) must give the chapter-09 densify pass exactly."""
    gt, tilts, imgs = cubocta_series
    init = np.vstack([np.delete(gt.numpy(), 1, 0), [[0.0, 0.0, 5.0]]])  # one site missing, one vacuum splat
    out = {}
    for case, kw in (
        ("chapter 09", {}),
        ("explicit defaults with models present", dict(add_method="residual", prune_method="amplitude",
                                                       cand_add_model=_NeverCalled(), cand_rm_model=_NeverCalled())),
        ("classifier methods before cand_start", dict(add_method="classifier", prune_method="classifier", cand_start=10,
                                                      cand_add_model=_NeverCalled(), cand_rm_model=_NeverCalled())),
    ):  # fmt: skip
        rec = TomographyLiteSplat.from_dataset(imgs, tilts, SAMPLING, init=init, sigma_init=SIGMA, device="cpu", rng=0)
        with torch.no_grad():
            rec.obj_model._log_amplitudes[-1] -= 2.0
        base = dict(every=1, start=0, prune_amplitude_frac=0.2, merge_distance=1.5, add_from_residual=True,
                    split_from_residual=True, max_add_frac=0.5, voxel_size=SAMPLING)  # fmt: skip
        rec.reconstruct(num_iter=3, learn_pose=False, densify=DensifyParams.v1(**{**base, **kw}),
                        uniform_every=1, uniform_window=(0.5, 2.0))  # fmt: skip
        o = rec.obj_model
        out[case] = (o._positions.detach().clone(), o._log_amplitudes.detach().clone(), rec._densify_history)
        assert rec.cand_history == [] and o._atom_tags is None, case
    ref = out["chapter 09"]
    assert ref[0].shape[0] == 13 and sum(h["added"] for h in ref[2]) == 1 and sum(h["pruned"] for h in ref[2]) == 1
    for case, (p, a, h) in out.items():
        assert torch.equal(p, ref[0]) and torch.equal(a, ref[1]) and h == ref[2], case
