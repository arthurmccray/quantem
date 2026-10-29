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
