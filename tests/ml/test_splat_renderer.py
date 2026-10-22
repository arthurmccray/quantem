"""Tests for ``quantem.core.ml.splat.renderer``: the analytic projection of isotropic 3D
Gaussians onto tilt images. All CPU. These tests pin the geometric conventions (tilt sign, axis
order, pixel centres, shift order) against (i) the closed-form Gaussian, (ii) the incoherent
HAADF simulator ``hfo2_haadf_zproj.splat_gaussians`` + ``au_carbon_sim.rot_y_matrix`` (copied
here in numpy), and (iii) quantem's INR ray projector."""

import math

import numpy as np
import pytest
import torch

from quantem.core.ml.splat.renderer import (
    backproject_images,
    project_positions,
    render_projections,
    window_radius_for,
)
from quantem.tomography.atom_trace import render_isotropic
from quantem.tomography.dataset_models import TomographyINRDataset
from quantem.tomography.utils import pose_to_rotations

SAMPLING = 0.5  # Angstrom / px
SIGMA = 0.8  # Angstrom
SHAPE = (24, 24)


@pytest.fixture(scope="module")
def three_atoms():
    """Three atoms with distinct z (so the tilt mixes z into the image), float64."""
    pos = torch.tensor([[1.3, -0.7, 2.1], [-2.2, 1.9, -1.4], [0.4, 3.1, 0.6]], dtype=torch.float64)
    amp = torch.tensor([1.0, 0.7, 1.3], dtype=torch.float64)
    sig = torch.full((3,), SIGMA, dtype=torch.float64)
    return pos, amp, sig


def _rotations(tilts, z1=0.0, z3=0.0):
    t = torch.tensor(tilts, dtype=torch.float64)
    return pose_to_rotations(t, torch.full_like(t, z1), torch.full_like(t, z3))


# --- numpy reference copied from hfo2_haadf_zproj.splat_gaussians / au_carbon_sim.rot_y_matrix ---
def _rot_y_matrix(tilt_deg):
    th = np.deg2rad(tilt_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _splat_gaussians_ref(pos_xy, weights, gpts, pixel_A, x0_A, y0_A, sigma_A, trunc_sigma=4.0):
    n0, n1 = gpts
    R = int(np.ceil(trunc_sigma * sigma_A / pixel_A))
    S = 2 * R + 1
    np0, np1 = n0 + 2 * S, n1 + 2 * S
    acc = np.zeros(np0 * np1)
    off = np.arange(-R, R + 1)
    norm = 1.0 / (2.0 * np.pi * sigma_A * sigma_A)
    fx = (pos_xy[:, 0] - x0_A) / pixel_A + S
    fy = (pos_xy[:, 1] - y0_A) / pixel_A + S
    ii = np.round(fx).astype(int)[:, None] + off[None, :]
    jj = np.round(fy).astype(int)[:, None] + off[None, :]
    gx = np.exp(((ii - fx[:, None]) * pixel_A) ** 2 * (-0.5 / sigma_A**2))
    gy = np.exp(((jj - fy[:, None]) * pixel_A) ** 2 * (-0.5 / sigma_A**2))
    w3 = (weights * norm)[:, None, None] * gx[:, :, None] * gy[:, None, :]
    ok = (
        (ii[:, :, None] >= 0)
        & (ii[:, :, None] < np0)
        & (jj[:, None, :] >= 0)
        & (jj[:, None, :] < np1)
    )
    idx = np.clip(ii, 0, np0 - 1)[:, :, None] * np1 + np.clip(jj, 0, np1 - 1)[:, None, :]
    w3[~ok] = 0.0
    acc += np.bincount(idx.ravel(), weights=w3.ravel(), minlength=np0 * np1)
    return acc.reshape(np0, np1)[S : S + n0, S : S + n1]


def test_single_gaussian_matches_closed_form_and_conserves_mass():
    pos = torch.tensor([[0.37, -0.81, 2.0]], dtype=torch.float64)
    amp = torch.tensor([2.5], dtype=torch.float64)
    sig = torch.tensor([3 * SAMPLING], dtype=torch.float64)  # 3 px wide
    rot = torch.eye(3, dtype=torch.float64)[None]
    H, W = SHAPE
    img = render_projections(
        pos,
        amp,
        sig,
        rot,
        torch.zeros(1, 2, dtype=torch.float64),
        SAMPLING,
        SHAPE,
        window_radius_for(float(sig), SAMPLING, 6.0),
    )
    rows = (torch.arange(H, dtype=torch.float64) - (H - 1) / 2) * SAMPLING
    cols = (torch.arange(W, dtype=torch.float64) - (W - 1) / 2) * SAMPLING
    d2 = (rows[:, None] - pos[0, 1]) ** 2 + (cols[None, :] - pos[0, 0]) ** 2
    expected = amp / (2 * math.pi * sig**2) * torch.exp(-d2 / (2 * sig**2))
    assert torch.allclose(img[0], expected, rtol=1e-6, atol=1e-12), "closed-form mismatch"
    mass = float(img.sum() * SAMPLING**2 / amp)
    assert abs(mass - 1.0) < 1e-3, f"mass not conserved: {mass}"


def test_matches_zproj_simulator_convention(three_atoms):
    """Simulator: image axis 0 = atom x (tilt-mixed), axis 1 = atom y = tilt axis, +tilt is a
    right-handed rotation about +y. Renderer: specimen (x, y, z) = atom (y, x, z) with
    R = pose_to_rotations(tilt, 0, 0) and images unflipped."""
    pos, amp, sig = three_atoms
    atoms_xyz = pos.numpy()[:, [1, 0, 2]]  # renderer (x, y, z) -> simulator atom (x, y, z)
    n = SHAPE[0]
    x0 = -(n - 1) / 2 * SAMPLING
    tilts = [-30.0, 45.0]
    r = window_radius_for(SIGMA, SAMPLING, 4.0)
    imgs = render_projections(
        pos,
        amp,
        sig,
        _rotations(tilts),
        torch.zeros(2, 2, dtype=torch.float64),
        SAMPLING,
        SHAPE,
        r,
    )
    for k, t in enumerate(tilts):
        rotated = atoms_xyz @ _rot_y_matrix(t).T
        ref = _splat_gaussians_ref(rotated[:, :2], amp.numpy(), SHAPE, SAMPLING, x0, x0, SIGMA)
        assert np.allclose(imgs[k].numpy(), ref, rtol=1e-10, atol=1e-14), f"tilt {t}"
    # the test is only meaningful if the tilt actually moves the image
    assert float((imgs[0] - imgs[1]).abs().max()) > 0.05 * float(imgs.max())


def test_matches_inr_ray_projector(three_atoms):
    """Project a voxelised copy of the Gaussians with quantem's INR ray machinery
    (create_batch_rays / transform_batch_rays / integrate_rays) at nonzero z1, tilt, z3 and
    shifts, and compare with the analytic renderer. The residual is trilinear-sampling error
    (1.6 % at sigma = 3.2 voxels); swapping z1 and z3 raises it to ~60 %, which is the control."""
    pos, amp, sig = three_atoms
    N, s = 64, 0.25
    sig_vox = SIGMA / s
    vox_pos = (pos / s + (N - 1) / 2)[:, [2, 1, 0]].to(torch.float32)  # (z, y, x) voxel coords
    peak = (amp / ((2 * math.pi) ** 1.5 * SIGMA**3)).to(torch.float32)  # 3D density peak
    volume = render_isotropic(vox_pos, peak, float(sig_vox), (N, N, N), int(4 * sig_vox) + 1)
    z1, tilt, z3 = 30.0, 35.0, -25.0
    shifts = torch.tensor([[1.5, -2.25]])  # px, (x = col, y = row)
    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    n_samples = 4 * N
    rays = TomographyINRDataset.create_batch_rays(
        ii.reshape(-1).float(), jj.reshape(-1).float(), N, n_samples
    )
    spec = TomographyINRDataset.transform_batch_rays(
        rays,
        torch.full((N * N,), z1),
        torch.full((N * N,), tilt),
        torch.full((N * N,), z3),
        shifts.expand(N * N, 2),
        N,
        1.0,
    )
    grid = spec.reshape(1, 1, N * N, n_samples, 3)  # grid_sample wants (x, y, z) normalised
    sampled = torch.nn.functional.grid_sample(
        volume[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0, 0, 0]  # (N*N, n_samples)
    inr = TomographyINRDataset.integrate_rays(sampled.reshape(-1), n_samples, N * N).reshape(N, N)
    inr = inr * (N - 1) * s / 2  # normalised ray length -> Angstrom

    def analytic(a1, a3):
        rot = pose_to_rotations(torch.tensor([tilt]), torch.tensor([a1]), torch.tensor([a3]))
        return render_projections(
            pos.float(),
            amp.float(),
            sig.float(),
            rot.float(),
            shifts,
            s,
            (N, N),
            window_radius_for(SIGMA, s, 5.0),
        )[0]

    img = analytic(z1, z3)
    err = float((img - inr).abs().max() / img.max())
    assert err < 0.03, f"INR / analytic mismatch {err:.3f}"
    corr = torch.corrcoef(torch.stack((img.reshape(-1), inr.reshape(-1))))[0, 1]
    assert float(corr) > 0.999
    err_swapped = float((analytic(z3, z1) - inr).abs().max() / img.max())
    assert err_swapped > 10 * err, "test cannot tell z1 from z3"


def test_gradcheck_all_inputs(three_atoms):
    pos, amp, sig = three_atoms
    rot = _rotations([-20.0, 50.0], z1=3.0, z3=-4.0).requires_grad_(True)
    shifts = torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=torch.float64, requires_grad=True)
    r = window_radius_for(SIGMA, SAMPLING, 6.0)
    args = (
        pos.clone().requires_grad_(True),
        amp.clone().requires_grad_(True),
        sig.clone().requires_grad_(True),
        rot,
        shifts,
    )

    def fn(p, a, sg, R, sh):
        return render_projections(p, a, sg, R, sh, SAMPLING, (16, 16), r)

    assert torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_tilt_chunking_and_background(three_atoms):
    pos, amp, sig = three_atoms
    rot = _rotations([-60.0, 0.0, 60.0])
    shifts = torch.zeros(3, 2, dtype=torch.float64)
    bg = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    r = window_radius_for(SIGMA, SAMPLING)
    full = render_projections(pos, amp, sig, rot, shifts, SAMPLING, SHAPE, r)
    chunked = render_projections(pos, amp, sig, rot, shifts, SAMPLING, SHAPE, r, tilt_chunk=1)
    with_bg = render_projections(pos, amp, sig, rot, shifts, SAMPLING, SHAPE, r, background=bg)
    assert torch.equal(full, chunked)
    assert torch.allclose(with_bg - full, bg[:, None, None].expand_as(full))


def test_project_positions_shift_order_is_col_row():
    pos = torch.tensor([[1.0, 2.0, 0.0]])  # x = 1 -> col, y = 2 -> row
    rot = torch.eye(3)[None]
    c = project_positions(pos, rot, torch.tensor([[0.5, 0.25]]), 1.0, (9, 9))[0, 0]
    assert torch.allclose(c, torch.tensor([2.0 + 4 + 0.25, 1.0 + 4 + 0.5]))  # (row, col)


def test_backproject_peaks_at_atoms(three_atoms):
    pos, amp, sig = three_atoms
    tilts = torch.arange(-60.0, 61.0, 10.0).tolist()
    rot = _rotations(tilts)
    shifts = torch.zeros(len(tilts), 2, dtype=torch.float64)
    imgs = render_projections(
        pos, amp, sig, rot, shifts, SAMPLING, SHAPE, window_radius_for(SIGMA, SAMPLING)
    )
    v = 0.5
    D = 24
    vol = backproject_images(imgs, rot, shifts, SAMPLING, (D, D, D), v)
    assert vol.shape == (D, D, D)
    for p in pos:
        k, i, j = (p[2] / v + (D - 1) / 2, p[1] / v + (D - 1) / 2, p[0] / v + (D - 1) / 2)
        k, i, j = int(round(float(k))), int(round(float(i))), int(round(float(j)))
        local = vol[max(k - 2, 0) : k + 3, max(i - 2, 0) : i + 3, max(j - 2, 0) : j + 3]
        kk, ii, jj = np.unravel_index(int(local.argmax()), local.shape)
        assert abs(kk - min(k, 2)) <= 1 and abs(ii - min(i, 2)) <= 1 and abs(jj - min(j, 2)) <= 1
