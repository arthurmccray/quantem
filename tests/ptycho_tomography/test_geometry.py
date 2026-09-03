"""Tests for the ptycho-tomography rotation conventions.

These pin the geometry against (a) the tomography module's ``transform_batch_rays`` (same Euler
convention by construction) and (b) the simulated tilt-series metadata (``bead_centers_lab_A`` in
``phantom_tilt_series_v1/tilt+035_meta.json``), so any convention drift fails loudly.
"""

import numpy as np
import pytest
import torch

from quantem.ptycho_tomography.geometry import (
    PtychoTomoPatchData,
    rot_beam_to_spec,
    slab_z_centers,
)
from quantem.tomography.dataset_models import TomographyINRDataset

# Anchor values copied verbatim from
# /wigeon/users/amccray/ptycho_tomo/phantom_tilt_series_v1/tilt+035_meta.json (tilt_deg = 35.0).
# bead_centers_A are specimen-frame, bead_centers_lab_A the same beads in the tilted (beam) frame.
TILT_DEG_ANCHOR = 35.0
BEADS_SPEC_ZYX = np.array(
    [
        [-4.800000000000001, 0.0, 12.0],
        [-4.800000000000001, 10.392304845413264, -5.999999999999997],
        [-4.800000000000001, -10.39230484541326, -6.000000000000005],
    ]
)
BEADS_BEAM_ZYX = np.array(
    [
        [-3.9319298125871613, -2.7531668944850214, 12.0],
        [-9.89271099129301, 5.759710864509648, -5.999999999999997],
        [2.028851366118685, -11.266044653479689, -6.000000000000005],
    ]
)


def phantom_gen_rotation_zxz(euler_deg: tuple[float, float, float]) -> np.ndarray:
    """Reference copy of the sim's ``phantom_gen.rotation_matrix_zxz`` (specimen → beam/lab)."""
    z1, xa, z3 = (np.deg2rad(a) for a in euler_deg)

    def rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)

    def rx(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)

    return rz(z1) @ rx(xa) @ rz(z3)


def test_identity_at_zero():
    R = rot_beam_to_spec(0.0, 0.0, 0.0)
    assert R.shape == (1, 3, 3)
    assert torch.allclose(R[0], torch.eye(3), atol=1e-7)


def test_pure_tilt_90_permutation():
    # T(0, 90, 0) maps beam (z, y, x) -> spec (y, -z, x)
    R = rot_beam_to_spec(0.0, 90.0, 0.0)[0]
    v = torch.tensor([1.0, 2.0, 3.0])  # (z, y, x)
    expected = torch.tensor([2.0, -1.0, 3.0])
    assert torch.allclose(R @ v, expected, atol=1e-6)


def test_orthogonality_and_inverse():
    R = rot_beam_to_spec(torch.tensor([10.0, -20.0]), torch.tensor([35.0, 70.0]), 5.0)
    eye = torch.eye(3).expand(2, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-6)
    # pure tilt: negating the angle transposes the matrix
    Rp = rot_beam_to_spec(0.0, 35.0, 0.0)
    Rm = rot_beam_to_spec(0.0, -35.0, 0.0)
    assert torch.allclose(Rp.transpose(-1, -2), Rm, atol=1e-6)


def test_coaxial_composition():
    Ra = rot_beam_to_spec(0.0, 20.0, 0.0)[0]
    Rb = rot_beam_to_spec(0.0, 15.0, 0.0)[0]
    Rab = rot_beam_to_spec(0.0, 35.0, 0.0)[0]
    assert torch.allclose(Ra @ Rb, Rab, atol=1e-6)


def test_metadata_anchor_tilt35():
    """rot_beam_to_spec(0, tilt, 0) recovers specimen-frame bead centers from beam-frame ones."""
    R = rot_beam_to_spec(0.0, TILT_DEG_ANCHOR, 0.0, dtype=torch.float64)[0]
    beam = torch.from_numpy(BEADS_BEAM_ZYX)
    spec = torch.from_numpy(BEADS_SPEC_ZYX)
    assert torch.allclose((R @ beam.T).T, spec, atol=1e-9)


def test_matches_phantom_gen_transpose():
    """Documented relation to the sim: rot_beam_to_spec(z1, x, z3) == G(-z3, x, -z1).T."""
    for euler in [(0.0, 35.0, 0.0), (0.0, -70.0, 0.0), (10.0, 25.0, -40.0)]:
        z1, x, z3 = euler
        ours = rot_beam_to_spec(z1, x, z3, dtype=torch.float64)[0].numpy()
        g = phantom_gen_rotation_zxz((-z3, x, -z1))
        np.testing.assert_allclose(ours, g.T, atol=1e-12)


def test_matches_tomography_transform_batch_rays():
    """Same Euler convention as the tomography module, just (z,y,x) vs (x,y,z) ordering."""
    torch.manual_seed(0)
    batch, n = 4, 7
    rays_xyz = torch.randn(batch, n, 3)
    z1 = torch.tensor([0.0, 10.0, -25.0, 5.0])
    x = torch.tensor([0.0, 35.0, -70.0, 90.0])
    z3 = torch.tensor([0.0, -5.0, 15.0, 30.0])
    shifts = torch.zeros(batch, 2)
    tomo = TomographyINRDataset.transform_batch_rays(
        rays_xyz, z1=z1, x=x, z3=z3, shifts=shifts, N=100, sampling_rate=1.0
    )
    R = rot_beam_to_spec(z1, x, z3)  # (batch, 3, 3) in (z, y, x)
    ours_zyx = torch.einsum("bij,bnj->bni", R, rays_xyz.flip(-1))
    assert torch.allclose(ours_zyx.flip(-1), tomo, atol=1e-5)


def test_shift_sign_matches_tomography_transform_batch_rays():
    """Pose shifts are subtracted from the beam-frame lateral coordinates BEFORE rotation, with
    the same sign as the tomography module. Ours are ``(dy, dx)`` in Å; tomography's are
    ``(x, y)`` in pixels normalized by ``sampling_rate * 2 / (N - 1)`` (== 1 here)."""
    torch.manual_seed(1)
    batch, n = 4, 7
    rays_xyz = torch.randn(batch, n, 3)
    z1 = torch.tensor([0.0, 10.0, -25.0, 5.0])
    x = torch.tensor([0.0, 35.0, -70.0, 90.0])
    z3 = torch.tensor([0.0, -5.0, 15.0, 30.0])
    shifts_yx = torch.tensor([[0.0, 0.0], [0.7, -0.3], [-1.2, 0.4], [0.5, 0.5]])  # (dy, dx)
    N = 100
    tomo = TomographyINRDataset.transform_batch_rays(
        rays_xyz, z1=z1, x=x, z3=z3, shifts=shifts_yx.flip(-1), N=N, sampling_rate=(N - 1) / 2
    )
    R = rot_beam_to_spec(z1, x, z3)
    rays_zyx = rays_xyz.flip(-1).clone()
    rays_zyx[..., 1] -= shifts_yx[:, None, 0]  # y_b - dy  (object_models.py forward)
    rays_zyx[..., 2] -= shifts_yx[:, None, 1]  # x_b - dx
    ours_zyx = torch.einsum("bij,bnj->bni", R, rays_zyx)
    assert torch.allclose(ours_zyx.flip(-1), tomo, atol=1e-5)


def test_batched_broadcast_and_grad():
    tilts = torch.tensor([-70.0, -35.0, 0.0, 35.0, 70.0], requires_grad=True)
    R = rot_beam_to_spec(0.0, tilts, 0.0)
    assert R.shape == (5, 3, 3)
    R.sum().backward()
    assert tilts.grad is not None and torch.isfinite(tilts.grad).all()


def test_slab_z_centers():
    z = slab_z_centers(4, 8.0)
    assert torch.allclose(z, torch.tensor([-3.0, -1.0, 1.0, 3.0]))
    z1 = slab_z_centers(1, 5.0)
    assert torch.allclose(z1, torch.tensor([0.0]))
    z16 = slab_z_centers(16, 56.08)
    assert abs(z16.mean().item()) < 1e-6
    assert torch.allclose(torch.diff(z16), torch.full((15,), 56.08 / 16), atol=1e-6)


def test_patch_data_fields():
    # physical-units refactor: coords_yx_A / shifts_A carry Å (renamed from the normalized
    # coords_yx / pixel shifts_px so stale unit assumptions fail loudly)
    pd = PtychoTomoPatchData(
        coords_yx_A=torch.zeros(2, 4, 4, 2),
        rotations=torch.eye(3).expand(2, 3, 3),
        tilt_indices=torch.zeros(2, dtype=torch.long),
    )
    assert pd.shifts_A is None
    assert pd.coords_yx_A.shape == (2, 4, 4, 2)
    pd2 = PtychoTomoPatchData(
        coords_yx_A=torch.zeros(2, 4, 4, 2),
        rotations=torch.eye(3).expand(2, 3, 3),
        tilt_indices=torch.zeros(2, dtype=torch.long),
        shifts_A=torch.tensor([[0.5, -0.5], [1.0, 2.0]]),
    )
    assert pd2.shifts_A is not None and pd2.shifts_A.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------- phase 2: tilt-axis angles
def test_known_point_pins_for_z1_and_z3():
    """Hand-computed images under the pinned composition Rz(-z1) Rx(x) Rz(-z3), components
    (z, y, x). _rz(t): y' = c y + s x, x' = -s y + c x. With z1 = 90 the matrix is _rz(-90):
    (0, 0, 1) -> (0, -1, 0) and (0, 1, 0) -> (0, 0, 1); z3 = 90 alone gives the same map (the
    two angles coincide at x = 0); with x = 90 between them they differ."""
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    ex = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    R = rot_beam_to_spec(90.0, 0.0, 0.0, dtype=torch.float64)[0]
    assert torch.allclose(R @ ex, torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(R @ ey, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-12)
    R3 = rot_beam_to_spec(0.0, 0.0, 90.0, dtype=torch.float64)[0]
    assert torch.allclose(R3, R, atol=1e-12)
    # z1 = 90 after a 90 deg tilt: Rz(-90) Rx(90); ex is on the tilt axis so Rx leaves it,
    # then Rz(-90) sends it to -ey. ey -> Rx(90): (z,y,x) = (1, 0, 0) -> Rz(-90) keeps z.
    R13 = rot_beam_to_spec(90.0, 90.0, 0.0, dtype=torch.float64)[0]
    assert torch.allclose(
        R13 @ ex, torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64), atol=1e-12
    )
    assert torch.allclose(R13 @ ey, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12)
    # z3 = 90 before a 90 deg tilt: Rx(90) Rz(-90); ex -> Rz(-90): -ey -> Rx(90): (z,y,x)
    # (0,-1,0) -> (-1, 0, 0)
    R31 = rot_beam_to_spec(0.0, 90.0, 90.0, dtype=torch.float64)[0]
    assert torch.allclose(
        R31 @ ex, torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12
    )
    assert not torch.allclose(R13, R31)
    # a constant z1 is a global rotation of the specimen frame about the beam axis
    Rc = rot_beam_to_spec(7.0, 35.0, 0.0, dtype=torch.float64)[0]
    assert torch.allclose(
        Rc,
        rot_beam_to_spec(7.0, 0.0, 0.0, dtype=torch.float64)[0]
        @ rot_beam_to_spec(0.0, 35.0, 0.0, dtype=torch.float64)[0],
        atol=1e-12,
    )


def test_euler_round_trip_both_branches():
    from quantem.ptycho_tomography.geometry import euler_from_rot_beam_to_spec

    g = torch.Generator().manual_seed(3)
    n = 200
    z1 = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * 170.0
    z3 = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * 170.0
    mag = 5.0 + torch.rand(n, generator=g, dtype=torch.float64) * 80.0
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(torch.float64)
    x = mag * sign
    R = rot_beam_to_spec(z1, x, z3, dtype=torch.float64)
    eul, gimbal = euler_from_rot_beam_to_spec(R, x)  # branch chosen by the nominal tilt
    assert not gimbal.any()
    assert torch.allclose(eul[:, 0], z1, atol=1e-9)
    assert torch.allclose(eul[:, 1], x, atol=1e-9)
    assert torch.allclose(eul[:, 2], z3, atol=1e-9)
    # the nominal tilt only has to be on the right side: a coarse nominal still picks the branch
    eul2, _ = euler_from_rot_beam_to_spec(R, torch.sign(x) * 30.0)
    assert torch.allclose(eul2, eul, atol=1e-9)
    # and the matrix rebuilt from the triplet is the input matrix on either branch
    R2 = rot_beam_to_spec(eul[:, 0], eul[:, 1], eul[:, 2], dtype=torch.float64)
    assert torch.allclose(R2, R, atol=1e-12)


def test_euler_gimbal_and_matrix_gauge_identity():
    """At x = 0 only z1 + z3 is defined (gimbal flag); and with the reference tilt at 0 deg the
    matrix gauge R_ref^T R_t reads (z1_t - (z1_ref + z3_ref), x_t, z3_t): the only surviving
    gauge is a constant offset of z1, which is why a scorer must gauge-fix BOTH tables."""
    from quantem.ptycho_tomography.geometry import euler_from_rot_beam_to_spec

    R0 = rot_beam_to_spec(4.0, 0.0, -1.5, dtype=torch.float64)
    eul, gimbal = euler_from_rot_beam_to_spec(R0, 0.0)
    assert bool(gimbal[0])
    assert eul[0, 0].item() == pytest.approx(2.5, abs=1e-9)  # z1 + z3
    assert eul[0, 2].item() == pytest.approx(0.0)
    z1r, z3r = 3.0, -1.0  # reference tilt (0 deg) learned with a nonzero (gauge) triplet
    R_ref = rot_beam_to_spec(z1r, 0.0, z3r, dtype=torch.float64)[0]
    z1t, xt, z3t = 2.5, -40.0, -2.5
    R_t = rot_beam_to_spec(z1t, xt, z3t, dtype=torch.float64)[0]
    rel, gim = euler_from_rot_beam_to_spec((R_ref.T @ R_t)[None], xt)
    assert not bool(gim[0])
    assert rel[0].tolist() == pytest.approx([z1t - (z1r + z3r), xt, z3t], abs=1e-9)


def test_angle_grads_flow_through_rot_beam_to_spec():
    z1 = torch.zeros(3, requires_grad=True)
    z3 = torch.zeros(3, requires_grad=True)
    R = rot_beam_to_spec(z1, torch.tensor([-35.0, 0.0, 35.0]), z3)
    (R * torch.arange(27.0).view(3, 3, 3)).sum().backward()
    assert z1.grad is not None and z3.grad is not None
    assert torch.isfinite(z1.grad).all() and torch.isfinite(z3.grad).all()
    assert (z1.grad != 0).any() and (z3.grad != 0).any()
