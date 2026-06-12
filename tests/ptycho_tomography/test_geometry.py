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
    pd = PtychoTomoPatchData(
        coords_yx=torch.zeros(2, 4, 4, 2),
        rotations=torch.eye(3).expand(2, 3, 3),
        tilt_indices=torch.zeros(2, dtype=torch.long),
    )
    assert pd.shifts_px is None
    assert pd.coords_yx.shape == (2, 4, 4, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
