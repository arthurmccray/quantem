"""Tests for ``quantem.core.ml.splat.seeding``: the filtered back-projection seed volume keeps the
renderer's pixel-centre convention and sharpens the crude back-projection."""

import numpy as np
import torch

from quantem.core.ml.splat import (
    backproject_images,
    fbp_volume,
    filter_images,
    render_projections,
    window_radius_for,
)
from quantem.tomography.utils import pose_to_rotations

SAMPLING = 0.5
SIGMA = 0.6


def _peak_subvoxel(vol):
    """argmax with a per-axis parabolic refinement, in voxel units (k, i, j)."""
    k, i, j = np.unravel_index(int(torch.argmax(vol)), vol.shape)
    out = []
    for axis, c in enumerate((k, i, j)):
        idx = [k, i, j]
        vals = []
        for d in (-1, 0, 1):
            idx[axis] = c + d
            vals.append(float(vol[tuple(idx)]))
        a, b, cc = vals
        denom = a - 2 * b + cc
        out.append(c + (0.5 * (a - cc) / denom if denom != 0 else 0.0))
    return np.array(out)


def test_fbp_volume_pins_pixel_centre_convention_and_sharpens_backprojection():
    """One atom off-centre; the FBP peak must land on it to a fraction of a voxel. An ``N // 2``
    centre (radon_torch) would put it half a voxel off along every even axis, and a tilt-sign
    error would mirror z, so this pins the CONVENTIONS.md pixel centre ``(H - 1) / 2``."""
    pos = torch.tensor([[1.3, -0.7, 2.1]], dtype=torch.float64)
    tilts = torch.arange(-70.0, 71.0, 10.0, dtype=torch.float64)
    rot = pose_to_rotations(tilts, torch.zeros_like(tilts), torch.zeros_like(tilts))
    shifts = torch.zeros(len(tilts), 2, dtype=torch.float64)
    imgs = render_projections(
        pos, torch.tensor([1.0], dtype=torch.float64), torch.full((1,), SIGMA, dtype=torch.float64),
        rot, shifts, SAMPLING, (24, 24), window_radius_for(SIGMA, SAMPLING),
    )  # fmt: skip
    shape = (24, 24, 24)
    bp = backproject_images(imgs, rot, shifts, SAMPLING, shape, SAMPLING)
    fbp = fbp_volume(imgs, rot, shifts, SAMPLING, shape, SAMPLING, "ramp")
    assert fbp_volume(imgs, rot, shifts, SAMPLING, shape, SAMPLING, None).equal(bp)
    for name, vol in (("bp", bp), ("fbp", fbp)):
        k, i, j = _peak_subvoxel(vol)
        found = np.array([(j - 11.5) * SAMPLING, (i - 11.5) * SAMPLING, (k - 11.5) * SAMPLING])
        assert np.abs(found - pos[0].numpy()).max() < 0.1, (name, found)
    # the ramp filter removes the 1/r halo: the volume falls off faster away from the atom, along the
    # filtered direction (y, 1 A off) and along the beam (z, 2 A off, where the halo is the missing-wedge tail)
    k, i, j = (int(v) for v in np.round(_peak_subvoxel(fbp)))
    off_y = {n: float(v[k, i + 2, j] / v[k, i, j]) for n, v in (("bp", bp), ("fbp", fbp))}
    off_z = {n: float(v[k + 4, i, j] / v[k, i, j]) for n, v in (("bp", bp), ("fbp", fbp))}
    assert off_y["fbp"] < 0.5 * off_y["bp"], off_y
    assert off_z["fbp"] < 0.25 * off_z["bp"], off_z

def test_filter_images_filters_rows_only_and_windows_attenuate():
    imgs = torch.zeros(2, 32, 16)
    imgs[:, 16, 5] = 1.0  # a point: its row profile becomes the ramp kernel, its column stays a delta
    f = filter_images(imgs, "ramp")
    assert f.shape == imgs.shape
    assert torch.allclose(f[:, :, 4], torch.zeros(2, 32)) and torch.allclose(f[:, :, 6], torch.zeros(2, 32))
    assert float(f[0, 16, 5]) > 0 and float(f[0, 17, 5]) < 0  # ramp: positive centre, negative side lobes
    assert float(filter_images(imgs, "hann")[0, 16, 5]) < float(f[0, 16, 5])
