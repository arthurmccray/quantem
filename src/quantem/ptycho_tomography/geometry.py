"""Rotation conventions and the dataset→object payload for joint ptycho-tomography.

Frames and conventions
----------------------
All 3-vectors are ordered ``(z, y, x)`` matching the volume layout ``(Nz, Ny, Nx)`` used by the
phantom/simulation utilities and the ptychography object models (beam axis first):

* **beam frame** (per tilt): ``z`` along the beam (multislice propagation direction), ``y`` = scan
  rows, ``x`` = scan columns. Physical units are Å, origin at the object-box center.
* **specimen frame**: coincides with the beam frame at zero tilt; the reconstructed 3D object
  lives here.

``rot_beam_to_spec(z1, x, z3)`` builds the matrix that maps beam-frame coordinates into the
specimen frame. The Euler composition (intrinsic Z-X-Z, degrees, ``Rz(-z1) @ Rx(x) @ Rz(-z3)``
with the elementary rotations below) is **identical to the tomography module's**
``transform_batch_rays`` (``quantem.tomography.dataset_models``), expressed here in ``(z, y, x)``
component order instead of tomography's ``(x, y, z)``. The tilt axis is ``x`` (scan columns) for
both, and a tilt series enters as ``x = tilt_deg`` with the same sign.

Relation to the simulation (``phantom_gen.rotation_matrix_zxz``): the simulation rasterizes the
phantom at orientation ``euler_deg`` using a specimen→lab matrix ``G``; for a pure tilt
``euler_deg = (0, tilt, 0)``, ``rot_beam_to_spec(0, tilt, 0) == G.T`` exactly — so querying the
specimen-frame object at ``rot_beam_to_spec(0, tilt_deg, 0) @ r_beam`` reproduces the simulated
orientation with the meta-file ``tilt_deg`` used directly. (General Euler triplets map as
``rot_beam_to_spec(z1, x, z3) == G(-z3, x, -z1).T``; only pure tilts are used by the sims.)
This is pinned numerically against ``bead_centers_lab_A`` from the simulated tilt-series metadata
in ``tests/ptycho_tomography/test_geometry.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _rz(theta_deg: torch.Tensor) -> torch.Tensor:
    """Elementary rotation about the beam axis ``z``, components ``(z, y, x)``, shape (..., 3, 3).

    Matches the tomography module's in-plane rotation: ``x' = c·x − s·y``, ``y' = s·x + c·y``.
    """
    t = torch.deg2rad(theta_deg)
    c, s = torch.cos(t), torch.sin(t)
    o, i = torch.zeros_like(c), torch.ones_like(c)
    rows = torch.stack([i, o, o, o, c, s, o, -s, c], dim=-1)
    return rows.reshape(*theta_deg.shape, 3, 3)


def _rx(theta_deg: torch.Tensor) -> torch.Tensor:
    """Elementary rotation about the tilt axis ``x`` (scan columns), components ``(z, y, x)``.

    Matches the tomography module's tilt rotation: ``y' = c·y − s·z``, ``z' = s·y + c·z``.
    """
    t = torch.deg2rad(theta_deg)
    c, s = torch.cos(t), torch.sin(t)
    o, i = torch.zeros_like(c), torch.ones_like(c)
    rows = torch.stack([c, s, o, -s, c, o, o, o, i], dim=-1)
    return rows.reshape(*theta_deg.shape, 3, 3)


def rot_beam_to_spec(
    z1_deg: torch.Tensor | float,
    x_deg: torch.Tensor | float,
    z3_deg: torch.Tensor | float,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Beam-frame → specimen-frame rotation matrices, shape ``(T, 3, 3)``, components ``(z, y, x)``.

    Differentiable in the angle tensors (for future pose refinement). Angles in degrees with the
    tomography module's Z-X-Z convention; a pure tilt series is ``rot_beam_to_spec(0, tilts, 0)``.
    Scalars and 1D tensors broadcast together.
    """
    angles = []
    for a in (z1_deg, x_deg, z3_deg):
        t = a if isinstance(a, torch.Tensor) else torch.tensor(float(a))
        angles.append(torch.atleast_1d(t.to(device=device, dtype=dtype or torch.float32)))
    z1, x, z3 = torch.broadcast_tensors(*angles)
    return _rz(-z1) @ _rx(x) @ _rz(-z3)


def slab_z_centers(
    num_slices: int,
    total_thickness: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Beam-frame z of each multislice slab center, shape ``(S,)``, centered about 0 (Å).

    The slab of slice ``s`` spans ``[s·t, (s+1)·t] − T/2`` with ``t = T/S``; the object is queried
    at slab centers so the per-slice transmission approximates the slab's line integral
    (midpoint rule). Slice 0 is the beam-entrance slab (most negative z).
    """
    t = total_thickness / num_slices
    idx = torch.arange(num_slices, device=device, dtype=dtype or torch.float32)
    return (idx + 0.5) * t - total_thickness / 2.0


@dataclass
class PtychoTomoPatchData:
    """Object-query payload passed opaquely from the dataset model to the object model.

    Carries everything the rotated-coordinate object query needs for one batch. Defined at module
    level (not nested) so it pickles across the multi-GPU forkserver spawn.

    Attributes
    ----------
    coords_yx:
        ``(B, Hroi, Wroi, 2)`` normalized beam-frame ``(row, col)`` patch coordinates in
        ``[-1, 1]`` over the padded 2D object extent (same construction as the implicit-object
        ``_scan_coords``); fractional scan positions are baked in.
    rotations:
        ``(B, 3, 3)`` beam→specimen rotation matrices (``rot_beam_to_spec``), components
        ``(z, y, x)``.
    tilt_indices:
        ``(B,)`` long tensor mapping each batch element to its tilt (diagnostics / per-tilt
        extensions).
    shifts_px:
        Optional ``(B, 2)`` beam-frame alignment shifts in pixels, subtracted before rotation
        (per-tilt pose refinement; unused/None in v1).
    """

    coords_yx: torch.Tensor
    rotations: torch.Tensor
    tilt_indices: torch.Tensor
    shifts_px: torch.Tensor | None = None
