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


def euler_from_rot_beam_to_spec(
    R: torch.Tensor, nominal_x_deg: torch.Tensor | float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert ``rot_beam_to_spec``: ``(T, 3)`` Euler triplets ``(z1, x, z3)`` in degrees from
    ``(T, 3, 3)`` beam→specimen matrices, plus a ``(T,)`` bool gimbal flag.

    Derivation (components ``(z, y, x)``, ``a = -z1``, ``b = -z3``):
    ``R = Rz(a) Rx(x) Rz(b)`` has ``R[0] = (cos x, sin x cos b, sin x sin b)`` and
    ``R[:, 0] = (cos x, -cos a sin x, sin a sin x)``, so ``x = atan2(hypot(R01, R02), R00)``,
    ``z3 = atan2(-R02, R01)`` and ``z1 = atan2(-R20, -R10)``. The triplet is two-valued —
    ``(z1 + 180, -x, z3 + 180)`` gives the same matrix — and the branch whose ``x`` is nearest
    ``nominal_x_deg`` is returned (a negative tilt must read as a negative x, not as a 180°
    in-plane flip). At ``sin x ≈ 0`` (the reference tilt) z1 and z3 are not separately
    defined — only their sum — so the sum is returned in z1 with z3 = 0 and the flag set;
    callers scoring angles must gauge-fix the MATRIX (``R_refᵀ R_t``) and never subtract
    triplets across that singularity.
    """
    Rm = torch.atleast_3d(R)
    nominal = torch.atleast_1d(torch.as_tensor(nominal_x_deg, dtype=Rm.dtype, device=Rm.device))
    nominal = torch.broadcast_to(nominal, Rm.shape[:1])
    rad2deg = 180.0 / torch.pi
    x_p = torch.atan2(torch.hypot(Rm[:, 0, 1], Rm[:, 0, 2]), Rm[:, 0, 0]) * rad2deg
    z1_p = torch.atan2(-Rm[:, 2, 0], -Rm[:, 1, 0]) * rad2deg
    z3_p = torch.atan2(-Rm[:, 0, 2], Rm[:, 0, 1]) * rad2deg

    def wrap(a: torch.Tensor) -> torch.Tensor:
        return torch.remainder(a + 180.0, 360.0) - 180.0

    alt = torch.stack([wrap(z1_p + 180.0), -x_p, wrap(z3_p + 180.0)], dim=-1)
    pri = torch.stack([wrap(z1_p), x_p, wrap(z3_p)], dim=-1)
    take_alt = (alt[:, 1] - nominal).abs() < (pri[:, 1] - nominal).abs()
    out = torch.where(take_alt[:, None], alt, pri)
    gimbal = torch.sin(x_p / rad2deg).abs() < 1e-9
    if bool(gimbal.any()):
        s = wrap(torch.atan2(Rm[:, 2, 1], Rm[:, 1, 1]) * rad2deg)  # z1 + z3 at sin x = 0
        gim = torch.stack([s, x_p, torch.zeros_like(s)], dim=-1)
        out = torch.where(gimbal[:, None], gim, out)
    return out, gimbal


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

    Coordinate convention (2026-07-10, physical-units refactor): all coordinates are **physical
    Å in the beam frame**, with the **origin at the specimen-box center** — which is anchored to
    the scan-footprint center by the dataset model — and the tilt/rotation axis passing through
    it. The object model owns the mapping from specimen-frame Å to its normalized ``[-1, 1]``
    support box; the dataset no longer needs to know anything about padding or box extents.

    Attributes
    ----------
    coords_yx_A:
        ``(B, Hroi, Wroi, 2)`` beam-frame ``(row, col)`` patch coordinates in **Å**, origin at
        the scan-footprint center (== specimen-box center); fractional scan positions are baked
        in. (Renamed from the pre-refactor ``coords_yx``, which was normalized over the padded
        pixel grid — the rename makes stale unit assumptions fail loudly.)
    rotations:
        ``(B, 3, 3)`` beam→specimen rotation matrices (``rot_beam_to_spec``), components
        ``(z, y, x)``.
    tilt_indices:
        ``(B,)`` long tensor mapping each batch element to its tilt (diagnostics / per-tilt
        extensions).
    shifts_A:
        Optional ``(B, 2)`` per-position beam-frame ``(dy, dx)`` pose shifts in Å (the per-tilt
        table ``PtychoTomoDatasetRaster._pose_shifts`` gathered per batch element), SUBTRACTED
        from ``coords_yx_A`` before rotation — a specimen physically displaced by ``+d`` in the
        beam frame is recovered as ``shift = +d``. The same subtraction is applied to the beam
        centre used for ``window_dz_A``. ``None`` whenever no tilt carries a shift and none is
        being learned (the pre-pose path, bit-identical).
    window_dz_A:
        Optional ``(B,)`` per-batch-element slab-window offset along the beam in Å (plan-view
        slab-window mode, 2026-07-16): the multislice slab stack for element ``b`` is displaced
        by ``window_dz_A[b]`` along the beam so it tracks the tilted specimen slab at that scan
        position instead of spanning the whole rotated extent. The object model adds it to the
        beam-frame z of every quadrature point; the reconstruction loop must pre-propagate the
        probe by the same distance (Fresnel factor) so the physics stays consistent. None =
        classic fixed stack.
    """

    coords_yx_A: torch.Tensor
    rotations: torch.Tensor
    tilt_indices: torch.Tensor
    shifts_A: torch.Tensor | None = None
    window_dz_A: torch.Tensor | None = None
