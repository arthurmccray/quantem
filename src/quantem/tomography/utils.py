import torch
import torch.nn.functional as F

# --- Projection Operator Utils ---


def rot_ZXZ(mags, z1, x, z3, device, mode="bilinear"):
    if not isinstance(x, torch.Tensor) or not isinstance(z1, torch.Tensor):
        z1 = torch.tensor(z1, dtype=torch.float32, device=device)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        z3 = torch.tensor(z3, dtype=torch.float32, device=device)
    curr_mags = mags

    curr_mags = differentiable_rotz_vectorized(curr_mags, z1, mode)
    curr_mags = differentiable_rotx_vectorized(curr_mags, x, mode)

    curr_mags = differentiable_rotz_vectorized(curr_mags, z3, mode)

    return curr_mags


def differentiable_rotz_vectorized(mags, theta, mode="bilinear"):
    _, dimz, dimy, dimx = mags.shape

    if theta.dim() == 0:
        theta = theta.unsqueeze(0)

    theta_rad = torch.deg2rad(theta)

    cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)
    affine_matrix = torch.stack(
        [cos_t, -sin_t, torch.zeros_like(theta), sin_t, cos_t, torch.zeros_like(theta)], dim=-1
    ).view(-1, 2, 3)

    mags = mags.permute(1, 0, 2, 3)

    def transform_slice(mag_slice):
        grid = F.affine_grid(affine_matrix, mag_slice.unsqueeze(0).shape, align_corners=False)
        return F.grid_sample(mag_slice.unsqueeze(0), grid, mode=mode, align_corners=False).squeeze(
            0
        )

    rotated_mags = torch.vmap(transform_slice)(mags)
    return rotated_mags.permute(1, 0, 2, 3)


def differentiable_rotx_vectorized(mags, theta, mode="bilinear"):
    _, dimz, dimy, dimx = mags.shape

    if theta.dim() == 0:
        theta = theta.unsqueeze(0)

    theta_rad = torch.deg2rad(theta)

    cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)
    affine_matrix = torch.stack(
        [cos_t, -sin_t, torch.zeros_like(theta), sin_t, cos_t, torch.zeros_like(theta)], dim=-1
    ).view(-1, 2, 3)

    mags = mags.permute(3, 0, 1, 2)

    def transform_slice(mag_slice):
        grid = F.affine_grid(affine_matrix, mag_slice.unsqueeze(0).shape, align_corners=False)
        return F.grid_sample(mag_slice.unsqueeze(0), grid, mode=mode, align_corners=False).squeeze(
            0
        )

    rotated_mags = torch.vmap(transform_slice)(mags)
    return rotated_mags.permute(1, 2, 3, 0)


def tv_loss_1d(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    1D Total Variation Loss.

    Encourages piecewise smoothness by penalizing differences between
    adjacent elements.

    Args:
        x:         Input tensor of shape (N, C, L) or (N, L) or (L,)
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        Scalar loss (or per-sample tensor if reduction='none')
    """
    # Difference between adjacent elements along the last dimension
    diff = x[..., 1:] - x[..., :-1]  # shape: (..., L-1)
    tv = diff.abs()  # L1 variant  ← most common

    if reduction == "mean":
        return tv.mean()
    elif reduction == "sum":
        return tv.sum()
    elif reduction == "none":
        return tv
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")


# --- Pose helpers (Gaussian-splat path) ---


def pose_to_rotations(
    tilt_deg: torch.Tensor, z1_deg: torch.Tensor, z3_deg: torch.Tensor
) -> torch.Tensor:
    """Specimen -> beam rotation matrices for quantem's ZXZ tomography poses.

    ``R = Rz(z3) @ Rx(-tilt) @ Rz(z1)``, the transpose (inverse) of the beam -> specimen map
    used by ``TomographyINRDataset.transform_batch_rays``
    (``p_spec = Rz(-z1) @ Rx(tilt) @ Rz(-z3) @ p_beam``), so that ``q = R @ p_spec`` is the
    beam-frame point whose ``(x, y)`` land on the detector. Angles in degrees; ``x`` is the
    tilt axis (detector columns), ``z`` the beam.

    Parameters
    ----------
    tilt_deg, z1_deg, z3_deg : Tensor
        ``(T,)`` angles in degrees (broadcastable).

    Returns
    -------
    Tensor
        ``(T, 3, 3)`` rotation matrices.
    """
    tilt = torch.deg2rad(torch.as_tensor(tilt_deg)).reshape(-1)
    z1 = torch.deg2rad(torch.as_tensor(z1_deg, dtype=tilt.dtype, device=tilt.device)).reshape(-1)
    z3 = torch.deg2rad(torch.as_tensor(z3_deg, dtype=tilt.dtype, device=tilt.device)).reshape(-1)
    tilt, z1, z3 = torch.broadcast_tensors(tilt, z1, z3)
    one = torch.ones_like(tilt)
    zero = torch.zeros_like(tilt)

    def _rz(a):
        c, s = torch.cos(a), torch.sin(a)
        return torch.stack(
            (
                torch.stack((c, -s, zero), -1),
                torch.stack((s, c, zero), -1),
                torch.stack((zero, zero, one), -1),
            ),
            -2,
        )

    def _rx(a):
        c, s = torch.cos(a), torch.sin(a)
        return torch.stack(
            (
                torch.stack((one, zero, zero), -1),
                torch.stack((zero, c, -s), -1),
                torch.stack((zero, s, c), -1),
            ),
            -2,
        )

    return _rz(z3) @ _rx(-tilt) @ _rz(z1)
