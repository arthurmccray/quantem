"""Analytic projection renderer for isotropic 3D Gaussians (pure torch).

Conventions (pinned by ``tests/ml/test_splat_renderer.py``; see also
``$SCRATCH/splat_dev_logs/00_reference/CONVENTIONS.md``):

* Specimen frame: Angstrom, origin at the rotation pivot (= image centre), positions ordered
  ``(x, y, z)`` with ``x`` along detector columns (the tilt axis), ``y`` along detector rows and
  ``z`` along the beam at zero tilt.
* Pose: ``q = R_k @ p`` maps the specimen frame to the beam frame of tilt ``k``; the detector
  point of ``q`` is ``row = q_y / s + (H - 1) / 2 + shift_y`` and
  ``col = q_x / s + (W - 1) / 2 + shift_x`` with ``s`` the sampling in Angstrom per pixel and the
  shifts in pixels ordered ``(x = col, y = row)``. ``R_k`` for quantem's tomography poses is
  :func:`quantem.tomography.utils.pose_to_rotations`.
* Pixel centres: pixel ``(i, j)`` sits at ``((i - (H - 1) / 2) s, (j - (W - 1) / 2) s)``; the
  Gaussians are evaluated exactly at the pixel centres (no pixel integration), as in the
  incoherent HAADF simulator ``hfo2_haadf_zproj.splat_gaussians``.
* Image model: ``I_k(u) = b_k + sum_i A_i / (2 pi sigma_i^2) exp(-|u - pi_k(p_i)|^2 / (2 sigma_i^2))``
  where ``A_i`` is the integrated 2D mass (image units x Angstrom^2) and ``sigma_i`` the
  (probe-convolved) width in Angstrom. An isotropic 3D Gaussian projects to this 2D Gaussian
  along any direction, so no covariance projection is needed.

Each Gaussian is stamped into a ``(2 r + 1)^2`` pixel window around its rounded projected
centre; the window *indices* are detached while the Gaussian is evaluated at the continuous
centre, so gradients reach the positions, amplitudes, widths, rotations and shifts (the
pattern of :func:`quantem.tomography.atom_trace.render_isotropic`). Memory is
``O(T N (2 r + 1)^2)`` per chunk of tilts (and the autograd graph keeps every chunk), which is
what the fused ``quantem.cuda.core.ml.splat_stamp`` kernel removes: ``render_projections``
takes ``backend="torch" | "cuda"`` (default from the environment variable
``SPLAT_RENDER_BACKEND``, else ``"torch"``); the CUDA backend does the projection, the unit
conversion and the background in torch and only the 2D stamping in the kernel, with the same
pixel-centre, rounding and clipping conventions, fp32 only.
"""

import importlib
import math
import os
import sys
import warnings
from typing import TYPE_CHECKING

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

__all__ = [
    "project_positions",
    "render_projections",
    "backproject_images",
    "window_radius_for",
]


def window_radius_for(sigma_max: float, sampling: float, window_sigmas: float = 4.0) -> int:
    """Half-width in pixels of the stamp window: ``ceil(window_sigmas * sigma_max / sampling)``.

    The default 4 sigma truncation leaves ``~exp(-8) = 3.4e-4`` of the mass outside the window,
    the same truncation as the HAADF simulator.
    """
    return int(math.ceil(float(window_sigmas) * float(sigma_max) / float(sampling)))


def project_positions(
    positions: "torch.Tensor",
    rotations: "torch.Tensor",
    shifts: "torch.Tensor",
    sampling: float,
    image_shape: tuple[int, int],
) -> "torch.Tensor":
    """Project specimen positions to fractional detector pixel coordinates.

    Parameters
    ----------
    positions : Tensor
        ``(N, 3)`` positions in Angstrom, specimen frame, ordered ``(x, y, z)``.
    rotations : Tensor
        ``(T, 3, 3)`` rotation matrices, specimen -> beam frame.
    shifts : Tensor
        ``(T, 2)`` image shifts in pixels, ordered ``(x = col, y = row)``.
    sampling : float
        Angstrom per pixel.
    image_shape : tuple[int, int]
        ``(H, W)``.

    Returns
    -------
    Tensor
        ``(T, N, 2)`` fractional pixel coordinates ordered ``(row, col)``.
    """
    H, W = (int(v) for v in image_shape)
    q = torch.einsum("tij,nj->tni", rotations, positions)  # (T, N, 3) beam frame
    row = q[..., 1] / sampling + (H - 1) / 2 + shifts[:, None, 1]
    col = q[..., 0] / sampling + (W - 1) / 2 + shifts[:, None, 0]
    return torch.stack((row, col), dim=-1)


def _stamp_offsets(window_radius: int, device) -> "torch.Tensor":
    rng = torch.arange(-window_radius, window_radius + 1, device=device)
    o0, o1 = torch.meshgrid(rng, rng, indexing="ij")
    return torch.stack((o0.reshape(-1), o1.reshape(-1)), dim=1)  # (W2, 2)


BACKENDS = ("torch", "cuda")
BACKEND_ENV = "SPLAT_RENDER_BACKEND"
_cuda_fallback_warned = False


def _cuda_stamp_or_none():
    """The ``quantem.cuda`` stamp op, or ``None`` (with a one-time warning) when ``quantem.cuda``
    is not importable or is an older wheel without ``splat_stamp``."""
    global _cuda_fallback_warned
    reason = None
    try:
        cuda_ml = importlib.import_module("quantem.cuda.core.ml")
    except ImportError as exc:
        reason = f"quantem.cuda is not importable ({exc})"
        fn = None
    else:
        fn = getattr(cuda_ml, "splat_stamp", None)
        if fn is None:
            version = getattr(sys.modules.get("quantem.cuda"), "__version__", "?")
            reason = (
                f"quantem.cuda {version} ({getattr(cuda_ml, '__file__', '?')}) has no "
                "splat_stamp op (an older wheel is on the path)"
            )
    if fn is None and not _cuda_fallback_warned:
        _cuda_fallback_warned = True
        warnings.warn(
            f"render_projections: backend='cuda' requested but {reason}; falling back to the "
            "torch backend for this process.",
            RuntimeWarning,
            stacklevel=3,
        )
    return fn


def _cuda_inputs_ok(*tensors) -> bool:
    return all(t.dtype == torch.float32 and t.is_cuda for t in tensors)


def resolve_backend(backend: str | None = None, tensors=()):
    """``(name, stamp_fn)`` for ``backend`` (keyword, else ``$SPLAT_RENDER_BACKEND``, else
    ``"torch"``); ``stamp_fn`` is ``None`` for the torch path.

    The environment variable states a preference: with ``"cuda"`` from the environment, inputs
    the kernel cannot take (CPU or non-fp32 ``tensors``) silently use the torch path, so CPU
    tests and fp64 checks keep working under a project-wide default. The keyword is a demand:
    ``backend="cuda"`` with such inputs raises in the CUDA path.
    """
    explicit = backend is not None
    name = backend if explicit else os.environ.get(BACKEND_ENV, "torch")
    name = str(name).strip().lower()
    if name not in BACKENDS:
        raise ValueError(f"unknown splat backend {name!r}; expected one of {BACKENDS}")
    if name == "cuda":
        if not explicit and tensors and not _cuda_inputs_ok(*tensors):
            return "torch", None
        fn = _cuda_stamp_or_none()
        if fn is not None:
            return "cuda", fn
    return "torch", None


def _render_projections_cuda(
    stamp,
    positions: "torch.Tensor",
    amplitudes: "torch.Tensor",
    sigmas: "torch.Tensor",
    rotations: "torch.Tensor",
    shifts: "torch.Tensor",
    sampling: float,
    image_shape: tuple[int, int],
    window_radius: int,
    background: "torch.Tensor | None",
    tilt_chunk: int | None,
) -> "torch.Tensor":
    """The fused path: projection (torch) -> pixel-unit stamp (kernel) -> background (torch)."""
    H, W = (int(v) for v in image_shape)
    T = int(rotations.shape[0])
    N = int(positions.shape[0])
    for name, t in (("positions", positions), ("amplitudes", amplitudes), ("sigmas", sigmas)):
        if t.dtype != torch.float32:
            raise TypeError(
                f"render_projections backend='cuda' is fp32-only ({name} is {t.dtype})"
            )
        if not t.is_cuda:
            raise ValueError(
                f"render_projections backend='cuda' needs CUDA tensors ({name} on {t.device})"
            )
    device, dtype = positions.device, positions.dtype
    sig = sigmas.reshape(-1)
    if sig.shape[0] == 1 and N != 1:
        sig = sig.expand(N)
    if N == 0:
        out = torch.zeros(T, H, W, device=device, dtype=dtype)
        return out if background is None else out + background.reshape(T, 1, 1)
    s = float(sampling)
    sig_px = sig / s  # widths in pixels
    amp_px = amplitudes / (s * s)  # pixel-integrated mass
    r = int(window_radius)
    n_chunk = T if tilt_chunk is None else max(1, int(tilt_chunk))
    parts = []
    for t0 in range(0, T, n_chunk):
        t1 = min(T, t0 + n_chunk)
        centers = project_positions(positions, rotations[t0:t1], shifts[t0:t1], s, (H, W))
        parts.append(stamp(centers.to(torch.float32), amp_px, sig_px, (H, W), r))
    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    if background is not None:
        out = out + background.reshape(T, 1, 1).to(dtype)
    return out


def render_projections(
    positions: "torch.Tensor",
    amplitudes: "torch.Tensor",
    sigmas: "torch.Tensor",
    rotations: "torch.Tensor",
    shifts: "torch.Tensor",
    sampling: float,
    image_shape: tuple[int, int],
    window_radius: int,
    background: "torch.Tensor | None" = None,
    tilt_chunk: int | None = None,
    backend: str | None = None,
) -> "torch.Tensor":
    """Render tilt images of a set of isotropic Gaussians (differentiable in every tensor).

    Parameters
    ----------
    positions : Tensor
        ``(N, 3)`` Angstrom, specimen frame ``(x, y, z)``.
    amplitudes : Tensor
        ``(N,)`` integrated 2D mass per Gaussian (image units x Angstrom^2).
    sigmas : Tensor
        ``(N,)`` or ``(1,)`` widths in Angstrom (a single value broadcasts).
    rotations : Tensor
        ``(T, 3, 3)`` specimen -> beam rotations.
    shifts : Tensor
        ``(T, 2)`` pixel shifts ``(x = col, y = row)``.
    sampling : float
        Angstrom per pixel.
    image_shape : tuple[int, int]
        ``(H, W)``.
    window_radius : int
        Half-width of the stamp window in pixels (see :func:`window_radius_for`).
    background : Tensor, optional
        ``(T,)`` constant added to each image.
    tilt_chunk : int, optional
        Number of tilts rendered per pass (bounds memory); default all.
    backend : str, optional
        ``"torch"`` (this module) or ``"cuda"`` (the fused ``quantem.cuda`` stamp kernel; needs
        fp32 CUDA tensors; falls back to torch with a one-time warning when ``quantem.cuda`` is
        missing or lacks the op). Default: the environment variable ``SPLAT_RENDER_BACKEND``
        read at call time, else ``"torch"``; from the environment, ``"cuda"`` is a preference
        and CPU or non-fp32 inputs silently take the torch path.

    Returns
    -------
    Tensor
        ``(T, H, W)`` images.
    """
    name, stamp = resolve_backend(backend, (positions, amplitudes, sigmas))
    if name == "cuda":
        return _render_projections_cuda(
            stamp,
            positions,
            amplitudes,
            sigmas,
            rotations,
            shifts,
            sampling,
            image_shape,
            window_radius,
            background,
            tilt_chunk,
        )
    H, W = (int(v) for v in image_shape)
    T = int(rotations.shape[0])
    N = int(positions.shape[0])
    device, dtype = positions.device, positions.dtype
    sig = sigmas.reshape(-1)
    if sig.shape[0] == 1 and N != 1:
        sig = sig.expand(N)
    norm = amplitudes / (2.0 * math.pi * sig * sig)  # (N,)
    inv2s2 = 0.5 / (sig * sig)  # (N,)
    r = int(window_radius)
    off = _stamp_offsets(r, device)  # (W2, 2)
    n_chunk = T if tilt_chunk is None else max(1, int(tilt_chunk))
    out = torch.zeros(T, H, W, device=device, dtype=dtype)
    if N == 0:
        return out if background is None else out + background.reshape(T, 1, 1)
    for t0 in range(0, T, n_chunk):
        t1 = min(T, t0 + n_chunk)
        tc = t1 - t0
        centers = project_positions(
            positions, rotations[t0:t1], shifts[t0:t1], sampling, (H, W)
        )  # (tc, N, 2)
        ci = torch.round(centers.detach()).long()  # integer window centres, no gradient
        pix = ci[:, :, None, :] + off[None, None, :, :]  # (tc, N, W2, 2) long
        d = (pix.to(dtype) - centers[:, :, None, :]) * sampling  # Angstrom
        r2 = (d * d).sum(dim=-1)  # (tc, N, W2)
        val = norm[None, :, None] * torch.exp(-r2 * inv2s2[None, :, None])
        valid = (pix[..., 0] >= 0) & (pix[..., 0] < H) & (pix[..., 1] >= 0) & (pix[..., 1] < W)
        val = val * valid.to(dtype)
        prow = pix[..., 0].clamp(0, H - 1)
        pcol = pix[..., 1].clamp(0, W - 1)
        tidx = torch.arange(tc, device=device).view(tc, 1, 1)
        flat = (tidx * H + prow) * W + pcol
        acc = torch.zeros(tc * H * W, device=device, dtype=dtype)
        acc = acc.scatter_add(0, flat.reshape(-1), val.reshape(-1))
        out[t0:t1] = acc.view(tc, H, W)
    if background is not None:
        out = out + background.reshape(T, 1, 1).to(dtype)
    return out


@torch.no_grad()
def backproject_images(
    images: "torch.Tensor",
    rotations: "torch.Tensor",
    shifts: "torch.Tensor",
    sampling: float,
    volume_shape: tuple[int, int, int],
    voxel_size: float,
    voxel_chunk: int = 2_000_000,
) -> "torch.Tensor":
    """Unfiltered back-projection of tilt images in the renderer's own convention.

    Voxel ``(k, i, j)`` of the output sits at the specimen point
    ``((j - (Wv - 1) / 2) v, (i - (Hv - 1) / 2) v, (k - (D - 1) / 2) v)`` in ``(x, y, z)`` with
    ``v`` the voxel size; it is projected with :func:`project_positions` into every tilt and
    the images are sampled bilinearly there and averaged over tilts. Used to seed Gaussians
    and to locate residual peaks; not differentiable.

    Returns
    -------
    Tensor
        ``(D, Hv, Wv)`` volume indexed ``(z, y, x)``.
    """
    D, Hv, Wv = (int(v) for v in volume_shape)
    T, H, W = (int(v) for v in images.shape)
    device, dtype = images.device, images.dtype
    kk, ii, jj = torch.meshgrid(
        torch.arange(D, device=device, dtype=dtype),
        torch.arange(Hv, device=device, dtype=dtype),
        torch.arange(Wv, device=device, dtype=dtype),
        indexing="ij",
    )
    pts = torch.stack(
        (
            (jj - (Wv - 1) / 2) * voxel_size,
            (ii - (Hv - 1) / 2) * voxel_size,
            (kk - (D - 1) / 2) * voxel_size,
        ),
        dim=-1,
    ).reshape(-1, 3)
    out = torch.zeros(pts.shape[0], device=device, dtype=dtype)
    imgs = images[:, None]  # (T, 1, H, W)
    for p0 in range(0, pts.shape[0], int(voxel_chunk)):
        chunk = pts[p0 : p0 + voxel_chunk]
        c = project_positions(chunk, rotations, shifts, sampling, (H, W))  # (T, n, 2)
        gx = 2.0 * c[..., 1] / (W - 1) - 1.0
        gy = 2.0 * c[..., 0] / (H - 1) - 1.0
        grid = torch.stack((gx, gy), dim=-1)[:, None]  # (T, 1, n, 2)
        sampled = torch.nn.functional.grid_sample(
            imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # (T, 1, 1, n)
        out[p0 : p0 + voxel_chunk] = sampled[:, 0, 0].mean(dim=0)
    return out.view(D, Hv, Wv)
