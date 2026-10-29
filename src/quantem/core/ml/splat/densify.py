"""Add / remove Gaussians while keeping the optimizer state consistent.

The object model holds one torch optimizer with one param group per key (``positions``,
``amplitudes``, ``sigmas``, ...). When atoms are pruned, merged or added, every per-atom
Parameter is rebuilt and the Adam moments are re-indexed with :func:`remap_optimizer_state`
(rows of kept atoms follow their atom, new atoms start from zero moments), so a densification
step never resets the optimisation of the surviving atoms.
"""

from typing import TYPE_CHECKING

import math

import numpy as np
from scipy.spatial import cKDTree

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

__all__ = [
    "remap_optimizer_state",
    "prune_mask",
    "merge_close_atoms",
    "pixel_variance",
    "atom_support",
]


def remap_optimizer_state(
    optimizer: "torch.optim.Optimizer | None",
    old_param: "torch.Tensor",
    new_param: "torch.Tensor",
    keep_idx: "torch.Tensor | None",
    n_new: int,
) -> None:
    """Replace ``old_param`` by ``new_param`` in its param group and re-index its state.

    Parameters
    ----------
    optimizer : Optimizer or None
        The model optimizer (nothing happens when ``None`` or when ``old_param`` is not in it).
    old_param, new_param : Tensor
        The Parameter being replaced and its replacement; ``new_param`` has
        ``len(keep_idx) + n_new`` rows.
    keep_idx : Tensor or None
        Indices of the rows of ``old_param`` that survive (in order). ``None`` keeps all rows.
    n_new : int
        Number of appended rows (zero-initialised moments).
    """
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        params = group["params"]
        for i, p in enumerate(params):
            if p is old_param:
                state = optimizer.state.pop(p, None)
                if state is not None:
                    new_state = {}
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.shape == old_param.shape:
                            kept = v if keep_idx is None else v[keep_idx]
                            if n_new > 0:
                                pad = torch.zeros(
                                    (n_new, *v.shape[1:]), dtype=v.dtype, device=v.device
                                )
                                kept = torch.cat((kept, pad), dim=0)
                            new_state[k] = kept
                        else:
                            new_state[k] = v
                    optimizer.state[new_param] = new_state
                params[i] = new_param
                return


def prune_mask(
    amplitudes: "torch.Tensor",
    positions: "torch.Tensor",
    amplitude_min: float,
    box_half: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Boolean mask of atoms to KEEP: amplitude above ``amplitude_min`` and inside the box."""
    keep = amplitudes > amplitude_min
    if box_half is not None:
        keep = keep & (positions.abs() <= box_half.to(positions)).all(dim=-1)
    return keep


def merge_close_atoms(
    positions: "torch.Tensor",
    amplitudes: "torch.Tensor",
    sigmas: "torch.Tensor",
    min_distance: float,
    amplitude_mode: str = "sum",
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Merge pairs of atoms closer than ``min_distance`` (brightest wins).

    Returns ``(keep_idx, positions, amplitudes, sigmas)`` where the merged atom sits at the
    amplitude-weighted mean position and carries the brighter atom's sigma and either the
    summed amplitude (``amplitude_mode="sum"``, mass conserving: a pair on one site becomes one
    over-bright atom that the fit then dims) or the larger one (``"max"``: the twin is simply
    dropped). Greedy: pairs are processed brightest-first and an atom merges at most once.
    """
    if amplitude_mode not in ("sum", "max"):
        raise ValueError(f"amplitude_mode must be 'sum' or 'max', got {amplitude_mode!r}")
    pos = positions.detach().cpu().numpy()
    amp = amplitudes.detach().cpu().numpy()
    sig = sigmas.detach().cpu().numpy()
    n = pos.shape[0]
    pairs = cKDTree(pos).query_pairs(float(min_distance), output_type="ndarray")
    merged_into = -np.ones(n, dtype=int)
    new_pos, new_amp, new_sig = pos.copy(), amp.copy(), sig.copy()
    if len(pairs):
        order = np.argsort(-(amp[pairs[:, 0]] + amp[pairs[:, 1]]))
        for a, b in pairs[order]:
            if merged_into[a] >= 0 or merged_into[b] >= 0:
                continue
            keep, drop = (a, b) if amp[a] >= amp[b] else (b, a)
            tot = amp[keep] + amp[drop]
            new_pos[keep] = (amp[keep] * pos[keep] + amp[drop] * pos[drop]) / max(tot, 1e-30)
            new_amp[keep] = tot if amplitude_mode == "sum" else amp[keep]
            merged_into[drop] = keep
    keep_idx = np.where(merged_into < 0)[0]

    def t(a, dtype):
        return torch.as_tensor(a, dtype=dtype, device=positions.device)

    return (
        torch.as_tensor(keep_idx, device=positions.device),
        t(new_pos[keep_idx], positions.dtype),
        t(new_amp[keep_idx], amplitudes.dtype),
        t(new_sig[keep_idx], sigmas.dtype),
    )


def pixel_variance(
    pred: "torch.Tensor",
    intensity_scale: float,
    dose_e_per_A2: float,
    sampling: float,
    min_counts: float = 0.25,
    target: "torch.Tensor | None" = None,
    model_error_frac: float = 0.0,
    target_blur_px: float = 0.0,
) -> "torch.Tensor":
    """Expected shot-noise variance of every pixel, in the dataset's normalised image units.

    A normalised value ``v`` corresponds to ``v * k`` electron counts with
    ``k = intensity_scale * dose * sampling^2`` (the HAADF simulator's convention: saved image =
    counts / (dose * pixel^2), then divided by ``intensity_scale``), so ``Var(v) = v / k``. The
    model prediction stands in for the expectation, floored at ``min_counts`` counts so that
    empty pixels keep a finite variance. The floor must stay well below the counts per pixel
    inside the particle: at 3e3 e/A^2 a pixel holds ~0.7 counts, and a floor of one count made
    the whitened residual's chi^2 0.38 instead of 1 (chapter 06, 2026-10-27 13:50).

    Round 3 options (defaults = the behaviour above): ``target`` given -> the expected counts are
    ``max(pred, target)``, so pixels the model does not explain yet keep their measured variance instead of
    the empty-pixel floor (at 1e8 that floor made the whitened add flood vacuum, S3vm); ``model_error_frac``
    adds ``(frac * pred)^2``, a systematic floor capping the per-pixel SNR at ``1 / frac``. ``target_blur_px`` > 0
    smooths ``target`` with a Gaussian of that width (pixels) before the maximum, so that the shot-noise fluctuations of
    a low-dose image (under one count per pixel at 3e3) do not inflate the variance where the model already fits.
    """
    k = float(intensity_scale) * float(dose_e_per_A2) * float(sampling) ** 2
    if target is not None and target_blur_px > 0:
        target = _blur2d(target, float(target_blur_px))
    expect = pred if target is None else torch.maximum(pred, target)
    var = torch.clamp(expect, min=0.0) / k + float(min_counts) / (k * k)
    if model_error_frac > 0:
        var = var + (float(model_error_frac) * torch.clamp(pred, min=0.0)) ** 2
    return var


def _blur2d(img: "torch.Tensor", sigma_px: float) -> "torch.Tensor":
    """Separable Gaussian blur of a ``(T, H, W)`` stack (reflect padding), width ``sigma_px`` pixels."""
    r = max(1, int(math.ceil(3.0 * sigma_px)))
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    g = torch.exp(-0.5 * (x / sigma_px) ** 2)
    g = g / g.sum()
    out = img.reshape(-1, 1, *img.shape[-2:])
    out = torch.nn.functional.conv2d(torch.nn.functional.pad(out, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1))
    out = torch.nn.functional.conv2d(torch.nn.functional.pad(out, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1))
    return out.reshape(img.shape)


@torch.no_grad()
def atom_support(
    positions: "torch.Tensor",
    amplitudes: "torch.Tensor",
    sigmas: "torch.Tensor",
    rotations: "torch.Tensor",
    shifts: "torch.Tensor",
    sampling: float,
    image_shape: tuple[int, int],
    window_radius: int,
    residual: "torch.Tensor",
    variance: "torch.Tensor",
    tilt_chunk: int | None = None,
) -> "torch.Tensor":
    """Data support of every atom: the loss increase on removing it, in chi-squared units.

    With the squared loss and ``r = target - model`` (model including the atom), removing atom
    ``i`` with stamp ``g_i`` changes the summed squared residual by
    ``sum_pixels g_i (2 r + g_i)``; divided by the expected shot-noise variance per pixel this is
    the chi-squared the atom explains, and its square root is the atom's detection SNR. It is
    computed in closed form from the renderer's own windows (no re-rendering): a few counts of
    noise cannot keep an atom whose support is below a few sigma, and an atom that makes the
    model too bright has negative support. Returns ``(N,)``.
    """
    from quantem.core.ml.splat.renderer import _stamp_offsets, project_positions

    H, W = (int(v) for v in image_shape)
    T = int(rotations.shape[0])
    N = int(positions.shape[0])
    device, dtype = positions.device, positions.dtype
    out = torch.zeros(N, device=device, dtype=dtype)
    if N == 0:
        return out
    sig = sigmas.reshape(-1)
    if sig.shape[0] == 1 and N != 1:
        sig = sig.expand(N)
    norm = amplitudes / (2.0 * math.pi * sig * sig)
    inv2s2 = 0.5 / (sig * sig)
    r = int(window_radius)
    off = _stamp_offsets(r, device)
    n_chunk = T if tilt_chunk is None else max(1, int(tilt_chunk))
    res_flat = residual.reshape(-1)
    var_flat = variance.reshape(-1)
    for t0 in range(0, T, n_chunk):
        t1 = min(T, t0 + n_chunk)
        tc = t1 - t0
        centers = project_positions(positions, rotations[t0:t1], shifts[t0:t1], sampling, (H, W))
        ci = torch.round(centers).long()
        pix = ci[:, :, None, :] + off[None, None, :, :]  # (tc, N, W2, 2)
        d = (pix.to(dtype) - centers[:, :, None, :]) * sampling
        r2 = (d * d).sum(dim=-1)
        g = norm[None, :, None] * torch.exp(-r2 * inv2s2[None, :, None])  # (tc, N, W2)
        valid = (pix[..., 0] >= 0) & (pix[..., 0] < H) & (pix[..., 1] >= 0) & (pix[..., 1] < W)
        prow = pix[..., 0].clamp(0, H - 1)
        pcol = pix[..., 1].clamp(0, W - 1)
        tidx = (torch.arange(tc, device=device) + t0).view(tc, 1, 1)
        flat = (tidx * H + prow) * W + pcol
        rr = res_flat[flat]
        vv = var_flat[flat]
        out += ((g * (2.0 * rr + g) / vv) * valid.to(dtype)).sum(dim=(0, 2))
    return out
