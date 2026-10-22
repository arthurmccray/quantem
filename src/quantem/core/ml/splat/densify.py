"""Add / remove Gaussians while keeping the optimizer state consistent.

The object model holds one torch optimizer with one param group per key (``positions``,
``amplitudes``, ``sigmas``, ...). When atoms are pruned, merged or added, every per-atom
Parameter is rebuilt and the Adam moments are re-indexed with :func:`remap_optimizer_state`
(rows of kept atoms follow their atom, new atoms start from zero moments), so a densification
step never resets the optimisation of the surviving atoms.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

__all__ = ["remap_optimizer_state", "prune_mask", "merge_close_atoms"]


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
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Merge pairs of atoms closer than ``min_distance`` (brightest wins, amplitudes add).

    Returns ``(keep_idx, positions, amplitudes, sigmas)`` where the merged atom sits at the
    amplitude-weighted mean position and carries the summed amplitude and the brighter
    atom's sigma. Greedy: pairs are processed brightest-first and an atom merges at most once.
    """
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
            new_amp[keep] = tot
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
