"""Learned atomic prior: wraps a ``tricor.flowmatch`` velocity model as a clean-position
estimator for Gaussian-splat tomography.

Training convention (the ``JitterFlowMatchDataset`` in ``~/quantem-notebooks/splat/atom_prior``):
``x_t = x_1 - (1 - t) * sigma_max * eps`` with clean data at ``t = 1`` and the velocity target
``v = x_1 - x_0 = -sigma_max * eps``; the clean estimate is ``x_hat = x_t + (1 - t) * v_hat`` and
the noise level at time ``t`` is ``sigma_t = (1 - t) * sigma_max``.

The model only reads edge vectors (periodic minimum-image, cutoff ``cutoff``), so positions are
never wrapped: the atoms are placed in an orthorhombic box padded by ``pad >= cutoff + 3 sigma_max``
of vacuum, which makes the periodic graph a free-space graph. ``tricor``, ``graphite`` and
``lightning`` are imported lazily; without them the constructor raises an ``ImportError`` that
says how to install them.
"""

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

__all__ = ["AtomPrior"]

_INSTALL_HINT = (
    "the learned atomic prior needs tricor (editable from ~/atomode-spec-v1), graphite, "
    "torch_geometric and lightning; use the splat env (source ~/quantem-notebooks/splat/env.sh)"
)


def _lazy_graph_builders():
    try:
        from tricor.flowmatch.flow_utils import (
            periodic_radius_graph_cell_list,
            periodic_radius_graph_chunked,
        )
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(_INSTALL_HINT) from e
    return periodic_radius_graph_cell_list, periodic_radius_graph_chunked


def _build_graph(x: "torch.Tensor", cutoff: float, cell: "torch.Tensor"):
    cell_list, chunked = _lazy_graph_builders()
    box = torch.diagonal(cell)
    if bool((box >= 3.0 * cutoff).all()):
        return cell_list(x, cutoff, cell)
    return chunked(x, cutoff, cell)


class AtomPrior:
    """
    Clean-position estimator built on a trained unconditional flow-matching velocity model.

    Parameters
    ----------
    model : callable
        ``model(z, edge_index, edge_attr, t) -> (N, 3)`` velocity in Angstrom (the EMA module).
    sigma_max : float
        Noise scale the model was trained with (Angstrom).
    cutoff : float
        Graph cutoff (Angstrom), identical to training.
    species : list[int]
        Atomic numbers in the model's one-hot order (default gold only).
    pad : float or None
        Vacuum padding around the atoms; default ``cutoff + 3 sigma_max``.
    device : str
    t_range : tuple[float, float]
        Clamp for ``t`` (the model never saw ``t = 1`` exactly).
    """

    def __init__(
        self,
        model: Callable,
        *,
        sigma_max: float,
        cutoff: float = 5.0,
        species: list[int] | None = None,
        pad: float | None = None,
        device: str | torch.device = "cpu",
        t_range: tuple[float, float] = (0.02, 0.98),
    ):
        self.model = model
        self.sigma_max = float(sigma_max)
        self.cutoff = float(cutoff)
        self.species = [79] if species is None else [int(z) for z in species]
        self.pad = float(pad) if pad is not None else self.cutoff + 3.0 * self.sigma_max
        self.device = torch.device(device)
        self.t_range = (float(t_range[0]), float(t_range[1]))

    # region --- construction ---
    @classmethod
    def from_checkpoint(
        cls,
        ckpt: str | Path,
        *,
        config_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        use_ema: bool = True,
        **overrides,
    ) -> "AtomPrior":
        """Load a ``LitUncondFlowMatch`` Lightning checkpoint and the ``config.json`` written by
        the training script (``sigma_max``, ``cutoff``, ``species``)."""
        try:
            from tricor.flowmatch.velocity_model_uncond import LitUncondFlowMatch
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(_INSTALL_HINT) from e
        ckpt = Path(ckpt)
        cfg_path = Path(config_path) if config_path is not None else _find_config(ckpt)
        cfg = json.loads(cfg_path.read_text()) if cfg_path is not None else {}
        lit = LitUncondFlowMatch.load_from_checkpoint(str(ckpt), map_location="cpu")
        module = lit.ema_model.module if use_ema else lit.model
        module = module.eval().to(torch.device(device))
        for p in module.parameters():
            p.requires_grad_(False)
        kw = {
            "sigma_max": cfg.get("sigma_max"),
            "cutoff": cfg.get("cutoff", 5.0),
            "species": cfg.get("species", [79]),
        }
        kw.update(overrides)
        if kw["sigma_max"] is None:
            raise ValueError("sigma_max not found in config.json; pass sigma_max=... explicitly")
        prior = cls(module, device=device, **kw)
        prior.checkpoint = str(ckpt)
        return prior

    # endregion

    # region --- time / noise mapping ---
    def t_of_sigma(self, sigma: float) -> float:
        """Flow time for a noise level in Angstrom: ``t = 1 - sigma / sigma_max`` (clamped)."""
        t = 1.0 - float(sigma) / self.sigma_max
        return float(min(max(t, self.t_range[0]), self.t_range[1]))

    def sigma_of_t(self, t: float) -> float:
        return (1.0 - float(t)) * self.sigma_max

    def _clamp_t(self, t: float | None) -> float:
        if t is None:
            t = 0.5
        return float(min(max(float(t), self.t_range[0]), self.t_range[1]))

    # endregion

    # region --- inference ---
    def _box(self, p: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        lo = p.min(dim=0).values - self.pad
        extent = p.max(dim=0).values - p.min(dim=0).values + 2.0 * self.pad
        box = torch.clamp(extent, min=3.0 * self.cutoff + 1e-3)
        return lo, torch.diag(box)

    def _onehot(self, n: int, numbers: "torch.Tensor | None") -> "torch.Tensor":
        ns = len(self.species)
        if numbers is None:
            z = torch.zeros(n, ns, device=self.device)
            z[:, 0] = 1.0
            return z
        idx = torch.as_tensor(
            [self.species.index(int(v)) for v in numbers.tolist()], device=self.device
        )
        return torch.nn.functional.one_hot(idx, ns).float()

    @torch.no_grad()
    def velocity(
        self,
        positions_A: "torch.Tensor",
        t: float | None = None,
        *,
        numbers: "torch.Tensor | None" = None,
        rotation: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Model velocity ``v_hat`` (Angstrom) at the given positions treated as ``x_t``."""
        t = self._clamp_t(t)
        p = positions_A.detach().to(self.device, torch.float32)
        n = p.shape[0]
        lo, cell = self._box(p)
        ei, ev = _build_graph(p - lo, self.cutoff, cell)
        if rotation is not None:
            ev = ev @ rotation.to(ev)
        ea = torch.hstack([ev, ev.norm(dim=-1, keepdim=True)])
        tt = torch.full((n, 1), t, device=self.device)
        v = self.model(self._onehot(n, numbers), ei, ea, tt)
        if rotation is not None:
            v = v @ rotation.to(v).T
        return v

    @torch.no_grad()
    def clean_estimate(
        self,
        positions_A: "torch.Tensor",
        t: float | None = None,
        *,
        sigma: float | None = None,
        n_draws: int = 1,
        noise_scale: float = 1.0,
        n_rot: int = 1,
        numbers: "torch.Tensor | None" = None,
        generator: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        """Clean-structure estimate ``x_hat`` for the current positions.

        ``x_t = x + noise_scale * (1 - t) * sigma_max * eps`` (the training corruption; ``noise_scale=0``
        treats ``x`` itself as the noisy state), ``x_hat = x_t + (1 - t) * v_hat``, averaged over
        ``n_draws`` noise draws and ``n_rot`` random rotations (test-time augmentation for the
        non-equivariant network). Returned in the input frame and dtype.
        """
        t = self.t_of_sigma(sigma) if sigma is not None else self._clamp_t(t)
        x = positions_A.detach().to(self.device, torch.float32)
        acc = torch.zeros_like(x)
        count = 0
        for _ in range(max(1, int(n_draws))):
            for k in range(max(1, int(n_rot))):
                eps = torch.randn(x.shape, device=self.device, generator=generator)
                x_t = x + float(noise_scale) * (1.0 - t) * self.sigma_max * eps
                rot = None if (k == 0 and n_rot == 1) else _random_rotation(self.device, generator)
                v = self.velocity(x_t, t, numbers=numbers, rotation=rot)
                acc += x_t + (1.0 - t) * v
                count += 1
        return (acc / count).to(positions_A.dtype).to(positions_A.device)

    @torch.no_grad()
    def velocity_rmsd(
        self,
        positions_A: "torch.Tensor",
        t: float | None = None,
        pos_ref: "torch.Tensor | None" = None,
        **kw,
    ) -> dict[str, float]:
        """Diagnostics: rms ``|v_hat|``, rms implied step ``(1 - t) |v_hat|`` and, if a reference is
        given, the index-matched RMSD of ``x_hat`` to it."""
        t = self._clamp_t(t)
        v = self.velocity(positions_A, t, **kw)
        out = {
            "t": t,
            "sigma_t": self.sigma_of_t(t),
            "rms_v": float(v.norm(dim=-1).pow(2).mean().sqrt()),
        }
        out["rms_step"] = (1.0 - t) * out["rms_v"]
        if pos_ref is not None:
            x_hat = self.clean_estimate(positions_A, t, noise_scale=0.0)
            d = (x_hat - pos_ref.to(x_hat)).norm(dim=-1)
            out["rmsd_to_ref"] = float(d.pow(2).mean().sqrt())
        return out

    # endregion


def _find_config(ckpt: Path) -> Path | None:
    for d in (ckpt.parent, ckpt.parent.parent):
        c = d / "config.json"
        if c.exists():
            return c
    return None


def _random_rotation(device, generator=None) -> "torch.Tensor":
    """Uniform random rotation matrix (Shoemake, via a random unit quaternion)."""
    q = torch.randn(4, device=device, generator=generator)
    q = q / q.norm()
    w, x, y, z = q.tolist()
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        device=device,
        dtype=torch.float32,
    )


def lattice_sites_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Index-matched RMSD helper used by the validation scripts."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(math.sqrt((d * d).sum(axis=1).mean()))
