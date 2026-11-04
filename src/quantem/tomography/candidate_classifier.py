"""Graph-network classifiers for adding and removing atoms in the Gaussian-splat fit (chapter 10 of the splat-prior project).

One architecture, two trained instances:

* ``kind="add"``: the nodes are the real atoms and the candidate sites of
  :func:`quantem.tomography.candidate_sites.generate_candidate_sites`; the output is, per candidate, the probability
  that the site holds an atom (1 = a missing atom, 0 = an empty site, e.g. a relaxed vacancy or a site outside the
  surface). Candidates receive messages and do not send them: every edge whose sender is a candidate is dropped, so the
  real atoms' states never depend on the candidates.
* ``kind="remove"``: the nodes are the atoms only; the output is, per atom, the probability that it is a real atom
  (0 = a spurious atom, e.g. one inserted between lattice sites or one put back into a relaxed vacancy).

The network is a MeshGraphNets-shaped message-passing net in plain torch: encoder MLPs for the node and edge inputs,
``num_convs`` residual convolutions (edge update ``e <- e + MLP([h_send, h_recv, e])``, node update
``h <- h + MLP([h, sum of e into the node])``, both MLPs ending in LayerNorm, edge states carried across layers),
a decoder MLP to one logit per node.

Inputs are scale-free: positions are divided by ``d1`` (the training set's median first-shell distance, stored in the
checkpoint; a caller may pass another). Edges join nodes closer than ``cutoff * d1``; edge input = (dx / d1, |dx| / d1)
with dx = x_receiver - x_sender; node input = one-hot type (real, candidate) and coordination / ``coord_norm``, the
number of real atoms within ``coord_radius * d1``. The featurizer :meth:`CandidateClassifier.featurize` is the one
code path used in training and at inference.

The network is not rotation-equivariant (dx enters as a vector); it is trained with a fresh random rotation of every
sample (:func:`random_rotations`), and inference is deterministic (no rotation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from quantem.tomography.energy_prior import neighbour_list_torch

__all__ = [
    "ClassifierConfig",
    "Graph",
    "CandidateClassifier",
    "random_rotations",
]

KINDS = ("add", "remove")


@dataclass
class ClassifierConfig:
    """Architecture and featurizer settings (saved in the checkpoint). Lengths are in units of ``d1`` except ``d1``."""

    kind: str = "add"
    d1: float = 2.8545  # Angstrom; median first-shell distance of the training set (au_cand_v1)
    cutoff: float = 1.75  # edge cutoff / d1 (5.0 A for gold)
    coord_radius: float = 1.2  # coordination counts real atoms within this / d1
    coord_norm: float = 12.0  # coordination is divided by this
    dim: int = 128  # node and edge state width
    num_convs: int = 6
    mlp_depth: int = 3  # linear layers per MLP (graphite's MeshGraphNetsConv uses 3)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        for name in ("d1", "cutoff", "coord_radius", "coord_norm"):
            setattr(
                self, name, float(getattr(self, name))
            )  # plain floats (numpy scalars break weights_only loads)
        for name in ("dim", "num_convs", "mlp_depth"):
            setattr(self, name, int(getattr(self, name)))


class Graph(NamedTuple):
    """A featurized graph (or a disjoint batch of graphs): node inputs (N, 3), directed edges (2, E) as
    (sender, receiver), edge inputs (E, 4)."""

    node_attr: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor


def _mlp(dims: list[int], layer_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    if layer_norm:
        layers.append(nn.LayerNorm(dims[-1]))
    return nn.Sequential(*layers)


class MGNConv(nn.Module):
    """MeshGraphNets convolution (as graphite's ``MeshGraphNetsConv``): residual edge then node update, sum
    aggregation at the receiver ``edge_index[1]``."""

    def __init__(self, dim: int, depth: int = 3):
        super().__init__()
        self.edge_mlp = _mlp([3 * dim] + [dim] * depth, layer_norm=True)
        self.node_mlp = _mlp([2 * dim] + [dim] * depth, layer_norm=True)

    def forward(
        self, h: torch.Tensor, edge_index: torch.Tensor, e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s, r = edge_index[0], edge_index[1]
        e = e + self.edge_mlp(torch.cat([h[s], h[r], e], dim=-1))
        agg = torch.zeros_like(h).index_add_(0, r, e)
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h, e


def random_rotations(
    n: int,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``n`` rotation matrices (n, 3, 3) drawn uniformly (Haar) from normalised Gaussian quaternions."""
    q = torch.randn(n, 4, generator=generator, dtype=torch.float64, device="cpu")
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=1,
    ).reshape(n, 3, 3)
    return R.to(device=device, dtype=dtype)


def _as_tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x))
    return x.to(device=device, dtype=dtype)


class CandidateClassifier(nn.Module):
    """Per-node existence classifier; see the module docstring. Build with a :class:`ClassifierConfig` or load a
    trained one with :meth:`from_checkpoint`."""

    def __init__(self, config: ClassifierConfig | None = None, **overrides):
        super().__init__()
        cfg = config if config is not None else ClassifierConfig()
        if overrides:
            cfg = ClassifierConfig(**{**asdict(cfg), **overrides})
        self.config = cfg
        d = cfg.dim
        self.node_encoder = _mlp([3] + [d] * cfg.mlp_depth, layer_norm=True)
        self.edge_encoder = _mlp([4] + [d] * cfg.mlp_depth, layer_norm=True)
        self.convs = nn.ModuleList([MGNConv(d, cfg.mlp_depth) for _ in range(cfg.num_convs)])
        self.decoder = _mlp([d] * cfg.mlp_depth + [1], layer_norm=False)
        self.checkpoint: str | None = None

    # region --- properties ---
    @property
    def kind(self) -> str:
        return self.config.kind

    @property
    def d1(self) -> float:
        return self.config.d1

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # endregion

    # region --- featurizer (shared by training and inference) ---
    def featurize(
        self,
        positions: torch.Tensor,
        is_candidate: torch.Tensor | None = None,
        d1: float | None = None,
    ) -> Graph:
        """Graph of one structure: ``positions`` (N, 3) Angstrom (atoms and, for the add model, candidates),
        ``is_candidate`` (N,) bool (None = no candidates). ``d1`` overrides the checkpoint's value."""
        cfg = self.config
        d1 = cfg.d1 if d1 is None else float(d1)
        x = positions / d1
        n = x.shape[0]
        if is_candidate is None:
            is_candidate = torch.zeros(n, dtype=torch.bool, device=x.device)
        if cfg.kind == "remove" and bool(is_candidate.any()):
            raise ValueError("the remove model takes atoms only (is_candidate must be all False)")
        ei = neighbour_list_torch(x, cfg.cutoff)
        s, r = ei[0], ei[1]
        rel = x[r] - x[s]
        dist = rel.norm(dim=-1)
        real_sender = ~is_candidate[s]
        coord = torch.zeros(n, dtype=x.dtype, device=x.device)
        near = real_sender & (dist < cfg.coord_radius)
        coord.index_add_(0, r[near], torch.ones_like(dist[near]))
        keep = real_sender  # candidates receive messages, never send them
        edge_index = ei[:, keep]
        edge_attr = torch.cat([rel[keep], dist[keep, None]], dim=-1)
        cand = is_candidate.to(x.dtype)
        node_attr = torch.stack([1.0 - cand, cand, coord / cfg.coord_norm], dim=-1)
        return Graph(node_attr, edge_index, edge_attr)

    @staticmethod
    def collate(graphs: list[Graph]) -> Graph:
        """One disjoint graph from several (edge indices offset by the preceding node counts)."""
        offs, o = [], 0
        for g in graphs:
            offs.append(o)
            o += g.node_attr.shape[0]
        return Graph(
            torch.cat([g.node_attr for g in graphs]),
            torch.cat([g.edge_index + off for g, off in zip(graphs, offs)], dim=1),
            torch.cat([g.edge_attr for g in graphs]),
        )

    # endregion

    # region --- network ---
    def forward_graph(self, graph: Graph, return_hidden: bool = False):
        """Logits (N,) of a featurized graph; with ``return_hidden`` also the final node states (N, dim)."""
        h = self.node_encoder(graph.node_attr)
        e = self.edge_encoder(graph.edge_attr)
        for conv in self.convs:
            h, e = conv(h, graph.edge_index, e)
        logits = self.decoder(h).squeeze(-1)
        return (logits, h) if return_hidden else logits

    def forward(
        self,
        positions: torch.Tensor,
        is_candidate: torch.Tensor | None = None,
        d1: float | None = None,
        return_hidden: bool = False,
    ):
        return self.forward_graph(self.featurize(positions, is_candidate, d1), return_hidden)

    # endregion

    # region --- inference API ---
    @torch.no_grad()
    def score_candidates(self, atoms, candidates, d1: float | None = None) -> torch.Tensor:
        """Add model: probability (C,) that each candidate site holds an atom, for ``atoms`` (N, 3) and
        ``candidates`` (C, 3) in Angstrom (numpy or torch, any device; the result is on the atoms' device, float32)."""
        if self.kind != "add":
            raise ValueError("score_candidates needs the add model (kind='add')")
        out_dev = atoms.device if isinstance(atoms, torch.Tensor) else torch.device("cpu")
        dev = self.device
        a = _as_tensor(atoms, dev)
        c = _as_tensor(candidates, dev).reshape(-1, 3)
        n = a.shape[0]
        is_c = torch.zeros(n + c.shape[0], dtype=torch.bool, device=dev)
        is_c[n:] = True
        logits = self.forward(torch.cat([a, c]), is_c, d1)
        return torch.sigmoid(logits[n:]).to(out_dev)

    @torch.no_grad()
    def score_atoms(self, atoms, d1: float | None = None) -> torch.Tensor:
        """Remove model: probability (N,) that each atom of ``atoms`` (N, 3) Angstrom is real."""
        if self.kind != "remove":
            raise ValueError("score_atoms needs the remove model (kind='remove')")
        out_dev = atoms.device if isinstance(atoms, torch.Tensor) else torch.device("cpu")
        a = _as_tensor(atoms, self.device)
        return torch.sigmoid(self.forward(a, None, d1)).to(out_dev)

    # endregion

    # region --- checkpoints ---
    def save_checkpoint(self, path: str | Path, extra: dict | None = None) -> None:
        """``{"config", "state_dict", "extra"}`` with torch.save (loadable with ``weights_only=True``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        torch.save(
            {"config": asdict(self.config), "state_dict": state, "extra": extra or {}}, path
        )

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu", **overrides
    ) -> "CandidateClassifier":
        """Load a checkpoint written by :meth:`save_checkpoint` (the training script saves the EMA weights), frozen,
        in eval mode, on ``device``. ``overrides`` replace config fields that do not change the weights' shapes
        (e.g. ``d1``)."""
        blob = torch.load(Path(path), map_location="cpu", weights_only=True)
        cfg = ClassifierConfig(**{**blob["config"], **overrides})
        model = cls(cfg)
        model.load_state_dict(blob["state_dict"])
        model = model.eval().to(torch.device(device))
        for p in model.parameters():
            p.requires_grad_(False)
        model.checkpoint = str(path)
        return model

    # endregion

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"kind={c.kind}, d1={c.d1:.4f} A, cutoff={c.cutoff} d1, coord_radius={c.coord_radius} d1, "
            f"dim={c.dim}, num_convs={c.num_convs}, params={sum(p.numel() for p in self.parameters()):,}"
        )
