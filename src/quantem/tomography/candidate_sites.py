"""Candidate lattice sites implied by the atoms already present (chapter 10 of the splat-prior project).

Given the positions of a crystal-like set of atoms, :func:`generate_candidate_sites` proposes the empty sites
that the local structure implies, with nothing fcc- or gold-specific hard-coded: the first-shell distance,
the neighbour cutoff and the full coordination are all measured from the atoms, and every proposed site is a
translation of a neighbour's own shell onto an under-coordinated atom, so twins, grain boundaries and other
lattices are handled by the same rule. The candidates are consumed by the candidate classifier (the add
proposal inside the bookkeeping, the decoys of its training set) and, through :func:`empty_interior_sites`,
give the vacancy list of a structure whose truth records none.

The rule (fixed by the chapter-10 plan):

1. ``d1`` = the median nearest-neighbour distance; neighbours = atoms closer than ``r_nn = nn_factor * d1``;
   ``c_max`` = the structure's full coordination = a high quantile of the per-atom neighbour count.
2. For every atom ``i`` with fewer than ``c_max`` neighbours: proposals ``x_i + (x_k - x_j)`` for every
   neighbour ``j`` of ``i`` and every neighbour ``k`` of ``j`` (the neighbour's own shell translated onto
   ``i``); ``k = i`` is the bond reflection ``x_i - (x_j - x_i)``, so the reflections are included.
3. A proposal survives only if it is farther than ``min_dist_factor * d1`` from every atom. The survivors
   are clustered at ``link_distance`` (single linkage, cluster position = mean, support = number of
   proposals); clusters with support below ``min_support`` are dropped, and at most ``max_frac`` times the
   number of under-coordinated atoms are kept, highest support first.
4. Optional extra positions (residual peaks) are merged at the same distance: one within ``link_distance``
   of a cluster joins it and sets its ``from_residual`` flag; the others become candidates of support 0.

Everything runs in torch on the positions' device (chunked brute force; no python loop over atoms), so the
bookkeeping can call it on the GPU every densify pass; numpy input is accepted and gives numpy output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from quantem.tomography.energy_prior import neighbour_list_torch

__all__ = ["CandidateParams", "CandidateSites", "generate_candidate_sites", "empty_interior_sites"]


@dataclass
class CandidateParams:
    """Parameters of :func:`generate_candidate_sites` (defaults = the chapter-10 plan)."""

    nn_factor: float = 1.2  # neighbours are atoms closer than nn_factor * d1
    min_dist_factor: float = 0.8  # a proposal closer than this * d1 to any atom is discarded
    link_distance: float = 1.0  # Angstrom; proposals closer than this are one site
    min_support: int = 2  # clusters with fewer proposals are dropped
    max_frac: float = 1.5  # keep at most max_frac * (number of under-coordinated atoms) clusters
    c_max_quantile: float = 0.99  # c_max = this quantile of the neighbour counts
    chunk: int = 32768  # proposals per chunk of the distance filter
    d1: float | None = (
        None  # Angstrom; first-shell distance given by the caller instead of measured (None = measured)
    )


@dataclass
class CandidateSites:
    """Output of :func:`generate_candidate_sites`: ``C`` candidates and the structure measurements behind them.

    Arrays are torch tensors on the input device, or numpy arrays when the input was numpy.
    """

    positions: torch.Tensor | np.ndarray  # (C, 3) Angstrom, the atoms' frame
    support: torch.Tensor | np.ndarray  # (C,) proposals in the cluster (0 for an unmatched extra)
    coordination: torch.Tensor | np.ndarray  # (C,) atoms within r_nn of the candidate
    from_residual: torch.Tensor | np.ndarray  # (C,) bool
    d1: float  # median nearest-neighbour distance, Angstrom
    r_nn: float  # neighbour cutoff, Angstrom
    c_max: int  # the structure's full coordination
    atom_coordination: torch.Tensor | np.ndarray  # (N,) neighbours of every atom
    n_edges: int  # directed neighbour pairs within r_nn
    n_under: int  # atoms with coordination < c_max
    n_proposals: int  # proposals before the distance filter
    n_kept_proposals: int  # proposals that passed the distance filter
    n_clusters: int  # clusters before the support and count cuts
    params: CandidateParams = field(default_factory=CandidateParams)

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def to_numpy(self) -> CandidateSites:
        """The same result with numpy arrays (a no-op when it already holds numpy arrays)."""
        if isinstance(self.positions, np.ndarray):
            return self

        def _np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy()

        return CandidateSites(
            positions=_np(self.positions),
            support=_np(self.support),
            coordination=_np(self.coordination),
            from_residual=_np(self.from_residual),
            d1=self.d1,
            r_nn=self.r_nn,
            c_max=self.c_max,
            atom_coordination=_np(self.atom_coordination),
            n_edges=self.n_edges,
            n_under=self.n_under,
            n_proposals=self.n_proposals,
            n_kept_proposals=self.n_kept_proposals,
            n_clusters=self.n_clusters,
            params=self.params,
        )


# --- building blocks -------------------------------------------------------------------------------------


def nearest_neighbour_distances(x: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Distance from every atom to its nearest other atom, ``(N,)``, by chunked brute force (``inf`` for N = 1)."""
    n = x.shape[0]
    out = torch.full((n,), math.inf, dtype=x.dtype, device=x.device)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        d = (x[i0:i1, None, :] - x[None, :, :]).norm(dim=-1)
        rows = torch.arange(i1 - i0, device=x.device)
        d[rows, rows + i0] = math.inf
        out[i0:i1] = d.min(dim=1).values
    return out


def count_within(
    points: torch.Tensor, x: torch.Tensor, radius: float, chunk: int = 1024
) -> torch.Tensor:
    """Number of atoms ``x`` closer than ``radius`` to every point, ``(P,)`` (chunked brute force)."""
    out = torch.zeros(points.shape[0], dtype=torch.long, device=x.device)
    for i0 in range(0, points.shape[0], chunk):
        d = (points[i0 : i0 + chunk, None, :] - x[None, :, :]).norm(dim=-1)
        out[i0 : i0 + chunk] = (d < radius).sum(dim=1)
    return out


def _csr(index: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort ``index`` (row ids of edges) into CSR form: ``(order, offsets (n + 1,), degree (n,))``."""
    order = torch.argsort(index, stable=True)
    deg = torch.bincount(index, minlength=n)
    offsets = torch.zeros(n + 1, dtype=torch.long, device=index.device)
    offsets[1:] = torch.cumsum(deg, dim=0)
    return order, offsets, deg


def _expand_ranges(
    starts: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(source row of every element, start + 0..count-1)`` for the concatenation of the ranges."""
    total = int(counts.sum())
    rows = torch.repeat_interleave(torch.arange(counts.shape[0], device=counts.device), counts)
    first = torch.cumsum(counts, dim=0) - counts
    local = torch.arange(total, device=counts.device) - first[rows]
    return rows, starts[rows] + local


def _grid_merge(
    x: torch.Tensor, w: torch.Tensor, cell: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge points that fall into the same cubic cell of side ``cell``: weighted mean position and summed
    weight per occupied cell, in cell-key order (deterministic)."""
    ijk = torch.floor(x / cell).to(torch.long)
    ijk = ijk - ijk.min(dim=0).values
    span = ijk.max(dim=0).values + 1
    key = (ijk[:, 0] * span[1] + ijk[:, 1]) * span[2] + ijk[:, 2]
    _, inv = torch.unique(key, return_inverse=True)
    k = int(inv.max()) + 1 if inv.numel() else 0
    wsum = torch.zeros(k, dtype=x.dtype, device=x.device).index_add_(0, inv, w)
    xsum = torch.zeros(k, 3, dtype=x.dtype, device=x.device).index_add_(0, inv, x * w[:, None])
    return xsum / wsum[:, None], wsum


def _link_pairs(
    x: torch.Tensor, radius: float, chunk: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairs ``(a, b)`` with ``a < b`` closer than ``radius`` (chunked brute force)."""
    a_all, b_all = [], []
    n = x.shape[0]
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        d = (x[i0:i1, None, :] - x[None, :, :]).norm(dim=-1)
        a, b = (d < radius).nonzero(as_tuple=True)
        a = a + i0
        keep = a < b
        a_all.append(a[keep])
        b_all.append(b[keep])
    return torch.cat(a_all), torch.cat(b_all)


def _components(n: int, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Connected-component label (the smallest member index) of ``n`` nodes under the links ``(a, b)``, by
    min-label propagation until stable."""
    labels = torch.arange(n, device=a.device)
    while True:
        new = labels.clone()
        new.scatter_reduce_(0, a, labels[b], reduce="amin")
        new.scatter_reduce_(0, b, labels[a], reduce="amin")
        new = new[new]  # pointer jumping: one hop toward the root per sweep
        if torch.equal(new, labels):
            return labels
        labels = new


def cluster_points(
    x: torch.Tensor, w: torch.Tensor | None, link_distance: float, max_rounds: int = 50
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-linkage clustering at ``link_distance``: weighted mean position and total weight per cluster.

    Points are first merged per cubic cell of side ``link_distance / sqrt(3)`` (anything sharing a cell is
    within ``link_distance``), then the cell means are linked pairwise and merged, repeated until no two
    clusters are closer than ``link_distance``; so the output clusters are pairwise at least ``link_distance``
    apart. Deterministic: clusters come out in cell-key order.
    """
    if w is None:
        w = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
    if x.shape[0] == 0:
        return x.reshape(0, 3), w.reshape(0)
    xc, wc = _grid_merge(x, w, link_distance / math.sqrt(3.0))
    for _ in range(max_rounds):
        a, b = _link_pairs(xc, link_distance)
        if a.numel() == 0:
            break
        lab = _components(xc.shape[0], a, b)
        _, inv = torch.unique(lab, return_inverse=True)
        k = int(inv.max()) + 1
        wsum = torch.zeros(k, dtype=xc.dtype, device=xc.device).index_add_(0, inv, wc)
        xsum = torch.zeros(k, 3, dtype=xc.dtype, device=xc.device).index_add_(
            0, inv, xc * wc[:, None]
        )
        xc, wc = xsum / wsum[:, None], wsum
    return xc, wc


# --- the generator ---------------------------------------------------------------------------------------


def generate_candidate_sites(
    positions: torch.Tensor | np.ndarray,
    extra: torch.Tensor | np.ndarray | None = None,
    params: CandidateParams | None = None,
    **overrides,
) -> CandidateSites:
    """Empty lattice sites implied by ``positions`` ``(N, 3)`` in Angstrom (see the module docstring).

    ``extra`` ``(M, 3)``: positions proposed by something else (residual peaks), merged in at
    ``link_distance`` and flagged ``from_residual``. ``params`` or keyword overrides of
    :class:`CandidateParams` set the rule. Numpy input gives numpy output; torch input stays on its device.
    """
    p = params if params is not None else CandidateParams()
    if overrides:
        p = CandidateParams(**{**p.__dict__, **overrides})
    as_numpy = isinstance(positions, np.ndarray)
    x = torch.as_tensor(np.asarray(positions) if as_numpy else positions)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        x = x.to(torch.float64)
    x = x.detach()
    dev, dt = x.device, x.dtype
    n = int(x.shape[0])
    if n < 2:
        raise ValueError("at least two atoms are needed to measure the nearest-neighbour distance")
    xe = None
    if extra is not None:
        xe = torch.as_tensor(np.asarray(extra) if isinstance(extra, np.ndarray) else extra)
        xe = xe.detach().to(device=dev, dtype=dt).reshape(-1, 3)

    # 1. measured structure: d1, the neighbour list, the coordination and c_max. The median nearest-neighbour
    #    distance is biased low when every atom carries a position error above ~0.15 A per coordinate (the minimum
    #    over 12 noisy neighbour distances); a caller that knows d1 better (a training set, a fit that measured the
    #    first shell from the pair-distance peak) passes it through ``params.d1``.
    d1 = float(p.d1) if p.d1 is not None else float(nearest_neighbour_distances(x).median())
    r_nn = p.nn_factor * d1
    r_min = p.min_dist_factor * d1
    r_wide = (
        r_nn + r_min + 1e-3
    )  # an atom within r_min of a proposal of atom i is within r_nn + r_min of i
    edges = neighbour_list_torch(x, r_wide)  # (2, E) sender, receiver; symmetric, no self edges
    e_i, e_j = edges[1], edges[0]  # j is a neighbour of i
    e_d = (x[e_i] - x[e_j]).norm(dim=-1)
    near = e_d < r_nn
    nn_i, nn_j = e_i[near], e_j[near]
    coord = torch.bincount(nn_i, minlength=n)
    c_max = int(torch.quantile(coord.to(torch.float64), p.c_max_quantile, interpolation="higher"))
    under = coord < c_max
    n_under = int(under.sum())

    # 2. proposals: every neighbour j of an under-coordinated i, every neighbour k of j (k = i included: the
    #    bond reflection), proposal = x_i + x_k - x_j
    order, offsets, deg = _csr(nn_i, n)
    nn_j_sorted = nn_j[order]  # neighbours of atom a are nn_j_sorted[offsets[a]:offsets[a + 1]]
    sel = under[nn_i]
    i_sel, j_sel = nn_i[sel], nn_j[sel]
    rows, slots = _expand_ranges(offsets[j_sel], deg[j_sel])
    i_rep, j_rep, k_rep = i_sel[rows], j_sel[rows], nn_j_sorted[slots]
    prop = x[i_rep] + x[k_rep] - x[j_rep]
    n_proposals = int(prop.shape[0])

    # 3a. distance filter against the atoms within r_wide of the origin atom i (and i itself)
    w_order, w_offsets, w_deg = _csr(e_i, n)
    e_j_sorted = e_j[w_order]
    d_max = int(w_deg.max()) if w_deg.numel() else 0
    wide = torch.full((n, d_max + 1), -1, dtype=torch.long, device=dev)
    wide[:, 0] = torch.arange(n, device=dev)
    w_rows, w_slots = _expand_ranges(w_offsets[:-1], w_deg)
    wide[w_rows, w_slots - w_offsets[:-1][w_rows] + 1] = e_j_sorted[w_slots]
    keep = torch.zeros(n_proposals, dtype=torch.bool, device=dev)
    for i0 in range(0, n_proposals, p.chunk):
        i1 = min(i0 + p.chunk, n_proposals)
        idx = wide[i_rep[i0:i1]]
        d = (x[idx.clamp(min=0)] - prop[i0:i1, None, :]).norm(dim=-1)
        d = torch.where(idx >= 0, d, torch.full_like(d, math.inf))
        keep[i0:i1] = d.min(dim=1).values > r_min
    prop = prop[keep]
    n_kept = int(prop.shape[0])

    # 3b. cluster, support cut, count cap
    cpos, csup = cluster_points(prop, None, p.link_distance)
    n_clusters = int(cpos.shape[0])
    csup = torch.round(csup).to(torch.long)
    ok = csup >= p.min_support
    cpos, csup = cpos[ok], csup[ok]
    n_keep = int(math.floor(p.max_frac * n_under))
    if cpos.shape[0] > n_keep:
        rank = torch.argsort(-csup, stable=True)[:n_keep]
        cpos, csup = cpos[rank], csup[rank]
    flag = torch.zeros(cpos.shape[0], dtype=torch.bool, device=dev)

    # 4. merge the extra positions
    if xe is not None and xe.shape[0] > 0:
        if cpos.shape[0] > 0:
            d = (xe[:, None, :] - cpos[None, :, :]).norm(dim=-1)
            dmin, nearest = d.min(dim=1)
            joined = dmin < p.link_distance
            flag[nearest[joined]] = True
            xe = xe[~joined]
        if xe.shape[0] > 0:
            xe_pos, _ = cluster_points(xe, None, p.link_distance)
            cpos = torch.cat([cpos, xe_pos], dim=0)
            csup = torch.cat([csup, torch.zeros(xe_pos.shape[0], dtype=torch.long, device=dev)])
            flag = torch.cat([flag, torch.ones(xe_pos.shape[0], dtype=torch.bool, device=dev)])

    ccoord = (
        count_within(cpos, x, r_nn)
        if cpos.shape[0]
        else torch.zeros(0, dtype=torch.long, device=dev)
    )
    out = CandidateSites(
        positions=cpos,
        support=csup,
        coordination=ccoord,
        from_residual=flag,
        d1=d1,
        r_nn=r_nn,
        c_max=c_max,
        atom_coordination=coord,
        n_edges=int(nn_i.shape[0]),
        n_under=n_under,
        n_proposals=n_proposals,
        n_kept_proposals=n_kept,
        n_clusters=n_clusters,
        params=p,
    )
    return out.to_numpy() if as_numpy else out


def empty_interior_sites(
    positions: torch.Tensor | np.ndarray,
    min_coordination: int = 10,
    params: CandidateParams | None = None,
    **overrides,
) -> CandidateSites:
    """The structure's own vacancies: the candidates of :func:`generate_candidate_sites` with at least
    ``min_coordination`` atoms within ``r_nn`` (an empty site surrounded by atoms on nearly every side)."""
    cand = generate_candidate_sites(positions, None, params, **overrides)
    sel = cand.coordination >= min_coordination
    return CandidateSites(
        positions=cand.positions[sel],
        support=cand.support[sel],
        coordination=cand.coordination[sel],
        from_residual=cand.from_residual[sel],
        d1=cand.d1,
        r_nn=cand.r_nn,
        c_max=cand.c_max,
        atom_coordination=cand.atom_coordination,
        n_edges=cand.n_edges,
        n_under=cand.n_under,
        n_proposals=cand.n_proposals,
        n_kept_proposals=cand.n_kept_proposals,
        n_clusters=cand.n_clusters,
        params=cand.params,
    )
