"""MACE energy as an atomic prior for the Gaussian-splat fit (chapter 07 of the splat-prior project).

:class:`EnergyPrior` holds a MACE model (e.g. ``mace-mp-0`` medium) in fp32 on one device, frozen and in
eval mode, builds the neighbour list in torch on that device, and exposes

* ``energy(x) -> (E_total, E_per_atom)`` for ``(N, 3)`` positions in Angstrom (non-periodic, no cell),
* ``forces(x) = -dE_total/dx`` by autograd, ``energy_and_forces(x)`` for both from one forward/backward,
* ``per_atom_energy(x)`` (the add / remove signal of step 3),
* ``clean_estimate(x, t=None)`` = ``fire_steps`` FIRE steps on the MACE energy from ``x`` (torch FIRE, no ASE),
  so it satisfies the :class:`~quantem.tomography.tomography_splat.AtomPrior` protocol and works unchanged
  with ``PriorSchedule(mode="loss" | "proximal")`` and every schedule (``t`` is ignored: the energy has no
  noise level; the move is bounded by ``fire_steps * fire_max_step`` per atom instead).

The third way in is ``PriorSchedule(mode="energy")``, which calls :meth:`energy_and_forces` and adds
``lam * E(x)`` to the loss through ``ObjectGaussianSplat.set_prior_gradient`` (the gradient ``-lam * F`` is
held constant between refreshes, exactly as ``x_hat`` is held in the quadratic mode).

MACE itself runs in fp32 throughout; the per-atom energies come out of the model in fp64 (``node_energy``),
the total is the model's fp32 ``energy``, i.e. the same two quantities ``mace.calculators.MACECalculator``
reports as ``energy`` and ``energies`` (their sums differ by fp32 accumulation round-off; compare like with
like).
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

import torch

__all__ = ["EnergyPrior", "neighbour_list_torch", "fire_relax"]

DEFAULT_MACE_MODEL = os.path.expandvars(
    "$SCRATCH/splat_tomo/models/mace/2023-12-03-mace-128-L1_epoch-199.model"
)


def neighbour_list_torch(
    positions: torch.Tensor, cutoff: float, chunk: int = 2048
) -> torch.Tensor:
    """Directed edges ``(2, E)`` (sender, receiver) of all pairs closer than ``cutoff`` (no self edges), by
    brute force on the positions' device in chunks of ``chunk`` rows (``chunk * N * 3`` floats per chunk).

    Non-periodic, so no shifts. Exact differences (no matrix-multiply distance trick), so the edge set is the
    one matscipy builds for the same cutoff up to pairs within float round-off of the cutoff, where the MACE
    radial envelope is zero anyway.
    """
    x = positions.detach()
    n = x.shape[0]
    senders, receivers = [], []
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        d = (x[i0:i1, None, :] - x[None, :, :]).norm(dim=-1)
        mask = d < cutoff
        rows = torch.arange(i1 - i0, device=x.device)
        mask[rows, rows + i0] = False
        s, r = mask.nonzero(as_tuple=True)
        senders.append(s + i0)
        receivers.append(r)
    return torch.stack([torch.cat(senders), torch.cat(receivers)], dim=0)


def fire_relax(
    energy_and_forces,
    x0: torch.Tensor,
    steps: int,
    fmax: float = 0.05,
    max_step: float = 0.1,
    max_disp: float | None = None,
    dt: float = 0.1,
    dt_max: float = 1.0,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    callback=None,
) -> tuple[torch.Tensor, dict]:
    """FIRE (Bitzek et al. 2006, the ASE variant: unit masses, ``v += dt * F``) on ``energy_and_forces(x) ->
    (E_total, E_per_atom, F)`` from ``x0`` for at most ``steps`` steps, stopping early when ``max_i |F_i| < fmax``.

    ``max_step`` caps every atom's displacement per step (Angstrom; ASE caps the norm of the whole step vector
    instead, which for hundreds of atoms is a much smaller per-atom move). ``max_disp`` (optional) caps every
    atom's total displacement from ``x0``. Returns the relaxed positions (detached) and a dict with the
    per-step ``energy`` (eV), ``fmax`` (eV/A) and the number of ``steps`` taken; ``callback(k, x, E, F)`` is
    called after every force evaluation.
    """
    x = x0.detach().clone()
    v = torch.zeros_like(x)
    alpha, n_pos = alpha_start, 0
    hist: dict = {"energy": [], "fmax": [], "steps": 0, "converged": False}
    for k in range(steps + 1):
        e, _, f = energy_and_forces(x)
        fm = float(f.norm(dim=-1).max())
        hist["energy"].append(float(e))
        hist["fmax"].append(fm)
        if callback is not None:
            callback(k, x, e, f)
        if fm < fmax:
            hist["converged"] = True
            break
        if k == steps:
            break
        vf = float((v * f).sum())
        if vf > 0.0:
            vn, fn = v.norm(), f.norm()
            if fn > 0:
                v = (1.0 - alpha) * v + alpha * (vn / fn) * f
            if n_pos > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha *= f_alpha
            n_pos += 1
        else:
            v = torch.zeros_like(v)
            alpha, n_pos = alpha_start, 0
            dt *= f_dec
        v = v + dt * f
        dr = dt * v
        norm = dr.norm(dim=-1, keepdim=True)
        dr = dr * torch.clamp(max_step / norm.clamp_min(1e-12), max=1.0)
        x = x + dr
        if max_disp is not None:
            d = x - x0
            dn = d.norm(dim=-1, keepdim=True)
            x = x0 + d * torch.clamp(max_disp / dn.clamp_min(1e-12), max=1.0)
        hist["steps"] = k + 1
    return x, hist


class EnergyPrior:
    """The MACE energy of the current atoms as a prior (see the module docstring).

    Parameters
    ----------
    model_path : str
        A MACE model file (``torch.load``-able); default the project's ``mace-mp-0`` medium.
    device : str or torch.device
    dtype : torch.dtype
        Working precision of the model (fp32 by default; fp64 only for checks).
    species : int or sequence of int
        Atomic number of every atom (one species, default Au = 79) or one per atom (then ``N`` is fixed);
        a per-call ``numbers`` argument overrides it.
    cutoff : float or None
        Neighbour cutoff in Angstrom; ``None`` reads ``model.r_max``.
    nl_every : int
        Rebuild the neighbour list on every ``nl_every``-th call (1 = every call, the default; a larger value
        reuses the last edge set while the atom count is unchanged, which is only safe while the atoms move
        less than the radial envelope's margin between rebuilds).
    fire_steps, fire_fmax, fire_max_step, fire_max_disp, fire_dt
        The ``clean_estimate`` FIRE: number of steps, early-stop force threshold (eV/A), per-step and total
        per-atom displacement caps (Angstrom; ``None`` = no total cap), initial time step (ASE default 0.1;
        the velocity builds up from zero, so with few steps a larger ``fire_dt`` moves further).
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        species: int | Sequence[int] = 79,
        cutoff: float | None = None,
        nl_every: int = 1,
        fire_steps: int = 10,
        fire_fmax: float = 0.05,
        fire_max_step: float = 0.1,
        fire_max_disp: float | None = None,
        fire_dt: float = 0.1,
        model: torch.nn.Module | None = None,
    ):
        self.model_path = DEFAULT_MACE_MODEL if model_path is None else str(model_path)
        self.device = torch.device(device)
        self.dtype = dtype
        if model is None:
            model = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model = model.to(self.device).to(dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.z_table = [int(z) for z in self.model.atomic_numbers.tolist()]
        self.cutoff = float(self.model.r_max) if cutoff is None else float(cutoff)
        self.species = species
        self.nl_every = max(1, int(nl_every))
        self.fire_steps = int(fire_steps)
        self.fire_fmax = float(fire_fmax)
        self.fire_max_step = float(fire_max_step)
        self.fire_max_disp = fire_max_disp
        self.fire_dt = float(fire_dt)
        self.n_calls = 0  # model evaluations (energy or energy+forces)
        self.total_time_s = 0.0  # wall time inside them (cuda-synchronised)
        self._edge_cache: tuple[int, torch.Tensor] | None = None
        self._one_hot_cache: tuple[tuple, torch.Tensor] | None = None

    # ------------------------------------------------------------------ graph construction
    def _numbers(self, n: int, numbers: torch.Tensor | None) -> tuple:
        if numbers is not None:
            return tuple(int(z) for z in numbers.tolist())
        if isinstance(self.species, int):
            return (int(self.species),) * n
        zs = tuple(int(z) for z in self.species)
        if len(zs) != n:
            raise ValueError(f"species has {len(zs)} entries for {n} atoms")
        return zs

    def _node_attrs(self, zs: tuple) -> torch.Tensor:
        if self._one_hot_cache is not None and self._one_hot_cache[0] == zs:
            return self._one_hot_cache[1]
        idx = torch.tensor([self.z_table.index(z) for z in zs], dtype=torch.long, device=self.device)
        one_hot = torch.nn.functional.one_hot(idx, num_classes=len(self.z_table)).to(self.dtype)
        self._one_hot_cache = (zs, one_hot)
        return one_hot

    def edges(self, x: torch.Tensor, force: bool = False) -> torch.Tensor:
        """Neighbour list ``(2, E)`` for ``x``; rebuilt every ``nl_every`` calls or when ``N`` changes."""
        n = x.shape[0]
        reuse = (
            not force
            and self._edge_cache is not None
            and self._edge_cache[0] == n
            and self.n_calls % self.nl_every != 0
        )
        if not reuse:
            self._edge_cache = (n, neighbour_list_torch(x, self.cutoff))
        return self._edge_cache[1]

    def _batch(self, x: torch.Tensor, numbers: torch.Tensor | None) -> dict:
        n = x.shape[0]
        edge_index = self.edges(x)
        e = edge_index.shape[1]
        lo, hi = x.detach().min(dim=0).values, x.detach().max(dim=0).values
        cell = torch.diag(hi - lo + 2.0 * self.cutoff + 1.0).to(self.dtype)  # unused without stress
        return {
            "positions": x,
            "node_attrs": self._node_attrs(self._numbers(n, numbers)),
            "edge_index": edge_index,
            "shifts": torch.zeros((e, 3), dtype=self.dtype, device=self.device),
            "unit_shifts": torch.zeros((e, 3), dtype=self.dtype, device=self.device),
            "cell": cell,
            "batch": torch.zeros(n, dtype=torch.long, device=self.device),
            "ptr": torch.tensor([0, n], dtype=torch.long, device=self.device),
        }

    def _tic(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _toc(self, t0: float) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.total_time_s += time.perf_counter() - t0
        self.n_calls += 1

    # ------------------------------------------------------------------ energies and forces
    def energy_and_forces(
        self, positions_A: torch.Tensor, numbers: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(E_total [eV, 0-d], E_per_atom [eV, (N,) fp64], F = -dE_total/dx [eV/A, (N, 3)])`` at ``x``.

        One forward and one backward of the model per call; the result tensors are detached and on the
        model's device, the forces in the dtype of ``positions_A``.
        """
        t0 = self._tic()
        x = positions_A.detach().to(device=self.device, dtype=self.dtype).requires_grad_(True)
        out = self.model(self._batch(x, numbers), compute_force=False)
        e_total = out["energy"].sum()
        (g,) = torch.autograd.grad(e_total, x, allow_unused=True)
        forces = torch.zeros_like(x) if g is None else -g
        self._toc(t0)
        return (
            e_total.detach(),
            out["node_energy"].detach().reshape(-1),
            forces.detach().to(positions_A.dtype),
        )

    @torch.no_grad()
    def energy(
        self, positions_A: torch.Tensor, numbers: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(E_total [eV], E_per_atom [eV, (N,)])`` at ``x`` (forward only)."""
        t0 = self._tic()
        x = positions_A.detach().to(device=self.device, dtype=self.dtype)
        # a plain no-grad forward (round 5 step 3, 2026-10-29): MACE sets requires_grad on the positions inside its
        # forward, which is allowed under no_grad; no graph is kept, so the call needs a fraction of the memory of
        # ``energy_and_forces`` (matters for the 17 652-atom S4m); the numbers are identical
        out = self.model(self._batch(x, numbers), compute_force=False)
        self._toc(t0)
        return out["energy"].sum().detach(), out["node_energy"].detach().reshape(-1)

    def per_atom_energy(
        self, positions_A: torch.Tensor, numbers: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-atom MACE energies ``(N,)`` in eV (the step-3 add / remove signal)."""
        return self.energy(positions_A, numbers)[1]

    def forces(
        self, positions_A: torch.Tensor, numbers: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``-dE_total/dx`` ``(N, 3)`` in eV/A."""
        return self.energy_and_forces(positions_A, numbers)[2]

    # ------------------------------------------------------------------ AtomPrior protocol
    def relax(
        self,
        positions_A: torch.Tensor,
        steps: int | None = None,
        fmax: float | None = None,
        max_step: float | None = None,
        max_disp: float | None = None,
        dt: float | None = None,
        callback=None,
    ) -> tuple[torch.Tensor, dict]:
        """FIRE on the MACE energy from ``x`` (defaults = the ``fire_*`` settings); see :func:`fire_relax`."""
        x0 = positions_A.detach().to(device=self.device, dtype=self.dtype)
        x, hist = fire_relax(
            self.energy_and_forces,
            x0,
            steps=self.fire_steps if steps is None else int(steps),
            fmax=self.fire_fmax if fmax is None else float(fmax),
            max_step=self.fire_max_step if max_step is None else float(max_step),
            max_disp=self.fire_max_disp if max_disp is None else max_disp,
            dt=self.fire_dt if dt is None else float(dt),
            callback=callback,
        )
        return x.to(device=positions_A.device, dtype=positions_A.dtype), hist

    def clean_estimate(self, positions_A: torch.Tensor, t: float | None = None) -> torch.Tensor:
        """``AtomPrior`` protocol: ``fire_steps`` FIRE steps from ``x`` (``t`` is ignored, see the module doc)."""
        return self.relax(positions_A)[0]
