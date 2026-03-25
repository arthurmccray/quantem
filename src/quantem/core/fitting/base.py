from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Self, Sequence, cast

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from quantem.core.ml.optimizer_mixin import (
    OptimizerMixin,
    OptimizerParams,
    OptimizerType,
    SchedulerParams,
    SchedulerType,
)


class _UnsetSentinel:
    """Marks an optional kwarg as omitted (vs ``None`` or ``{}``)."""

    __slots__ = ()


UNSET = _UnsetSentinel()

OptimizerUnitSpecDict = dict[str, OptimizerType | dict[str, Any]]
SchedulerUnitSpecDict = dict[str, SchedulerType | dict[str, Any]]


def parse_bounded_init(
    value: float | int | Sequence[float | int | None], *, name: str
) -> tuple[float, float | None, float | None]:
    """
    Parse a scalar or bounded initializer specification.

    Parameters
    ----------
    value : float | int | Sequence[float | int | None]
        Accepted forms:
        - ``x`` -> init ``x`` with no bounds.
        - ``(x0, delta)`` -> init ``x0`` with bounds ``[x0-|delta|, x0+|delta|]``.
        - ``(x0, lo, hi)`` -> init ``x0`` with explicit bounds.
    name : str
        Parameter name used in error messages.

    Returns
    -------
    tuple[float, float | None, float | None]
        Parsed ``(init, lo, hi)``.

    Raises
    ------
    ValueError
        If the sequence form is invalid, contains required ``None`` entries,
        has invalid ordering, or ``init`` lies outside explicit bounds.
    """
    if not isinstance(value, (list, tuple, np.ndarray)):
        x = float(cast(float | int, value))
        return x, None, None

    seq = list(value)
    if len(seq) == 0:
        raise ValueError(f"{name} cannot be empty.")
    if seq[0] is None:
        raise ValueError(f"{name} initial value cannot be None.")
    x0 = float(cast(float | int, seq[0]))

    if len(seq) == 1:
        return x0, None, None
    if len(seq) == 2:
        if seq[1] is None:
            raise ValueError(f"{name} delta cannot be None.")
        delta = abs(float(cast(float | int, seq[1])))
        return x0, x0 - delta, x0 + delta
    if len(seq) == 3:
        if seq[1] is None or seq[2] is None:
            raise ValueError(f"{name} bounds cannot contain None.")
        lo = float(cast(float | int, seq[1]))
        hi = float(cast(float | int, seq[2]))
        if lo > hi:
            raise ValueError(f"{name} has invalid bounds: lo ({lo}) > hi ({hi}).")
        if x0 < lo or x0 > hi:
            raise ValueError(f"{name} initial value {x0} is outside bounds [{lo}, {hi}].")
        return x0, lo, hi

    raise ValueError(f"{name} must be scalar, (x0, delta), or (x0, lo, hi).")


@dataclass
class RenderContext:
    shape: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    mask: torch.Tensor | None = None
    fields: dict[str, Any] = field(default_factory=dict)


class OriginND(nn.Module):
    def __init__(self, *, ndim: int, init: Sequence[float]):
        super().__init__()
        if int(ndim) <= 0:
            raise ValueError("ndim must be >= 1.")
        if len(init) != int(ndim):
            raise ValueError("init length must match ndim.")
        self.ndim = int(ndim)
        self.coords = nn.Parameter(torch.as_tensor(init, dtype=torch.float32).reshape(self.ndim))


class RenderComponent(nn.Module):
    DEFAULT_HARD_CONSTRAINTS: dict[str, Any] = {}
    DEFAULT_SOFT_CONSTRAINTS: dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self.hard_constraints: dict[str, Any] = dict(self.DEFAULT_HARD_CONSTRAINTS)
        self.soft_constraints: dict[str, Any] = dict(self.DEFAULT_SOFT_CONSTRAINTS)
        self.parameter_bounds: dict[str, tuple[float | None, float | None]] = {}

    @staticmethod
    def parse_bounded_init(
        value: float | int | Sequence[float | int | None], *, name: str
    ) -> tuple[float, float | None, float | None]:
        """
        Parse bounded initializer forms into ``(init, lo, hi)``.

        Parameters
        ----------
        value : float | int | Sequence[float | int | None]
            Scalar, ``(x0, delta)``, or ``(x0, lo, hi)``.
        name : str
            Parameter name used in error messages.

        Returns
        -------
        tuple[float, float | None, float | None]
            Parsed ``(init, lo, hi)``.
        """
        return parse_bounded_init(value, name=name)

    def register_parameter_bounds(
        self, parameter_name: str, lo: float | None, hi: float | None
    ) -> None:
        """
        Register hard bounds for a trainable parameter.

        Parameters
        ----------
        parameter_name : str
            Name of an ``nn.Parameter`` attribute on this component.
        lo : float | None
            Lower bound, or ``None`` for unbounded lower side.
        hi : float | None
            Upper bound, or ``None`` for unbounded upper side.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``lo > hi``.
        """
        if lo is not None and hi is not None and float(lo) > float(hi):
            raise ValueError(f"Invalid bounds for {parameter_name}: lo ({lo}) > hi ({hi}).")
        self.parameter_bounds[str(parameter_name)] = (
            None if lo is None else float(lo),
            None if hi is None else float(hi),
        )

    def _enforce_parameter_bounds(self) -> None:
        """
        Clamp registered parameters in-place to configured bounds.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If a registered parameter attribute is missing.
        TypeError
            If a registered attribute is not an ``nn.Parameter``.
        """
        if not self.parameter_bounds:
            return
        with torch.no_grad():
            for param_name, (lo, hi) in self.parameter_bounds.items():
                if not hasattr(self, param_name):
                    raise AttributeError(
                        f"Parameter '{param_name}' is not an attribute of {self.__class__.__name__}."
                    )
                param = getattr(self, param_name)
                if not isinstance(param, nn.Parameter):
                    raise TypeError(
                        f"Attribute '{param_name}' on {self.__class__.__name__} is not an nn.Parameter."
                    )
                if lo is None and hi is None:
                    continue
                if lo is None:
                    assert hi is not None
                    param.clamp_(max=float(hi))
                elif hi is None:
                    assert lo is not None
                    param.clamp_(min=float(lo))
                else:
                    param.clamp_(min=float(lo), max=float(hi))

    def _set_constraints(
        self,
        current: dict[str, Any],
        defaults: dict[str, Any],
        constraints: dict[str, Any],
        *,
        strict: bool,
    ) -> None:
        if strict:
            unknown = [k for k in constraints if k not in defaults]
            if unknown:
                keys = ", ".join(str(k) for k in unknown)
                raise KeyError(f"Unknown constraint keys: {keys}")
        current.update(constraints)

    def set_hard_constraints(self, constraints: dict[str, Any], strict: bool = True) -> None:
        self._set_constraints(
            self.hard_constraints, self.DEFAULT_HARD_CONSTRAINTS, constraints, strict=strict
        )

    def set_soft_constraints(self, constraints: dict[str, Any], strict: bool = True) -> None:
        self._set_constraints(
            self.soft_constraints, self.DEFAULT_SOFT_CONSTRAINTS, constraints, strict=strict
        )

    def apply_constraint_params(self, params: dict[str, Any], strict: bool = True) -> None:
        if not isinstance(params, dict):
            raise TypeError("constraint params must be a dict.")
        if "hard" in params or "soft" in params:
            hard = params.get("hard")
            soft = params.get("soft")
            if hard is not None:
                if not isinstance(hard, dict):
                    raise TypeError("constraint params 'hard' value must be a dict.")
                self.set_hard_constraints(hard, strict=strict)
            if soft is not None:
                if not isinstance(soft, dict):
                    raise TypeError("constraint params 'soft' value must be a dict.")
                self.set_soft_constraints(soft, strict=strict)
            return

        hard_updates: dict[str, Any] = {}
        soft_updates: dict[str, Any] = {}
        unknown: dict[str, Any] = {}
        for k, v in params.items():
            if k in self.DEFAULT_HARD_CONSTRAINTS:
                hard_updates[k] = v
            elif k in self.DEFAULT_SOFT_CONSTRAINTS:
                soft_updates[k] = v
            else:
                unknown[k] = v

        if unknown and strict:
            keys = ", ".join(str(k) for k in unknown.keys())
            raise KeyError(f"Unknown constraint keys for {self.__class__.__name__}: {keys}")
        if unknown:
            soft_updates.update(unknown)
        if hard_updates:
            self.set_hard_constraints(hard_updates, strict=strict)
        if soft_updates:
            self.set_soft_constraints(soft_updates, strict=strict)

    def effective_soft_constraints(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        effective = dict(self.soft_constraints)
        if isinstance(params, dict):
            effective.update(params)
        return effective

    def enforce_hard_constraints(self, ctx: RenderContext) -> None:
        self._enforce_parameter_bounds()

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        raise NotImplementedError

    def constraint_loss(
        self, ctx: RenderContext, params: dict[str, Any] | None = None
    ) -> torch.Tensor:
        return torch.zeros((), device=ctx.device, dtype=ctx.dtype)

    def optimizer_child_modules(self) -> list[tuple[str, nn.Module]]:
        """
        Submodules that own a separate optimizer parameter partition.

        Each entry is ``(sub_key, module)``; unit keys become
        ``f"{component_key}/{sub_key}"`` when collecting units for an
        :class:`AdditiveRenderModel`. The component's remaining trainable
        parameters (not belonging to any listed child) form a final partition
        under ``component_key`` alone.
        """
        return []


class AdditiveRenderModel(nn.Module):
    def __init__(self, *, origin: nn.Module, components: list[RenderComponent]):
        super().__init__()
        self.origin = origin
        self.components = nn.ModuleList(components)

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        if len(self.components) == 0:
            return torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype)
        out = self.components[0](ctx)
        for component in self.components[1:]:
            out = out + component(ctx)
        return out

    def _component_constraint_name(self, component: RenderComponent, idx: int) -> str:
        name = getattr(component, "name", None)
        if isinstance(name, str) and name:
            return name
        class_name = component.__class__.__name__
        if class_name:
            return class_name
        return f"component_{idx}"

    def apply_constraint_params(
        self, constraint_params: dict[str, Any], strict: bool = True
    ) -> None:
        if not isinstance(constraint_params, dict):
            raise TypeError("constraint_params must be a dict.")
        source = constraint_params.get("components")
        component_map = source if isinstance(source, dict) else constraint_params
        for target, params in component_map.items():
            if not isinstance(params, dict):
                if strict:
                    raise TypeError(f"Constraint params for '{target}' must be a dict.")
                continue
            target_str = str(target)
            name_matches: list[RenderComponent] = []
            class_matches: list[RenderComponent] = []
            for idx, module in enumerate(self.components):
                component = cast(RenderComponent, module)
                if self._component_constraint_name(component, idx) == target_str:
                    name_matches.append(component)
                if component.__class__.__name__ == target_str:
                    class_matches.append(component)
            targets = name_matches if name_matches else class_matches
            if not targets:
                if strict:
                    raise KeyError(f"No matching component for constraint target '{target_str}'.")
                continue
            for component in targets:
                component.apply_constraint_params(params, strict=strict)

    def apply_hard_constraints(self, ctx: RenderContext) -> None:
        for module in self.components:
            component = cast(RenderComponent, module)
            component.enforce_hard_constraints(ctx)

    def total_constraint_loss(self, ctx: RenderContext) -> torch.Tensor:
        loss = torch.zeros((), device=ctx.device, dtype=ctx.dtype)
        for module in self.components:
            component = cast(RenderComponent, module)
            loss = loss + component.constraint_loss(ctx)
        return loss


def _trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def collect_optimizer_units(
    model: AdditiveRenderModel,
) -> list[tuple[str, list[nn.Parameter]]]:
    """
    Partition trainable parameters into disjoint optimizer units.

    Units are ordered: ``origin`` (if nonempty), then per component in
    ``model.components`` — nested child modules first (see
    :meth:`RenderComponent.optimizer_child_modules`), then a remainder unit for
    trainable parameters of the component not owned by any listed child.
    Component keys match :meth:`AdditiveRenderModel._component_constraint_name`.
    Trainable tensors that also appear under ``model.origin`` are listed only in
    the ``origin`` unit, even when the same module instance is referenced from a
    nested component (e.g. shared :class:`OriginND` on a disk template).

    If the same ``nn.Module`` instance is both a top-level component and a nested
    child (e.g. :class:`DiskTemplate` listed in ``components`` and also exposed
    via :meth:`SyntheticDiskLattice.optimizer_child_modules`), its parameters are
    attributed only to the **first** matching unit (the top-level component key).
    The redundant nested unit (e.g. ``lat0/disk``) is omitted so partitions stay
    disjoint.

    Raises
    ------
    ValueError
        If child partitions overlap or duplicate ``sub_key`` names appear for one
        component.
    """
    units: list[tuple[str, list[nn.Parameter]]] = []
    assigned_globally: set[int] = set()
    origin_params = _trainable_parameters(model.origin)
    origin_param_ids = {id(p) for p in origin_params}
    if origin_params:
        units.append(("origin", origin_params))
        for p in origin_params:
            assigned_globally.add(id(p))

    def _trainable_in_partition(module: nn.Module) -> list[nn.Parameter]:
        """Trainable params for a non-origin unit (skip tensors owned by ``model.origin``)."""
        return [
            p for p in module.parameters() if p.requires_grad and id(p) not in origin_param_ids
        ]

    for idx, module in enumerate(model.components):
        component = cast(RenderComponent, module)
        base_key = model._component_constraint_name(component, idx)
        children = component.optimizer_child_modules()
        seen_sub_keys: set[str] = set()
        claimed_ids: set[int] = set()
        for sub_key, submod in children:
            sk = str(sub_key)
            if sk in seen_sub_keys:
                raise ValueError(
                    f"Duplicate optimizer_child_modules sub_key {sk!r} on "
                    f"{component.__class__.__name__} ({base_key!r})."
                )
            seen_sub_keys.add(sk)
            sub_params_full = _trainable_in_partition(submod)
            if any(id(p) in claimed_ids for p in sub_params_full):
                raise ValueError(
                    f"Overlapping trainable parameters between optimizer child modules "
                    f"on {component.__class__.__name__} ({base_key!r})."
                )
            for p in sub_params_full:
                claimed_ids.add(id(p))
            sub_params_new = [p for p in sub_params_full if id(p) not in assigned_globally]
            if sub_params_new:
                units.append((f"{base_key}/{sk}", sub_params_new))
                for p in sub_params_new:
                    assigned_globally.add(id(p))

        remainder = [
            p
            for p in component.parameters()
            if p.requires_grad
            and id(p) not in claimed_ids
            and id(p) not in origin_param_ids
            and id(p) not in assigned_globally
        ]
        if remainder:
            units.append((base_key, remainder))
            for p in remainder:
                assigned_globally.add(id(p))

    seen_all: set[int] = set()
    for key, params in units:
        for p in params:
            pid = id(p)
            if pid in seen_all:
                raise ValueError(f"Optimizer unit {key!r} repeats a parameter tensor.")
            seen_all.add(pid)

    return units


def _parse_optimizer_spec(spec: OptimizerType | dict[str, Any]) -> OptimizerType:
    if isinstance(spec, dict):
        return OptimizerParams.parse_dict(dict(spec))
    return spec


def _torch_optimizer_single(
    spec: OptimizerType, params: list[nn.Parameter]
) -> torch.optim.Optimizer:
    kw = spec.params()
    match spec:
        case OptimizerParams.Adam():
            return torch.optim.Adam(params, **kw)
        case OptimizerParams.AdamW():
            return torch.optim.AdamW(params, **kw)
        case OptimizerParams.SGD():
            return torch.optim.SGD(params, **kw)
        case OptimizerParams.NoneOptimizer():
            raise ValueError("NoneOptimizer cannot build a torch optimizer.")
        case _:
            raise NotImplementedError(f"Unknown optimizer type: {spec!r}")


def _torch_optimizer_merged(
    spec0: OptimizerType, param_groups: list[dict[str, Any]]
) -> torch.optim.Optimizer:
    match spec0:
        case OptimizerParams.Adam():
            return torch.optim.Adam(param_groups)
        case OptimizerParams.AdamW():
            return torch.optim.AdamW(param_groups)
        case OptimizerParams.SGD():
            return torch.optim.SGD(param_groups)
        case OptimizerParams.NoneOptimizer():
            raise ValueError("NoneOptimizer cannot build a torch optimizer.")
        case _:
            raise NotImplementedError(f"Unknown optimizer type: {spec0!r}")


def _parse_scheduler_spec(spec: SchedulerType | dict[str, Any]) -> SchedulerType:
    if isinstance(spec, dict):
        return SchedulerParams.parse_dict(dict(spec))
    return spec


def _scheduler_fingerprint(
    spec: SchedulerType, num_iter: int | None
) -> tuple[str, frozenset[tuple[str, Any]]]:
    s = copy.deepcopy(spec)
    if isinstance(s, SchedulerParams.NoneScheduler):
        return "none", frozenset()
    ctor_kw = s.params(1.0, num_iter=num_iter)
    return s._name, frozenset(sorted(ctor_kw.items()))


def _torch_scheduler_from_spec(
    spec: SchedulerType,
    optimizer: torch.optim.Optimizer,
    num_iter: int | None,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    s = copy.deepcopy(spec)
    if isinstance(s, SchedulerParams.NoneScheduler):
        return None
    base_lr = float(optimizer.param_groups[0]["lr"])
    ctor_kw = s.params(base_lr, num_iter=num_iter)
    match s:
        case SchedulerParams.Cyclic():
            return torch.optim.lr_scheduler.CyclicLR(optimizer, **ctor_kw)
        case SchedulerParams.Plateau():
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **ctor_kw)
        case SchedulerParams.Exponential():
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **ctor_kw)
        case SchedulerParams.Linear():
            return torch.optim.lr_scheduler.LinearLR(optimizer, **ctor_kw)
        case SchedulerParams.CosineAnnealing():
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **ctor_kw)
        case _:
            raise ValueError(f"Unknown scheduler type: {s!r}")


@dataclass
class FitResult:
    losses: list[float]
    lrs: list[float]
    final_loss: float
    num_steps: int
    metrics: dict[str, list[float]] = field(default_factory=dict)


class FitBase(OptimizerMixin):
    DEFAULT_LR = 1e-2
    DEFAULT_OPTIMIZER_TYPE = "adam"

    def __init__(self):
        super().__init__()
        # Core wiring
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.model: AdditiveRenderModel | None = None
        self.ctx: RenderContext | None = None

        # State/checkpoints
        self.state_initialized: dict[str, torch.Tensor] | None = None

        # Histories/results
        self.fit_history: dict[str, FitResult] = {}

        # Per-unit optimizer overrides (exact keys from collect_optimizer_units); None = broadcast only
        self._fit_optimizer_by_unit: dict[str, OptimizerType] | None = None
        self._render_optimizers: list[torch.optim.Optimizer] = []
        self._optimizer_unit_keys: list[str] = []

        self._fit_scheduler_by_unit: dict[str, SchedulerType] | None = None
        self._render_schedulers: list[
            torch.optim.lr_scheduler.LRScheduler
            | torch.optim.lr_scheduler.ReduceLROnPlateau
            | None
        ] = []
        self._fit_scheduler_last_num_iter: int | None = None

    def get_optimization_parameters(self) -> list[nn.Parameter]:
        if self.model is None:
            return []
        return [p for p in self.model.parameters() if p.requires_grad]

    def _nonempty_optimizer_unit_keys(self) -> set[str]:
        if self.model is None:
            return set()
        return {k for k, ps in collect_optimizer_units(self.model) if ps}

    def _resolve_unit_optimizer_spec(self, unit_key: str) -> OptimizerType:
        if self._fit_optimizer_by_unit and unit_key in self._fit_optimizer_by_unit:
            return self._fit_optimizer_by_unit[unit_key]
        return self._optimizer_params

    def _resolve_unit_scheduler_spec(self, unit_key: str) -> SchedulerType:
        if self._fit_scheduler_by_unit and unit_key in self._fit_scheduler_by_unit:
            return self._fit_scheduler_by_unit[unit_key]
        return self._scheduler_params

    def _build_render_optimizers(self) -> None:
        """
        Construct ``_render_optimizers`` from ``collect_optimizer_units`` and stored specs.

        If every unit resolves to the same optimizer class, uses one ``torch.optim``
        instance with one param group per unit (hyperparameters may differ per group).
        If classes differ, uses one optimizer per unit (stable unit-key order) and
        uses one LR scheduler per optimizer instance (see ``set_scheduler``).
        """
        self._render_schedulers = []
        self._scheduler = None
        self._render_optimizers = []
        self._optimizer_unit_keys = []
        self._optimizer = None

        if self.model is None:
            return

        if isinstance(self._optimizer_params, OptimizerParams.NoneOptimizer):
            return

        units = [(k, ps) for k, ps in collect_optimizer_units(self.model) if ps]
        if not units:
            return

        if self._fit_optimizer_by_unit:
            unit_key_set = {k for k, _ in units}
            stale = set(self._fit_optimizer_by_unit) - unit_key_set
            if stale:
                pruned = {
                    k: v for k, v in self._fit_optimizer_by_unit.items() if k in unit_key_set
                }
                self._fit_optimizer_by_unit = pruned if pruned else None

        specs = [self._resolve_unit_optimizer_spec(k) for k, _ in units]
        for s in specs:
            if isinstance(s, OptimizerParams.NoneOptimizer):
                raise ValueError(
                    "Per-unit or broadcast optimizer spec cannot be NoneOptimizer while training."
                )

        names = [s._name for s in specs]
        homogeneous = len(set(names)) == 1

        if homogeneous:
            param_groups: list[dict[str, Any]] = []
            for (_key, ps), spec in zip(units, specs):
                g = dict(spec.params())
                g["params"] = ps
                param_groups.append(g)
            merged = _torch_optimizer_merged(specs[0], param_groups)
            self._optimizer = merged
            self._render_optimizers = [merged]
            self._optimizer_unit_keys = [k for k, _ in units]
        else:
            opts: list[torch.optim.Optimizer] = []
            keys_out: list[str] = []
            for (key, ps), spec in zip(units, specs):
                opts.append(_torch_optimizer_single(spec, ps))
                keys_out.append(key)
            self._render_optimizers = opts
            self._optimizer_unit_keys = keys_out
            self._optimizer = opts[0]

    def _build_render_schedulers(self, num_iter: int | None) -> None:
        """
        Build ``_render_schedulers`` aligned with ``_render_optimizers``.

        With one torch optimizer (any number of param groups), PyTorch attaches one
        LR scheduler to that optimizer; per-unit scheduler specs must then be
        pairwise equivalent or a ``ValueError`` is raised. With multiple torch
        optimizers, each instance gets its own scheduler from the matching unit's
        resolved spec (``None`` scheduler allowed per unit).

        Optimizer and scheduler internal state are reinitialized when this runs
        (e.g. after trainability changes or ``reconnect_optimizer_to_parameters``).
        """
        self._render_schedulers = []
        self._scheduler = None

        if not self._render_optimizers or not self._optimizer_unit_keys:
            return

        if self._fit_scheduler_by_unit:
            valid = set(self._optimizer_unit_keys)
            stale = set(self._fit_scheduler_by_unit) - valid
            if stale:
                pruned = {k: v for k, v in self._fit_scheduler_by_unit.items() if k in valid}
                self._fit_scheduler_by_unit = pruned if pruned else None

        specs = [self._resolve_unit_scheduler_spec(k) for k in self._optimizer_unit_keys]

        if (
            isinstance(self._scheduler_params, SchedulerParams.NoneScheduler)
            and not self._fit_scheduler_by_unit
        ):
            return

        if len(self._render_optimizers) == 1:
            fingerprints = [_scheduler_fingerprint(s, num_iter) for s in specs]
            if len(set(fingerprints)) > 1:
                raise ValueError(
                    "With a single torch optimizer (multiple param groups), per-unit scheduler "
                    "specs must be equivalent (same type and constructor kwargs after parsing, "
                    "using a dummy base LR of 1.0). Use different optimizer classes per unit "
                    "so each unit has its own optimizer and independent scheduler, or use a "
                    "single broadcast scheduler."
                )
            opt = self._render_optimizers[0]
            sch = _torch_scheduler_from_spec(specs[0], opt, num_iter)
            if sch is not None:
                self._render_schedulers = [sch]
                self._scheduler = sch
        else:
            scheds: list[
                torch.optim.lr_scheduler.LRScheduler
                | torch.optim.lr_scheduler.ReduceLROnPlateau
                | None
            ] = []
            for opt, spec in zip(self._render_optimizers, specs):
                scheds.append(_torch_scheduler_from_spec(spec, opt, num_iter))
            self._render_schedulers = scheds
            self._scheduler = next((s for s in scheds if s is not None), None)

    def set_optimizer(
        self,
        opt_params: OptimizerType | dict[str, Any] | None = None,
        *,
        optimizer_by_unit: OptimizerUnitSpecDict | None | _UnsetSentinel = UNSET,
    ) -> None:
        """
        Configure the render optimizer(s) from broadcast and optional per-unit specs.

        Parameters
        ----------
        opt_params
            Broadcast defaults (parsed if dict). If omitted, previous broadcast is kept.
        optimizer_by_unit
            Optional mapping from unit keys (as returned by ``collect_optimizer_units``)
            to optimizer specs; missing units use the broadcast spec. Pass ``{}`` to clear
            per-unit overrides. When omitted entirely, previous per-unit overrides are
            unchanged.

        Notes
        -----
        **Single vs multiple torch optimizers:** if all units resolve to the same
        optimizer class (Adam, AdamW, SGD), a single optimizer with one param group
        per unit is used. If classes differ across units, one optimizer per unit is
        created and stepped in unit-key order each iteration.

        **Schedulers:** see ``set_scheduler`` for per-unit LR schedules when multiple
        torch optimizers are active.
        """
        if optimizer_by_unit is not UNSET:
            if optimizer_by_unit:
                if not isinstance(optimizer_by_unit, dict):
                    raise TypeError("optimizer_by_unit must be a dict or empty mapping.")
                self._fit_optimizer_by_unit = {
                    str(k): _parse_optimizer_spec(v) for k, v in optimizer_by_unit.items()
                }
            else:
                self._fit_optimizer_by_unit = None

        if opt_params is not None:
            if isinstance(opt_params, dict):
                opt_params = OptimizerParams.parse_dict(opt_params)
            if not isinstance(opt_params, OptimizerType):
                raise TypeError(
                    f"optimizer parameters must be OptimizerType, got {type(opt_params)}"
                )
            self._optimizer_params = opt_params

        if isinstance(self._optimizer_params, OptimizerParams.NoneOptimizer):
            self.remove_optimizer()
            return

        if self.model is None:
            self._optimizer = None
            self._render_optimizers = []
            self._optimizer_unit_keys = []
            return

        if self._fit_optimizer_by_unit:
            invalid = set(self._fit_optimizer_by_unit) - self._nonempty_optimizer_unit_keys()
            if invalid:
                raise ValueError(
                    "optimizer_by_unit keys not in current nonempty units: "
                    + ", ".join(sorted(invalid))
                )

        self._build_render_optimizers()

    def remove_optimizer(self) -> None:
        self._render_optimizers = []
        self._optimizer_unit_keys = []
        self._fit_optimizer_by_unit = None
        self._render_schedulers = []
        self._fit_scheduler_by_unit = None
        self._fit_scheduler_last_num_iter = None
        super().remove_optimizer()

    def zero_optimizer_grad(self) -> None:
        if self._render_optimizers:
            for opt in self._render_optimizers:
                opt.zero_grad(set_to_none=True)
        else:
            super().zero_optimizer_grad()

    def step_optimizer(self) -> None:
        if self._render_optimizers:
            for opt in self._render_optimizers:
                opt.step()
        else:
            super().step_optimizer()

    def get_learning_rates_by_unit(self) -> dict[str, float]:
        """Current LR per optimizer unit key (empty if no optimizer)."""
        if not self._render_optimizers or not self._optimizer_unit_keys:
            return {}
        if len(self._render_optimizers) == 1:
            opt = self._render_optimizers[0]
            out: dict[str, float] = {}
            for i, key in enumerate(self._optimizer_unit_keys):
                if i < len(opt.param_groups):
                    out[key] = float(opt.param_groups[i]["lr"])
            return out
        out = {}
        for key, opt in zip(self._optimizer_unit_keys, self._render_optimizers):
            out[key] = float(opt.param_groups[0]["lr"])
        return out

    def get_current_lr(self) -> float:
        """Mean LR across unit param groups; falls back to mixin behavior if unknown."""
        by_unit = self.get_learning_rates_by_unit()
        if by_unit:
            return float(sum(by_unit.values()) / len(by_unit))
        return super().get_current_lr()

    def set_scheduler(
        self,
        scheduler_params: SchedulerType | dict[str, Any] | None = None,
        num_iter: int | None = None,
        *,
        scheduler_by_unit: SchedulerUnitSpecDict | None | _UnsetSentinel = UNSET,
    ) -> None:
        """
        Configure LR schedulers from broadcast and optional per-unit specs.

        Parameters
        ----------
        scheduler_params
            Broadcast scheduler config (parsed if dict). If omitted, previous broadcast
            is kept.
        num_iter
            Passed to schedulers that need a step count (e.g. ``LinearLR``,
            ``CosineAnnealingLR``). Stored for rebuilds.
        scheduler_by_unit
            Per-unit overrides (keys match ``collect_optimizer_units``). ``{}``
            clears overrides. When the kwarg is omitted, previous overrides are kept.

        Notes
        -----
        **Single torch optimizer:** one ``LRScheduler`` is attached to that optimizer;
        it updates all param groups together. ``scheduler_by_unit`` is only allowed
        if every unit's resolved spec is equivalent (see ``_build_render_schedulers``).

        **Multiple torch optimizers:** one scheduler per optimizer, same order as
        units; each steps in ``step_scheduler`` (``ReduceLROnPlateau`` receives the
        same ``loss`` float on every instance).

        Requires ``_render_optimizers`` to be nonempty (set optimizers first).
        """
        if scheduler_by_unit is not UNSET:
            if scheduler_by_unit:
                if not isinstance(scheduler_by_unit, dict):
                    raise TypeError("scheduler_by_unit must be a dict or empty mapping.")
                self._fit_scheduler_by_unit = {
                    str(k): _parse_scheduler_spec(v) for k, v in scheduler_by_unit.items()
                }
            else:
                self._fit_scheduler_by_unit = None

        if scheduler_params is not None:
            if isinstance(scheduler_params, dict):
                scheduler_params = SchedulerParams.parse_dict(scheduler_params)
            if not isinstance(scheduler_params, SchedulerType):
                raise TypeError(
                    f"scheduler parameters must be SchedulerType, got {type(scheduler_params)}"
                )
            self._scheduler_params = scheduler_params

        if num_iter is not None:
            self._fit_scheduler_last_num_iter = int(num_iter)

        if not self._render_optimizers:
            self._render_schedulers = []
            self._scheduler = None
            return

        if self._fit_scheduler_by_unit:
            invalid = set(self._fit_scheduler_by_unit) - set(self._optimizer_unit_keys)
            if invalid:
                raise ValueError(
                    "scheduler_by_unit keys not in current optimizer unit keys: "
                    + ", ".join(sorted(invalid))
                )

        self._build_render_schedulers(num_iter)

    def step_scheduler(self, loss: float | None = None) -> None:
        if self._render_schedulers:
            for sch in self._render_schedulers:
                if sch is None:
                    continue
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if loss is not None:
                        sch.step(loss)
                else:
                    sch.step()
            return
        super().step_scheduler(loss)

    def reconnect_optimizer_to_parameters(self) -> None:
        """
        Rebuild optimizers so param references match the live model (e.g. after device moves).

        Optimizer momentum and LR scheduler step counters are not preserved across this
        rebuild; schedulers are recreated from stored broadcast and per-unit specs using
        ``_fit_scheduler_last_num_iter`` when set.
        """
        if self.model is None:
            super().reconnect_optimizer_to_parameters()
            return
        if isinstance(self._optimizer_params, OptimizerParams.NoneOptimizer):
            super().reconnect_optimizer_to_parameters()
            return
        self._build_render_optimizers()
        self._build_render_schedulers(self._fit_scheduler_last_num_iter)

    @property
    def state_current(self) -> dict[str, torch.Tensor] | None:
        if self.model is None:
            return None
        return self._get_model_state_dict_copy()

    @property
    def render_initialized(self) -> np.ndarray:
        if self.state_initialized is None:
            raise RuntimeError("initialized state is unavailable. Call .define_model(...) first.")
        return self._render_state_array(self.state_initialized)

    @property
    def render_current(self) -> np.ndarray:
        if self.model is None or self.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")
        return self.model(self.ctx).detach().cpu().numpy()

    def reset(
        self,
        reset_to: Literal["initialized"] = "initialized",
    ) -> Self:
        if reset_to != "initialized":
            raise ValueError("FitBase.reset only supports reset_to='initialized'.")
        if self.state_initialized is None:
            raise RuntimeError("initialized state is unavailable. Call .define_model(...) first.")
        self._load_model_state_dict_copy(self.state_initialized)
        self._clear_fit_history_all()
        return self

    def set_component_trainable(
        self, component_name: str, enabled: bool, rebuild_optimizer: bool = True
    ) -> None:
        """
        Enable or disable optimization for all parameters in one component.

        Parameters
        ----------
        component_name : str
            Resolved component name.
        enabled : bool
            If ``True``, mark component parameters trainable.
        rebuild_optimizer : bool, optional
            If ``True``, rebuild optimizer param groups after toggling.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model is not defined.
        KeyError
            If ``component_name`` is unknown.

        Notes
        -----
        When rebuilding, optimizers are reconstructed from stored broadcast and
        per-unit optimizer specs. LR schedulers are rebuilt from stored broadcast
        and per-unit scheduler specs; scheduler step counters reset (see
        ``_fit_scheduler_last_num_iter``).
        """
        component = self._resolve_component_by_name(component_name)
        for _, param in component.named_parameters(recurse=True):
            param.requires_grad_(bool(enabled))
        if rebuild_optimizer:
            self._rebuild_optimizer_after_trainability_change()

    def set_parameter_trainable(
        self,
        component_name: str,
        parameter_name: str,
        enabled: bool,
        rebuild_optimizer: bool = True,
    ) -> None:
        """
        Enable or disable optimization for one component parameter.

        Parameters
        ----------
        component_name : str
            Resolved component name.
        parameter_name : str
            Parameter name from ``component.named_parameters()``.
        enabled : bool
            If ``True``, mark parameter trainable.
        rebuild_optimizer : bool, optional
            If ``True``, rebuild optimizer param groups after toggling.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model is not defined.
        KeyError
            If ``component_name`` or ``parameter_name`` is unknown.

        Notes
        -----
        When rebuilding, optimizers and LR schedulers follow the same rules as
        :meth:`set_component_trainable` (scheduler step counters reset).
        """
        component = self._resolve_component_by_name(component_name)
        params = dict(component.named_parameters(recurse=True))
        if parameter_name not in params:
            known = ", ".join(sorted(params.keys()))
            raise KeyError(
                f"Parameter '{parameter_name}' not found in component '{component_name}'. "
                f"Known parameters: {known}"
            )
        params[parameter_name].requires_grad_(bool(enabled))
        if rebuild_optimizer:
            self._rebuild_optimizer_after_trainability_change()

    def set_parameters_trainable(
        self,
        component_name: str,
        parameter_names: list[str],
        enabled: bool,
        rebuild_optimizer: bool = True,
    ) -> None:
        """
        Enable or disable optimization for multiple component parameters.

        Parameters
        ----------
        component_name : str
            Resolved component name.
        parameter_names : list[str]
            Parameter names from ``component.named_parameters()``.
        enabled : bool
            If ``True``, mark parameters trainable.
        rebuild_optimizer : bool, optional
            If ``True``, rebuild optimizer param groups after toggling.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model is not defined.
        KeyError
            If any parameter name is unknown.

        Notes
        -----
        When rebuilding, optimizers and LR schedulers follow the same rules as
        :meth:`set_component_trainable` (scheduler step counters reset).
        """
        component = self._resolve_component_by_name(component_name)
        params = dict(component.named_parameters(recurse=True))
        missing = [name for name in parameter_names if name not in params]
        if missing:
            known = ", ".join(sorted(params.keys()))
            raise KeyError(
                f"Unknown parameters for component '{component_name}': {', '.join(missing)}. "
                f"Known parameters: {known}"
            )
        for name in parameter_names:
            params[name].requires_grad_(bool(enabled))
        if rebuild_optimizer:
            self._rebuild_optimizer_after_trainability_change()

    def get_component_trainable(self, component_name: str) -> dict[str, bool]:
        """
        Return trainability flags for one component's parameters.

        Parameters
        ----------
        component_name : str
            Resolved component name.

        Returns
        -------
        dict[str, bool]
            Mapping of parameter name to ``requires_grad``.

        Raises
        ------
        RuntimeError
            If the model is not defined.
        KeyError
            If ``component_name`` is unknown.
        """
        component = self._resolve_component_by_name(component_name)
        return {name: bool(param.requires_grad) for name, param in component.named_parameters()}

    def fit_render(
        self,
        *,
        target: torch.Tensor,
        n_steps: int,
        constraint_weight: float = 1.0,
        constraint_params: dict[str, Any] | None = None,
        optimizer_params: OptimizerType | dict[str, Any] | None = None,
        optimizer_by_unit: OptimizerUnitSpecDict | None = None,
        scheduler_params: SchedulerType | dict[str, Any] | None = None,
        scheduler_by_unit: SchedulerUnitSpecDict | None = None,
        progress: bool = False,
        run_key: str = "default",
        **kwargs: Any,
    ) -> FitResult:
        """
        Fit model parameters to a target render.

        Parameters
        ----------
        target : torch.Tensor
            Target tensor to fit.
        n_steps : int
            Number of optimization steps.
        constraint_weight : float, optional
            Multiplier applied to the summed soft-constraint loss.
        constraint_params : dict[str, Any] | None, optional
            Optional constraint updates applied once to matching components before
            optimization starts. If ``None``, existing component constraints are reused.
        optimizer_params : dict | None, optional
            Optimizer configuration override for this call.
        optimizer_by_unit : dict | None, optional
            Per-unit optimizer specs (keys match ``collect_optimizer_units``). Omitted
            entries use ``optimizer_params`` / stored broadcast defaults.
        scheduler_params : dict | None, optional
            Scheduler configuration override for this call.
        scheduler_by_unit : dict | None, optional
            Per-unit scheduler specs (same keys as optimizer units). Omitted entries
            use ``scheduler_params`` / stored broadcast defaults.
        progress : bool, optional
            If ``True``, display a progress bar.
        run_key : str, optional
            History key used to store/append fit metrics.
        **kwargs : Any
            Forwarded to internal forward/loss hooks.

        Returns
        -------
        FitResult
            Fit history and final loss metadata for this run key.

        Raises
        ------
        RuntimeError
            If model/context are undefined.

        Notes
        -----
        Hard constraints are applied after each optimizer step.
        """
        if self.model is None or self.ctx is None:
            raise RuntimeError("Model and context are not defined for fitting.")
        if constraint_params is not None:
            self.model.apply_constraint_params(constraint_params, strict=True)

        optimizer_rebuilt = False
        if optimizer_params is not None:
            obu: OptimizerUnitSpecDict | None | _UnsetSentinel = (
                optimizer_by_unit if optimizer_by_unit is not None else UNSET
            )
            self.set_optimizer(optimizer_params, optimizer_by_unit=obu)
            optimizer_rebuilt = True
        elif optimizer_by_unit is not None:
            self.set_optimizer(None, optimizer_by_unit=optimizer_by_unit)
            optimizer_rebuilt = True
        elif self.optimizer is None:
            if not isinstance(self.optimizer_params, OptimizerParams.NoneOptimizer):
                self.set_optimizer(self.optimizer_params)
            else:
                self.set_optimizer(
                    {
                        "type": getattr(self, "DEFAULT_OPTIMIZER_TYPE", "adamw"),
                        "lr": float(getattr(self, "DEFAULT_LR", self.DEFAULT_LR)),
                    }
                )
            optimizer_rebuilt = True

        n_steps = int(n_steps)
        scheduler_configured = False
        if scheduler_params is not None:
            sbu: SchedulerUnitSpecDict | None | _UnsetSentinel = (
                scheduler_by_unit if scheduler_by_unit is not None else UNSET
            )
            self.set_scheduler(scheduler_params, num_iter=n_steps, scheduler_by_unit=sbu)
            scheduler_configured = True
        elif scheduler_by_unit is not None:
            self.set_scheduler(None, num_iter=n_steps, scheduler_by_unit=scheduler_by_unit)
            scheduler_configured = True
        elif not self._render_schedulers and not isinstance(
            self.scheduler_params, SchedulerParams.NoneScheduler
        ):
            self.set_scheduler(self.scheduler_params, num_iter=n_steps)
            scheduler_configured = True
        elif optimizer_rebuilt and not scheduler_configured:
            self._build_render_schedulers(n_steps)

        pbar = tqdm(range(n_steps), desc="Fit render", disable=not progress)

        losses: list[float] = []
        lrs: list[float] = []
        for _ in pbar:
            self.zero_optimizer_grad()
            pred = self._forward_for_fit(target=target, **kwargs)
            data_loss = self._fidelity_loss(pred, target, **kwargs)
            constraint_loss = self._constraint_loss(pred, target, **kwargs)
            total_loss = data_loss + constraint_weight * constraint_loss
            total_loss.backward()
            self.step_optimizer()
            if self.model is None or self.ctx is None:
                raise RuntimeError("Model and context are not defined for fitting.")
            self.model.apply_hard_constraints(self.ctx)
            total_loss_value = float(total_loss.detach().cpu())
            self.step_scheduler(total_loss_value)
            losses.append(total_loss_value)
            lrs.append(float(self.get_current_lr()))

        key = str(run_key)
        if key in self.fit_history:
            prev = self.fit_history[key]
            prev.losses.extend(losses)
            prev.lrs.extend(lrs)
            prev.final_loss = prev.losses[-1] if prev.losses else float("nan")
            prev.num_steps = len(prev.losses)
            result = prev
        else:
            result = FitResult(
                losses=losses,
                lrs=lrs,
                final_loss=(losses[-1] if losses else float("nan")),
                num_steps=n_steps,
            )
            self.fit_history[key] = result
        return result

    def _iter_named_components(self) -> list[tuple[str, RenderComponent]]:
        """
        Return canonical component names paired with components.

        Returns
        -------
        list[tuple[str, RenderComponent]]
            ``(name, component)`` entries using the model's canonical naming
            rule. Names fall back to class-name/index behavior when ``.name`` is
            missing.

        Raises
        ------
        RuntimeError
            If the model is not defined.
        """
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        entries: list[tuple[str, RenderComponent]] = []
        for idx, module in enumerate(self.model.components):
            component = cast(RenderComponent, module)
            name = self.model._component_constraint_name(component, idx)
            entries.append((name, component))
        return entries

    def get_component_names(self) -> list[str]:
        """
        Return canonical component names.

        Returns
        -------
        list[str]
            Canonical component names.
        """
        return [name for name, _ in self._iter_named_components()]

    def _resolve_component_by_name(self, component_name: str) -> RenderComponent:
        target = str(component_name)
        for resolved_name, component in self._iter_named_components():
            if resolved_name == target:
                return component
        known = ", ".join(self.get_component_names())
        raise KeyError(f"Component not found: {target}. Known components: {known}")

    def _rebuild_optimizer_after_trainability_change(self) -> None:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        if isinstance(self._optimizer_params, OptimizerParams.NoneOptimizer):
            self.remove_optimizer()
            return
        self._build_render_optimizers()
        self._build_render_schedulers(self._fit_scheduler_last_num_iter)

    def _clone_state_dict(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in state.items()}

    def _get_model_state_dict_copy(self) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        return self._clone_state_dict(self.model.state_dict())

    def _load_model_state_dict_copy(self, state: dict[str, torch.Tensor]) -> None:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        self.model.load_state_dict(self._clone_state_dict(state), strict=True)

    def _clear_fit_history_all(self) -> None:
        self.fit_history.clear()

    def _clear_fit_history_run(self, run_key: str) -> None:
        self.fit_history.pop(str(run_key), None)

    def _render_state_array(self, state: dict[str, torch.Tensor]) -> np.ndarray:
        if self.model is None or self.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")
        live = self._get_model_state_dict_copy()
        try:
            self._load_model_state_dict_copy(state)
            arr = self.model(self.ctx).detach().cpu().numpy()
        finally:
            self._load_model_state_dict_copy(live)
        return arr

    def _forward_for_fit(self, *, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.model is None or self.ctx is None:
            raise RuntimeError("Model and context are not defined for fitting.")
        return self.model(self.ctx)

    def _fidelity_loss(
        self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        if self.ctx is not None and self.ctx.mask is not None:
            # TODO -- use loss modules (currently implemented in tomo branch)
            # and update them to allow for masking at module level
            diff = (pred - target) * self.ctx.mask
            denom = torch.clamp(torch.sum(self.ctx.mask), min=1.0)
            return torch.sum(diff * diff) / denom
        return self.loss_fn(pred, target)

    def _constraint_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.model is None or self.ctx is None:
            raise RuntimeError("Model and context are not defined for fitting.")
        return self.model.total_constraint_loss(self.ctx)


Component = RenderComponent
ModelContext = RenderContext
Model = AdditiveRenderModel
Parameter = nn.Parameter
