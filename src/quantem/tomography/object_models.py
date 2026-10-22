from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from numpy.typing import NDArray
from tqdm.auto import tqdm

from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.constraints import BaseConstraints, Constraints
from quantem.core.ml.ddp import DDPMixin
from quantem.core.ml.loss_functions import get_loss_module
from quantem.core.ml.models.model_base import PlanarDecompositionModel
from quantem.core.ml.optimizer_mixin import OptimizerMixin, OptimizerParams, OptimizerParamsType
from quantem.core.utils.rng import RNGMixin
from quantem.tomography.dataset_models import TomographyINRPretrainDataset
from quantem.tomography.tomography_context import ReconstructionContext


def _tomography_autocast(device: torch.device | str, use_bfloat16: bool):
    device = torch.device(device)
    enabled = bool(use_bfloat16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


class ObjConstraintParams:
    """
    Namespace class for object reconstruction constraint dataclasses and parsing utilities.

    Contains constraint definitions for pixelated and implicit neural representation
    (INR) object types, along with a factory method for instantiating the appropriate
    class from a configuration dictionary.

    Supported constraint types
    --------------------------
    ObjPixelatedConstraints
        Constraints for a voxel-grid (pixelated) object representation.
    ObjINRConstraints
        Constraints for a network-parameterized (INR) object representation;
        adds a sparsity term not present in the pixelated variant.

    Examples
    --------
    >>> ObjConstraintParams.parse_dict({"name": "obj_pixelated", "tv_vol": 0.01})
    ObjPixelatedConstraints(positivity=True, shrinkage=0.0, tv_vol=0.01)
    >>> ObjConstraintParams.parse_dict({"type": "obj_inr", "sparsity": 0.05})
    ObjINRConstraints(positivity=True, shrinkage=0.0, tv_vol=0.0, sparsity=0.05)
    """

    @dataclass
    class ObjPixelatedConstraints(Constraints):
        """
        Constraints for a pixelated (voxel-grid) object representation.

        Attributes
        ----------
        positivity : bool
            If ``True``, enforces non-negative values in the reconstruction.
        shrinkage : float
            Shrinkage regularization strength; pushes values toward zero.
        tv_vol : float
            Total variation regularization weight for the 3-D volume.
        soft_constraint_keys : list[str]
            Constraint fields penalized softly during optimization.
        hard_constraint_keys : list[str]
            Constraint fields enforced strictly during optimization.
        """

        positivity: bool = True
        shrinkage: float = 0.0
        tv_vol: float = 0.0
        _name: str = "obj_pixelated"

        soft_constraint_keys = ["tv_vol"]
        hard_constraint_keys = ["positivity", "shrinkage"]

    @dataclass
    class ObjINRConstraints(Constraints):
        """
        Constraints for an implicit neural representation (INR) object.

        Extends pixelated constraints with an additional sparsity term suited
        to the continuous, network-parameterized object representation.

        Attributes
        ----------
        positivity : bool
            If ``True``, enforces non-negative values in the reconstruction.
        shrinkage : float
            Shrinkage regularization strength; pushes values toward zero.
        tv_vol : float
            Total variation regularization weight for the 3-D volume.
        sparsity : float
            Sparsity regularization weight; encourages near-zero activations.
        soft_constraint_keys : list[str]
            Constraint fields penalized softly during optimization.
        hard_constraint_keys : list[str]
            Constraint fields enforced strictly during optimization.
        """

        positivity: bool = True
        shrinkage: float = 0.0
        tv_vol: float = 0.0
        sparsity: float = 0.0
        _name: str = "obj_inr"

        soft_constraint_keys = ["tv_vol", "sparsity"]
        hard_constraint_keys = ["positivity", "shrinkage"]

    @dataclass
    class ObjTensorDecompConstraints(Constraints):
        """
        Constraints for a tensor decomposition object representation.

        Attributes
        ----------
        positivity : bool
            If ``True``, enforces non-negative values in the reconstruction.
        shrinkage : float
            Shrinkage regularization strength; pushes values toward zero.
        tv_vol : float
            Total variation regularization weight for the 3-D volume.
        soft_constraint_keys : list[str]
            Constraint fields penalized softly during optimization.
        hard_constraint_keys : list[str]
            Constraint fields enforced strictly during optimization.
        """

        positivity: bool = True
        shrinkage: float = 0.0
        tv_vol: float = 0.0
        tv_plane: float = 0.0
        sparsity: float = 0.0
        _name: str = "obj_tensor_decomp"

        soft_constraint_keys = ["tv_vol", "tv_plane", "sparsity"]
        hard_constraint_keys = ["positivity", "shrinkage"]

    @dataclass
    class ObjSplatConstraints(Constraints):
        """
        Constraints for a Gaussian-splat (explicit atoms) object representation.

        Attributes
        ----------
        tie_amplitude : float
            Weight of ``mean((log A - mean(log A))^2)``: pulls every atom's amplitude toward
            the species mean (0 disables).
        tie_sigma : float
            Same for ``log sigma``.
        repulsion : float
            Weight of ``sum_{pairs d < min_distance} (min_distance - d)^2 / N`` over the cached
            neighbour pairs (0 disables).
        min_distance : float
            Pair distance (Angstrom) below which the repulsion acts.
        sigma_bounds : tuple[float, float]
            Hard clamp on sigma in Angstrom.
        amplitude_min : float
            Hard lower bound on the amplitude.
        box_pad : float
            Positions are clamped to the field-of-view box extended by this margin (Angstrom).
        """

        tie_amplitude: float = 0.0
        tie_sigma: float = 0.0
        repulsion: float = 0.0
        min_distance: float = 2.0
        sigma_bounds: tuple[float, float] = (0.15, 1.5)
        amplitude_min: float = 1e-6
        box_pad: float = 2.0
        _name: str = "obj_splat"

        soft_constraint_keys = ["tie_amplitude", "tie_sigma", "repulsion", "min_distance"]
        hard_constraint_keys = ["sigma_bounds", "amplitude_min", "box_pad"]

    @classmethod
    def parse_dict(
        cls, d: dict
    ) -> "ObjConstraintParams.ObjPixelatedConstraints | ObjConstraintParams.ObjINRConstraints | ObjConstraintParams.ObjTensorDecompConstraints":
        """
        Instantiate an object constraint dataclass from a configuration dictionary.

        The dictionary must contain a ``'name'`` or ``'type'`` key identifying
        which constraint class to construct. All remaining keys are forwarded as
        keyword arguments to the selected dataclass.

        Parameters
        ----------
        d : dict
            Configuration dictionary. Must include ``'name'`` or ``'type'``
            with one of the following values (case-insensitive):

            - ``'obj_pixelated'`` → :class:`ObjPixelatedConstraints`
            - ``'obj_inr'`` → :class:`ObjINRConstraints`

            The value may also be a class ``type`` object, in which case its
            ``__name__`` is used after lower-casing.

        Returns
        -------
        ObjPixelatedConstraints or ObjINRConstraints
            An instance of the appropriate object constraint dataclass.

        Raises
        ------
        ValueError
            If neither ``'name'`` nor ``'type'`` is present, if the value is not
            a string or type, or if the name does not match any known object
            constraint type.
        """
        d = dict(d)
        name = d.pop("name", None)
        type_ = d.pop("type", None)
        name = name or type_
        if name is None:
            raise ValueError("Must provide either 'name' or 'type' key")
        if isinstance(name, type):
            name = name.__name__.lower()
        elif isinstance(name, str):
            name = name.lower()
        else:
            raise ValueError(f"Unknown object constraint type: {name}")
        if name == "obj_pixelated":
            return ObjConstraintParams.ObjPixelatedConstraints(**d)
        elif name == "obj_inr":
            return ObjConstraintParams.ObjINRConstraints(**d)
        elif name == "obj_tensor_decomp":
            return ObjConstraintParams.ObjTensorDecompConstraints(**d)
        elif name == "obj_splat":
            return ObjConstraintParams.ObjSplatConstraints(**d)
        else:
            raise ValueError(f"Unknown object constraint type: {name.lower()}")


ObjConstraintsType = (
    ObjConstraintParams.ObjPixelatedConstraints
    | ObjConstraintParams.ObjINRConstraints
    | ObjConstraintParams.ObjTensorDecompConstraints
    | ObjConstraintParams.ObjSplatConstraints
)


def _unwrap(model: nn.Module | nn.parallel.DistributedDataParallel) -> PlanarDecompositionModel:
    """Unwrap a DistributedDataParallel model to get the underlying module ONLY for tensor decomposition models."""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return cast(PlanarDecompositionModel, model.module)
    return cast(PlanarDecompositionModel, model)


class ObjectBase(AutoSerialize, nn.Module, RNGMixin, OptimizerMixin):
    DEFAULT_LRS = {
        "object": 8e-6,
    }
    _token = object()
    """
    Base class for all ObjectModels to inherit from.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],  # pyright: ignore[reportRedeclaration]
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use a factory method to instantiate this class.")

        self._shape = shape

        # Initialize dependencies
        nn.Module.__init__(self)
        RNGMixin.__init__(self, rng=rng, device=device)
        OptimizerMixin.__init__(self)

        # --- Instantiation ----

    # --- Properties ---
    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Shape of the object (x, y, z).
        """
        return self._shape

    @shape.setter
    def shape(self, new_shape: tuple[int, int, int]):
        self._shape = new_shape

    @property
    def obj(self) -> torch.Tensor:
        """
        Returns the object, should be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def model(self) -> nn.Module:
        """
        Returns the model, should be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the object.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass, should be implemented in subclasses. Note for any nn.Module this is
        a required method.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the object, should be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def params(self) -> Generator[torch.nn.Parameter, None, None]:
        """
        Get the parameters that should be optimized for this model.

        Should be implemented in subclasses.
        """
        raise NotImplementedError

    # --- Helper Functions ---
    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        """Default: a single param group keyed by DEFAULT_OPTIMIZER_KEY.

        Hyperparameters are baked by ``set_optimizer``, not here — return only the tensors.
        """
        return {self.DEFAULT_OPTIMIZER_KEY: list(self.params)}

    @abstractmethod  # Each subclass should implement this.
    def to(self, device: str | torch.device):
        """
        Move the object to a device
        """

        raise NotImplementedError


class ObjectConstraints(BaseConstraints, ObjectBase):  # TODO: Ask Arthur why we still need this
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_tv_loss(self, ctx: ReconstructionContext) -> torch.Tensor:
        """
        Get the TV loss for the object model. Must be implemented in each subclass.
        """
        raise NotImplementedError


class ObjectPixelated(ObjectConstraints):
    """
    Object model for pixelated objects.

    Supports: Conventional algorithms (SIRT, FBP), and AD-based reconstructions.
    """

    DEFAULT_CONSTRAINTS = ObjConstraintParams.ObjPixelatedConstraints()

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        super().__init__(
            shape=shape,
            device=device,
            rng=rng,
            _token=self._token,
        )
        self.constraints: ObjConstraintsType = self.DEFAULT_CONSTRAINTS.copy()

    # --- Instantiation ----
    @classmethod
    def from_uniform(
        cls,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        # Initialize a torch.zeros volume with the given shape
        obj = torch.zeros(shape, device=device, dtype=torch.float32)
        obj_model = cls(shape=shape, device=device, rng=rng)
        obj_model._obj = obj
        return obj_model

    @classmethod
    def from_array(
        cls,
        initial_obj: torch.Tensor | np.ndarray,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        obj_model = cls(shape=initial_obj.shape, device=device, rng=rng)
        if isinstance(initial_obj, np.ndarray):
            initial_obj = torch.tensor(initial_obj, dtype=torch.float32)
        else:
            initial_obj = initial_obj.clone()
        obj_model._obj = initial_obj.to(device)
        return obj_model

    # --- Properties ----
    @property
    def obj(self) -> torch.Tensor:
        return self.apply_hard_constraints(
            self._obj
        )  # TODO: Normalization factor to ensure object agrees with INR.

    @obj.setter
    def obj(self, obj: torch.Tensor):
        self._obj = obj

    @property
    def obj_view(self) -> np.ndarray:
        return self.obj.cpu().unsqueeze(0).numpy()

    # @property
    # def soft_loss(self) -> torch.Tensor:
    #     return self.apply_soft_constraints(self._obj)

    @property
    def name(self) -> str:
        return "obj_pixelated"

    @property
    def obj_type(self) -> str:
        return "pixelated"

    @property
    def dtype(self) -> torch.dtype:
        return self._obj.dtype

    def apply_hard_constraints(
        self,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply hard constraints to the object model.

        Only hard constraint here is the positivity and shrinkage. TODO: Add the other hard constraints.
        """
        obj2 = pred.clone()
        if self.constraints.positivity:
            obj2 = torch.clamp(obj2, min=0.0, max=None)
        if self.constraints.shrinkage:
            obj2 = torch.max(obj2 - self.constraints.shrinkage, torch.zeros_like(obj2))

        # TODO: Need to implement the other hard constraints: Fourier Filter and Circular Mask.
        return obj2

    def apply_soft_constraints(self, ctx: ReconstructionContext) -> torch.Tensor:
        assert ctx.obj is not None, "ObjectPixelated requires ctx.obj to be set"
        soft_loss = torch.tensor(
            0.0, device=ctx.obj.device, dtype=ctx.obj.dtype, requires_grad=True
        )
        if self.constraints.tv_vol > 0:
            tv_loss = self.get_tv_loss(ctx)
            soft_loss += tv_loss
        return soft_loss

    # --- Forward method ---
    def forward(self, coords=None) -> torch.Tensor:
        return self.obj

    # --- Defining the TV loss ---
    def get_tv_loss(self, ctx: ReconstructionContext) -> torch.Tensor:
        assert ctx.obj is not None, "ObjectPixelated requires ctx.obj to be set"
        # TV over the three trailing spatial dims, leaving any leading channel/batch axes
        # intact. Works for a 3-D volume, obj_view's [1, D, H, W], and a multimodal
        # [C, D, H, W] (channels = elemental compositions), matching the INR / tensor-decomp
        # convention where the object carries a leading channel dimension.
        tv_d = torch.pow(ctx.obj[..., 1:, :, :] - ctx.obj[..., :-1, :, :], 2).sum()
        tv_h = torch.pow(ctx.obj[..., :, 1:, :] - ctx.obj[..., :, :-1, :], 2).sum()
        tv_w = torch.pow(ctx.obj[..., :, :, 1:] - ctx.obj[..., :, :, :-1], 2).sum()
        tv_loss = tv_d + tv_h + tv_w

        return tv_loss * self.constraints.tv_vol / ctx.obj.numel()

    # --- Helper Functions ---
    def to(self, device: str | torch.device):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self._obj = self._obj.to(device)
        self.reconnect_optimizer_to_parameters()
        return self


class ObjectINR(ObjectConstraints, DDPMixin):
    DEFAULT_CONSTRAINTS = ObjConstraintParams.ObjINRConstraints()

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        model: nn.Module | None = None,
        _token: object | None = None,
    ):
        super().__init__(
            shape=shape,
            device=device,
            rng=rng,
            _token=self._token,
        )
        self._pretrain_losses = []
        self._pretrain_lrs = []
        self.constraints: ObjConstraintParams.ObjINRConstraints = self.DEFAULT_CONSTRAINTS.copy()
        # Register the network submodule (important: real nn.Module attribute)
        if model is not None:
            self.setup_distributed(device=device)
            self._model = self.distribute_model(model)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        obj_model = cls(
            shape=shape,
            device=device,
            rng=rng,
            model=model,  # ✅ build/register in __init__
        )

        obj_model.setup_distributed(device=device)
        obj_model.to(device)
        return obj_model

    # --- Properties ---

    @property
    def model(self) -> nn.Module | nn.parallel.DistributedDataParallel:
        """
        Returns the INR model.
        """
        return self._model

    # @model.setter
    # def model(self, model: "nn.Module"):
    #     """
    #     This doesn't work -- can't have setters for torch sub modules
    #     https://github.com/pytorch/pytorch/issues/52664

    #     For now, upon initialization private variable `._model` is set to the built model.
    #     """
    #     raise RuntimeError("\n\n\nsetting model, this shouldn't be reachable???\n\n\n")

    @property
    def obj(self) -> torch.Tensor:
        return self._obj

    @obj.setter
    def obj(self, obj: torch.Tensor):
        self._obj = obj

    @property
    def obj_view(self) -> np.ndarray:
        """
        Returns the object as a view of the x, y, z axes.

        Matches the axes of conventionally reconstructed objects, this is the object that will be saved.
        """
        self.create_volume()
        return self._obj.cpu().numpy().transpose(0, 1, 3, 2)

    def apply_soft_constraints(
        self,
        ctx: ReconstructionContext,
    ) -> torch.Tensor:
        soft_loss = torch.tensor(0.0, device=ctx.coords.device)
        if self.constraints.tv_vol > 0:
            assert ctx.coords is not None, (
                "coords must be provided for INR object model to compute the TV loss"
            )
            soft_loss += self.get_tv_loss(ctx)

        if (
            isinstance(self.constraints, ObjConstraintParams.ObjINRConstraints)
            and self.constraints.sparsity > 0
        ):  # NOTE: For the linter, I must make this :)
            assert ctx.pred is not None, (
                "pred must be provided for INR object model to compute the sparsity loss"
            )
            sparsity_loss = self.constraints.sparsity * torch.norm(ctx.pred, p=1)
            soft_loss += sparsity_loss

        return soft_loss

    def apply_hard_constraints(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Apply hard constraints to the predicted values of the INR model.
        """

        if self.constraints.positivity:
            pred = torch.clamp(pred, min=0.0, max=None)
        if self.constraints.shrinkage:
            pred = torch.max(pred - self.constraints.shrinkage, torch.zeros_like(pred))

        return pred

    # --- Define get_tv_loss ---

    def get_tv_loss(self, ctx: ReconstructionContext) -> torch.Tensor:
        """
        Compute the total variation loss for the INR model.
        """
        assert ctx.coords is not None, "coords must be provided for INR object model"
        num_tv_samples = min(10_000, ctx.coords.shape[0])
        tv_indices = torch.randperm(ctx.coords.shape[0], device=ctx.coords.device)[:num_tv_samples]

        tv_coords = ctx.coords[tv_indices].detach().requires_grad_(True)
        tv_densities_recomputed = self.model(tv_coords)
        if isinstance(tv_densities_recomputed, tuple):
            tv_densities_recomputed = tv_densities_recomputed[0]

        # Ensure shape is [num_samples, num_channels]
        if tv_densities_recomputed.dim() == 1:
            tv_densities_recomputed = tv_densities_recomputed.unsqueeze(-1)

        # Compute gradients for each channel
        grad_outputs = torch.autograd.grad(
            outputs=tv_densities_recomputed,
            inputs=tv_coords,
            grad_outputs=torch.ones_like(tv_densities_recomputed),
            create_graph=True,
        )[0]  # Shape: [num_samples, coord_dim]

        # Compute TV loss - gradient magnitude per sample
        grad_norm = torch.norm(grad_outputs, dim=1)  # Shape: [num_samples]
        return self.constraints.tv_vol * grad_norm.mean()

    # --- Optimization Parameters ---
    @property
    def params(self) -> Generator[torch.nn.Parameter, None, None]:
        return self.model.parameters()  # type: ignore[attr-defined]

    # Pretraining
    @property
    def pretrained_weights(self) -> dict[str, torch.Tensor]:
        """get the pretrained weights of the INR model"""
        return self._pretrained_weights

    def _set_pretrained_weights(self, model: "torch.nn.Module"):
        """set the pretrained weights of the INR model"""
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Pretrained model must be a torch.nn.Module, got {type(model)}")
        self._pretrained_weights = deepcopy(model.state_dict())

    @property
    def pretrain_target(self) -> TomographyINRPretrainDataset:
        """get the pretrain target"""
        return self._pretrain_target

    @pretrain_target.setter
    def pretrain_target(self, target: TomographyINRPretrainDataset):
        """set the pretrain target"""
        self._pretrain_target = target

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the object.
        """
        # TODO: This is a temporary solution to get the dtype of the object.
        return torch.float32

    # --- Helper Functions ---
    def rebuild_model(self):
        self._model = self.distribute_model(self._model)

    # Reset method that goes back to the pretrained weights.
    def reset(self):
        """reset the model to the pretrained weights"""
        self.model.load_state_dict(self._pretrained_weights.copy())
        self._model = self.distribute_model(
            self.model
        )  # Maybe add a check to see if distributed or not, but not very computationally expensive to do this.

    # --- Forward Method ---

    def forward(self, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """forward pass for the INR model"""
        assert coords is not None, "ObjectINR.forward requires coords"

        all_densities = self.model(coords)

        if all_densities.dim() > 1:
            all_densities = all_densities.squeeze(-1)
        valid_mask = (
            (coords[:, 0] >= -1) & (coords[:, 0] <= 1) & (coords[:, 1] >= -1) & (coords[:, 1] <= 1)
        ).float()

        if all_densities.dim() > 1:
            valid_mask = valid_mask.unsqueeze(-1)
        # Multi-dimensional mask
        all_densities = all_densities * valid_mask

        all_densities = self.apply_hard_constraints(all_densities)

        return all_densities

    # Pretrain Loop

    def pretrain(
        self,
        pretrain_dataset: TomographyINRPretrainDataset,
        batch_size: int,
        reset: bool = False,
        num_iters: int = 10,
        num_workers: int = 0,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        loss_fn: Callable | str = "l1",
        verbose: bool = True,
        use_bfloat16: bool = True,
    ):
        """
        Pretrain the INR model to fit target volume.
        """

        if (
            pretrain_dataset is not None
        ):  # Need to make a check if there's already a pretrain dataset to not go through with the setup again.
            self.pretrain_dataset = pretrain_dataset
            (
                self.pretraining_dataloader,
                self.pretraining_sampler,
                self.pretraining_val_dataloader,
                self.pretraining_val_sampler,
            ) = self.setup_dataloader(pretrain_dataset, batch_size, num_workers=num_workers)

        if optimizer_params is not None:
            self.set_optimizer(optimizer_params)
        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_iters)

        if reset:
            self.reset()

        loss_fn = get_loss_module(loss_fn, self.dtype)

        self._pretrain(
            num_iters=num_iters,
            loss_fn=loss_fn,
            verbose=verbose,
            use_bfloat16=use_bfloat16,
        )

    def _pretrain(
        self,
        num_iters: int,
        loss_fn: Callable,
        verbose: bool,
        use_bfloat16: bool,
    ):
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")
        if self.scheduler is None:
            raise RuntimeError("Scheduler not set. Call set_scheduler() first.")

        self.model.train()
        optimizer = self.optimizer
        scheduler = self.scheduler

        pbar = tqdm(range(num_iters), desc="Pretraining", disable=not verbose)
        for a0 in pbar:
            epoch_loss = 0
            for batch_idx, batch in enumerate[Any](self.pretraining_dataloader):
                coords = batch["coords"].to(self.device, non_blocking=True)
                target = batch["target"].to(self.device, non_blocking=True)

                with _tomography_autocast(self.device, use_bfloat16):
                    outputs = self.forward(coords)
                    loss = loss_fn(outputs, target)

                loss.backward()
                epoch_loss += loss.item()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

            self._pretrain_losses.append(epoch_loss / len(self.pretraining_dataloader))
            print(
                f"Epoch {a0 + 1}/{num_iters}, Pretrain Loss: {epoch_loss / len(self.pretraining_dataloader):.4f}"
            )
            self._pretrain_lrs.append(optimizer.param_groups[0]["lr"])

    def create_volume(self, return_vol: bool = False):
        N = max(self._shape)
        with torch.no_grad():
            coords_1d = torch.linspace(-1, 1, N)
            x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
            inputs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
            model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

            inference_batch_size = 5 * N * N
            total_samples = N**3
            samples_per_gpu = total_samples // self.world_size
            remainder = total_samples % self.world_size

            if self.global_rank < remainder:
                start_idx = self.global_rank * (samples_per_gpu + 1)
                end_idx = start_idx + samples_per_gpu + 1
            else:
                start_idx = self.global_rank * samples_per_gpu + remainder
                end_idx = start_idx + samples_per_gpu

            inputs_subset = inputs[start_idx:end_idx]
            num_samples = inputs_subset.shape[0]

            outputs_list = []
            for batch_start in range(0, num_samples, inference_batch_size):
                batch_end = min(batch_start + inference_batch_size, num_samples)
                batch_coords = inputs_subset[batch_start:batch_end].to(
                    self.device, non_blocking=True
                )

                batch_outputs = model(batch_coords)  # (B, C) or (B,) etc.

                if isinstance(batch_outputs, tuple):
                    batch_outputs = batch_outputs[0]
                batch_outputs = self.apply_hard_constraints(batch_outputs)

                # Ensure shape is (B, C)
                if batch_outputs.dim() == 1:
                    batch_outputs = batch_outputs.unsqueeze(-1)  # (B, 1)

                outputs_list.append(batch_outputs.cpu())

            outputs = torch.cat(outputs_list, dim=0)  # (local_B, C)
            C = outputs.shape[-1]  # e.g. 5

            if self.world_size > 1:
                # gather variable-sized first dimension (local_B) while keeping channels
                local_B = outputs.shape[0]
                output_size = torch.tensor(local_B, device=self.device, dtype=torch.long)
                all_sizes = [
                    torch.zeros(1, device=self.device, dtype=torch.long)
                    for _ in range(self.world_size)
                ]
                dist.all_gather(all_sizes, output_size)
                max_size = max(size.item() for size in all_sizes)

                outputs_dev = outputs.to(self.device)  # (local_B, C)
                if local_B < max_size:
                    pad = torch.zeros(
                        (max_size - local_B, C),  # type: ignore
                        device=self.device,
                        dtype=outputs_dev.dtype,
                    )
                    outputs_padded = torch.cat([outputs_dev, pad], dim=0)  # (max_size, C)
                else:
                    outputs_padded = outputs_dev

                gathered_outputs = [
                    torch.empty((max_size, C), device=self.device, dtype=outputs_dev.dtype)  # type: ignore
                    for _ in range(self.world_size)
                ]
                dist.all_gather(gathered_outputs, outputs_padded.contiguous())

                trimmed_outputs = []
                for rank, size in enumerate(all_sizes):
                    trimmed_outputs.append(gathered_outputs[rank][: size.item(), :])

                pred_full = torch.cat(trimmed_outputs, dim=0).reshape(C, N, N, N).float()
            else:
                pred_full = outputs.reshape(C, N, N, N).float()

            if return_vol:
                return pred_full.detach().cpu()

            self._obj = pred_full.detach().cpu()

    def to(self, device: str | torch.device):  # pyright: ignore[reportIncompatibleMethodOverride] -> better to do this device change
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        if self.world_size == 1:
            self._model = self._model.to(device)
        elif not isinstance(self._model, torch.nn.parallel.DistributedDataParallel):
            self.distribute_model(self._model)
        self.reconnect_optimizer_to_parameters()


class ObjectTensorDecomp(ObjectINR):
    DEFAULT_CONSTRAINTS = ObjConstraintParams.ObjTensorDecompConstraints()

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        model: nn.Module | None = None,
        _token: object | None = None,
    ):
        super().__init__(
            shape=shape,
            device=device,
            rng=rng,
            _token=self._token,
        )
        self._pretrain_losses = []
        self._pretrain_lrs = []
        self.constraints: ObjConstraintParams.ObjTensorDecompConstraints = (
            self.DEFAULT_CONSTRAINTS.copy()
        )
        # Register the network submodule (important: real nn.Module attribute)
        if model is not None:
            self.setup_distributed(device=device)
            self._model = self.distribute_model(model)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        obj_model = cls(
            shape=shape,
            device=device,
            rng=rng,
            model=model,  # ✅ build/register in __init__
        )

        obj_model.setup_distributed(device=device)
        obj_model.to(device)
        return obj_model

    # --- Constraints ---

    def apply_soft_constraints(self, ctx: ReconstructionContext) -> torch.Tensor:
        soft_loss = torch.tensor(
            0.0, device=ctx.pred.device if ctx.pred is not None else self.device
        )
        if self.constraints.tv_vol > 0:
            assert ctx.coords is not None, "Coordinates must be provided for TV loss"
            assert ctx.pred is not None, "Prediction must be provided for TV loss"
            soft_loss += self.get_tv_loss(ctx)

        if self.constraints.sparsity > 0:  # NOTE: For the linter, I must make this :)
            assert ctx.all_densities is not None, (
                "All densities must be provided for sparsity loss"
            )
            sparsity_loss = self.constraints.sparsity * ctx.all_densities.abs().mean()
            soft_loss += sparsity_loss

        return soft_loss

    # TV Losses

    def get_tv_loss(self, ctx: ReconstructionContext) -> torch.Tensor:
        """
        Gets the summed total variational loss for the tensor decomposition model.

        _get_plane_tv_loss: Total-variation across the planes.
        _get_volume_tv_loss: Isotropic volume TV
        """
        assert ctx.coords is not None, "Coordinates must be provided for TV loss"
        assert ctx.pred is not None, "Prediction must be provided for TV loss"
        tv_loss = torch.tensor(0.0, device=ctx.pred.device)
        tv_loss += self._get_plane_tv_loss()
        tv_loss += self.get_volume_tv_loss(ctx.coords)
        return tv_loss

    def _get_plane_tv_loss(self) -> torch.Tensor:
        """
        Gets the total-variation across the planes.
        """
        is_tilted = self.model.tilted
        per_level = []

        model = _unwrap(self.model)
        for p in model.grids:
            # p: (3*T, C, H, W) for TILTED, (3, C, H, W) for KPlanes
            dh = (p[:, :, 1:, :] - p[:, :, :-1, :]).pow(2).mean(dim=(1, 2, 3))
            dw = (p[:, :, :, 1:] - p[:, :, :, :-1]).pow(2).mean(dim=(1, 2, 3))
            per_plane = dh + dw  # (3*T,) or (3,)

            if is_tilted:
                T = self.model.T
                per_rotation = per_plane.view(T, 3).sum(dim=1)  # sum 3 planes per rotation
                level_tv = per_rotation.mean()  # avg across rotations
            else:
                level_tv = per_plane.sum()

            per_level.append(level_tv)

        return self.constraints.tv_plane * torch.stack(per_level).sum()

    def get_volume_tv_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Isotropic volume TV via finite differences. Same form as the autograd
        version (L1 of gradient L2-norm) but avoids double-backward, so it
        works for KPlanesTILTED, CPTilted, and anything else.
        """
        num_tv_samples = min(10_000, coords.shape[0])
        tv_indices = torch.randperm(coords.shape[0], device=coords.device)[:num_tv_samples]
        tv_coords = coords[tv_indices]  # (N, 3)

        model = _unwrap(self.model)
        h = 2.0 / min(model.resolution)

        pred = model(tv_coords)
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)  # (N, 1)

        grads = []
        for axis in range(3):
            offset = torch.zeros(3, device=tv_coords.device)
            offset[axis] = h
            shifted_pred = self.model(tv_coords + offset)
            if isinstance(shifted_pred, tuple):
                shifted_pred = shifted_pred[0]
            if shifted_pred.dim() == 1:
                shifted_pred = shifted_pred.unsqueeze(-1)
            grads.append((shifted_pred - pred) / h)  # (N, 1)

        grad_stack = torch.stack(grads, dim=-1)  # (N, C, 3)
        grad_norm = torch.norm(grad_stack, dim=-1)  # (N, C)

        return self.constraints.tv_vol * grad_norm.mean()

    def apply_hard_constraints(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Apply hard constraints to the predicted values of the INR model.
        """

        if self.constraints.positivity:
            pred = torch.clamp(pred, min=0.0, max=None)
        if self.constraints.shrinkage:
            pred = torch.max(pred - self.constraints.shrinkage, torch.zeros_like(pred))

        return pred

    # --- Optimization Parameters ---
    @property
    def params(self) -> Generator[torch.nn.Parameter, None, None]:
        """
        Returns the optimization parameters, here we also check if PPLR is used and return the appropriate parameters.
        """

        return self.model.parameters()  # type: ignore[attr-defined]

    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        """PPLR: per-key param groups (hyperparameters are baked by set_optimizer)."""
        model = _unwrap(self.model)
        return {key: list(model.get_params()[key]) for key in model.param_keys}

    def _normalize_optimizer_params(self, params):
        """ObjectTensorDecomp requires a dict matching model.param_keys."""
        if not isinstance(params, dict) or self._is_single_optimizer_dict(params):
            raise TypeError(
                f"ObjectTensorDecomp requires dict[str, OptimizerParamsType] keyed by "
                f"param_keys; got {type(params)}"
            )
        model = _unwrap(self.model)
        expected = set(model.param_keys)
        got = set(params.keys())
        if got != expected:
            raise ValueError(
                f"optimizer_params keys must match model.param_keys: "
                f"got {got}, expected {expected}"
            )
        return super()._normalize_optimizer_params(params)

    def pretrain(self) -> None:
        raise NotImplementedError(
            "Tensor decomposition pretraining is not usually required, and for TILTED there is a two-phase warmup approach."
        )


class ObjectGaussianSplat(BaseConstraints[ObjConstraintParams.ObjSplatConstraints], ObjectBase):
    """
    Explicit-atom object: a set of isotropic 3D Gaussians with free position, amplitude and
    width, rendered analytically onto tilt images (:func:`quantem.core.ml.splat.render_projections`).

    Parameters are ``_positions`` ``(N, 3)`` in Angstrom (specimen frame, ``(x, y, z)`` with
    ``x`` the tilt axis), ``_log_amplitudes`` ``(N,)`` and ``_log_sigmas`` ``(N,)`` (log so that
    ``A, sigma > 0`` structurally) and an optional per-tilt constant ``_background`` ``(T,)``.
    The optimizer has one param group per key (``positions``, ``amplitudes``, ``sigmas``,
    ``background``). A learned prior enters through ``set_prior_target`` as the soft term
    ``lam * mean ||x - x_hat||^2`` (``x_hat`` detached).
    """

    DEFAULT_CONSTRAINTS = ObjConstraintParams.ObjSplatConstraints()
    DEFAULT_LRS = {"positions": 2e-2, "amplitudes": 1e-2, "sigmas": 5e-3, "background": 1e-2}
    PARAM_KEYS = ("positions", "amplitudes", "sigmas", "background")

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        super().__init__(shape=shape, device=device, rng=rng, _token=_token)
        self._positions: nn.Parameter
        self._log_amplitudes: nn.Parameter
        self._log_sigmas: nn.Parameter
        self._background: nn.Parameter | None = None
        self._box_half = torch.zeros(3)
        self._view_sampling = 1.0
        self._species = "Au"
        self._init_state: dict[str, torch.Tensor | None] = {}
        self._prior_target: torch.Tensor | None = None
        self._prior_lambda: float = 0.0
        self._pair_idx: torch.Tensor | None = None

    # region --- factories ---
    @classmethod
    def from_positions(
        cls,
        positions: NDArray | torch.Tensor,
        amplitude: float | NDArray | torch.Tensor,
        sigma: float | NDArray | torch.Tensor,
        *,
        fov_A: float | tuple[float, float, float],
        num_tilts: int | None = None,
        background: float | NDArray | torch.Tensor = 0.0,
        species: str = "Au",
        view_sampling: float = 0.5,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ObjectGaussianSplat":
        """Build from explicit positions (Angstrom, specimen frame) and shared or per-atom
        amplitude / sigma. ``fov_A`` is the (x, y, z) extent of the reconstruction box."""
        pos = torch.as_tensor(np.asarray(positions), dtype=dtype).reshape(-1, 3)
        n = pos.shape[0]
        amp = torch.as_tensor(np.asarray(amplitude), dtype=dtype).reshape(-1).expand(n).clone()
        sig = torch.as_tensor(np.asarray(sigma), dtype=dtype).reshape(-1).expand(n).clone()
        if (amp <= 0).any() or (sig <= 0).any():
            raise ValueError("amplitude and sigma must be > 0")
        fov = (fov_A,) * 3 if isinstance(fov_A, (int, float)) else tuple(fov_A)
        shape = tuple(int(np.ceil(f / view_sampling)) for f in fov[::-1])  # (z, y, x) voxels
        obj = cls(shape=shape, device=device, rng=rng, _token=cls._token)  # type: ignore[arg-type]
        obj._species = species
        obj._view_sampling = float(view_sampling)
        obj._box_half = torch.as_tensor([f / 2 for f in fov], dtype=dtype)
        obj._positions = nn.Parameter(pos)
        obj._log_amplitudes = nn.Parameter(torch.log(amp))
        obj._log_sigmas = nn.Parameter(torch.log(sig))
        if num_tilts is not None:
            bg = (
                torch.as_tensor(np.asarray(background), dtype=dtype)
                .reshape(-1)
                .expand(num_tilts)
                .clone()
            )
            obj._background = nn.Parameter(bg)
        obj._store_init_state()
        obj.to(device)
        return obj

    @classmethod
    def from_xyz(
        cls,
        path: str,
        amplitude: float,
        sigma: float,
        *,
        atom_to_spec: tuple[int, int, int] = (1, 0, 2),
        center: bool = False,
        **kwargs,
    ) -> "ObjectGaussianSplat":
        """Build from a 2-line-header xyz file (``El x y z`` in Angstrom). ``atom_to_spec``
        permutes the file columns into the specimen ``(x, y, z)`` order; the default maps the
        HAADF simulator's atom frame (image axis 0 = atom x, axis 1 = atom y = tilt axis)."""
        xyz = np.loadtxt(path, skiprows=2, usecols=(1, 2, 3), dtype=np.float64, ndmin=2)
        xyz = xyz[:, list(atom_to_spec)]
        if center:
            xyz = xyz - xyz.mean(axis=0, keepdims=True)
        return cls.from_positions(xyz, amplitude, sigma, **kwargs)

    @classmethod
    def from_volume(
        cls,
        volume: NDArray | torch.Tensor,
        voxel_size: float,
        amplitude: float,
        sigma: float,
        *,
        axis_order: tuple[str, str, str] = ("z", "y", "x"),
        blur_sigmas: tuple[float, float] | float = (1.0, 2.0),
        threshold_fraction: float = 0.99,
        min_distance: float = 0.0,
        max_peaks: int | None = None,
        **kwargs,
    ) -> "ObjectGaussianSplat":
        """Seed atoms from a volume with :func:`quantem.tomography.atom_trace.seed_peaks`.
        Voxel ``(k, i, j)`` of a ``("z", "y", "x")`` volume sits at
        ``((j - (W-1)/2) v, (i - (H-1)/2) v, (k - (D-1)/2) v)``; ``min_distance`` is in Angstrom."""
        from quantem.tomography.atom_trace import seed_peaks

        vol = torch.as_tensor(
            np.asarray(volume) if not isinstance(volume, torch.Tensor) else volume
        )
        vol = vol.detach().float().cpu()
        peaks, _ = seed_peaks(
            vol,
            blur_sigmas=blur_sigmas,
            threshold_fraction=threshold_fraction,
            min_distance=float(min_distance) / float(voxel_size),
            max_peaks=max_peaks,
        )
        shape = vol.shape
        pos = {}
        for axis, name in enumerate(axis_order):
            pos[name] = (peaks[:, axis].double() - (shape[axis] - 1) / 2) * float(voxel_size)
        xyz = torch.stack((pos["x"], pos["y"], pos["z"]), dim=1).numpy()
        if "fov_A" not in kwargs:
            kwargs["fov_A"] = tuple(
                float(shape[axis_order.index(n)] * voxel_size) for n in ("x", "y", "z")
            )
        return cls.from_positions(xyz, amplitude, sigma, **kwargs)

    @classmethod
    def from_dataset(
        cls,
        dset,
        sigma: float = 0.38,
        amplitude: float | None = None,
        *,
        voxel_size: float | None = None,
        fov_z_A: float | None = None,
        n_atoms_guess: int | None = None,
        **seed_kwargs,
    ) -> "ObjectGaussianSplat":
        """Seed atoms from the unfiltered back-projection of a ``TomographySplatDataset``
        (its current poses). ``amplitude`` defaults to the mean image mass per atom
        (``n_atoms_guess`` = number of seeds when not given)."""
        from quantem.core.ml.splat.renderer import backproject_images

        with torch.no_grad():
            target, rot, shifts = dset.batch(None)
            H, W = dset.image_shape
            s = dset.sampling
            v = float(voxel_size) if voxel_size is not None else 2.0 * s
            fov_xy = (W * s, H * s)
            fov_z = float(fov_z_A) if fov_z_A is not None else min(fov_xy)
            shape = (int(round(fov_z / v)), int(round(fov_xy[1] / v)), int(round(fov_xy[0] / v)))
            vol = backproject_images(target.detach(), rot.detach(), shifts.detach(), s, shape, v)
        obj = cls.from_volume(
            vol,
            v,
            1.0,
            sigma,
            fov_A=(fov_xy[0], fov_xy[1], fov_z),
            num_tilts=dset.num_tilts,
            **seed_kwargs,
        )
        n = obj.num_atoms if n_atoms_guess is None else int(n_atoms_guess)
        if amplitude is None:
            amplitude = float(target.sum(dim=(1, 2)).mean()) * s * s / max(n, 1)
        with torch.no_grad():
            obj._log_amplitudes.fill_(float(np.log(amplitude)))
        obj._store_init_state()
        return obj

    # endregion

    # region --- properties ---
    @property
    def num_atoms(self) -> int:
        return int(self._positions.shape[0])

    @property
    def positions(self) -> np.ndarray:
        return self._positions.detach().cpu().numpy()

    @property
    def amplitudes(self) -> torch.Tensor:
        return torch.exp(self._log_amplitudes)

    @property
    def sigmas(self) -> torch.Tensor:
        return torch.exp(self._log_sigmas)

    @property
    def amplitudes_np(self) -> np.ndarray:
        return self.amplitudes.detach().cpu().numpy()

    @property
    def sigmas_np(self) -> np.ndarray:
        return self.sigmas.detach().cpu().numpy()

    @property
    def background(self) -> torch.Tensor | None:
        return self._background

    @property
    def species(self) -> str:
        return self._species

    @property
    def box_half(self) -> torch.Tensor:
        return self._box_half

    @property
    def device(self) -> torch.device:
        return self._positions.device

    @property
    def dtype(self) -> torch.dtype:
        return self._positions.dtype

    @property
    def model(self) -> nn.Module:
        return self

    @property
    def params(self) -> Generator[torch.nn.Parameter, None, None]:
        yield self._positions
        yield self._log_amplitudes
        yield self._log_sigmas
        if self._background is not None:
            yield self._background

    @property
    def prior_target(self) -> torch.Tensor | None:
        return self._prior_target

    @property
    def obj(self) -> torch.Tensor:
        return self.render_volume()

    @property
    def obj_view(self) -> np.ndarray:
        """``(1, D, H, W)`` volume indexed ``(z, y, x)`` at ``view_sampling``; ``obj_view[0].sum(0)``
        is the zero-tilt projection on the image axes."""
        return self.render_volume().detach().cpu().numpy()[None]

    # endregion

    # region --- optimizer wiring ---
    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        groups = {
            "positions": [self._positions],
            "amplitudes": [self._log_amplitudes],
            "sigmas": [self._log_sigmas],
        }
        if self._background is not None:
            groups["background"] = [self._background]
        return groups

    def _normalize_optimizer_params(self, params):
        """A single spec is broadcast to every group; a dict must name exactly the groups."""
        keys = list(self.get_optimization_parameters().keys())
        if isinstance(params, OptimizerParamsType) or (
            isinstance(params, dict) and self._is_single_optimizer_dict(params)
        ):
            spec = (
                params
                if isinstance(params, OptimizerParamsType)
                else OptimizerParams.parse_dict(d=params)
            )
            return {k: deepcopy(spec) for k in keys}
        if not isinstance(params, dict):
            raise TypeError(f"optimizer_params must be a spec or a dict keyed by {keys}")
        if set(params) != set(keys):
            raise ValueError(f"optimizer_params keys must be {keys}, got {sorted(params)}")
        return super()._normalize_optimizer_params(params)

    # endregion

    # region --- forward / rendering ---
    def window_radius(self, sampling: float, window_sigmas: float = 4.0) -> int:
        from quantem.core.ml.splat.renderer import window_radius_for

        return window_radius_for(float(self.sigmas.detach().max()), sampling, window_sigmas)

    def forward(  # type: ignore[override]
        self,
        rotations: torch.Tensor,
        shifts: torch.Tensor,
        image_shape: tuple[int, int],
        sampling: float,
        tilt_idx: torch.Tensor | None = None,
        window_sigmas: float = 4.0,
        tilt_chunk: int | None = None,
    ) -> torch.Tensor:
        """Render ``(B, H, W)`` images for the given poses."""
        from quantem.core.ml.splat.renderer import render_projections

        bg = None
        if self._background is not None:
            bg = self._background if tilt_idx is None else self._background[tilt_idx]
            if bg.shape[0] != rotations.shape[0]:
                raise ValueError(
                    f"{bg.shape[0]} background values for {rotations.shape[0]} poses: pass tilt_idx"
                )
        return render_projections(
            self._positions,
            self.amplitudes,
            self.sigmas,
            rotations,
            shifts,
            sampling,
            image_shape,
            self.window_radius(sampling, window_sigmas),
            background=bg,
            tilt_chunk=tilt_chunk,
        )

    def render_volume(
        self, view_sampling: float | None = None, window_sigmas: float = 4.0
    ) -> torch.Tensor:
        """Sum of the 3D Gaussians on a ``(D, H, W)`` grid indexed ``(z, y, x)``."""
        from quantem.tomography.atom_trace import render_isotropic

        v = float(view_sampling) if view_sampling is not None else self._view_sampling
        D, H, W = self.shape
        with torch.no_grad():
            pos = self._positions.detach()
            vox = torch.stack(
                (
                    pos[:, 2] / v + (D - 1) / 2,
                    pos[:, 1] / v + (H - 1) / 2,
                    pos[:, 0] / v + (W - 1) / 2,
                ),
                dim=1,
            )
            sig = self.sigmas.detach()
            peak = self.amplitudes.detach() / ((2 * np.pi) ** 1.5 * sig**3)  # density per A^3
            r = int(np.ceil(window_sigmas * float(sig.max()) / v))
            return render_isotropic(vox, peak, sig / v, (D, H, W), r)

    # endregion

    # region --- constraints and prior ---
    def apply_hard_constraints(self) -> None:  # type: ignore[override]
        c = self.constraints
        with torch.no_grad():
            lo, hi = c.sigma_bounds
            self._log_sigmas.clamp_(np.log(lo), np.log(hi))
            self._log_amplitudes.clamp_(min=np.log(c.amplitude_min))
            half = (self._box_half + c.box_pad).to(self._positions)
            self._positions.copy_(torch.maximum(torch.minimum(self._positions, half), -half))

    def apply_soft_constraints(self, ctx: ReconstructionContext | None = None) -> torch.Tensor:  # type: ignore[override]
        self.reset_soft_constraint_losses()
        c = self.constraints
        loss = torch.zeros((), device=self.device, dtype=self.dtype)
        if c.tie_amplitude > 0:
            la = self._log_amplitudes
            term = c.tie_amplitude * ((la - la.mean().detach()) ** 2).mean()
            self.add_soft_constraint_loss("tie_amplitude", term)
            loss = loss + term
        if c.tie_sigma > 0:
            ls = self._log_sigmas
            term = c.tie_sigma * ((ls - ls.mean().detach()) ** 2).mean()
            self.add_soft_constraint_loss("tie_sigma", term)
            loss = loss + term
        if c.repulsion > 0 and self._pair_idx is not None and self._pair_idx.numel() > 0:
            pi = self._pair_idx
            d = (self._positions[pi[:, 0]] - self._positions[pi[:, 1]]).norm(dim=-1)
            pen = torch.clamp(c.min_distance - d, min=0.0)
            term = c.repulsion * (pen * pen).sum() / max(self.num_atoms, 1)
            self.add_soft_constraint_loss("repulsion", term)
            loss = loss + term
        if self._prior_target is not None and self._prior_lambda > 0:
            term = (
                self._prior_lambda
                * ((self._positions - self._prior_target) ** 2).sum(dim=-1).mean()
            )
            self.add_soft_constraint_loss("prior", term)
            loss = loss + term
        return loss

    def get_tv_loss(self, ctx: ReconstructionContext) -> torch.Tensor:
        return torch.zeros((), device=self.device, dtype=self.dtype)

    def set_prior_target(self, target: torch.Tensor, lam: float) -> None:
        """Store a detached clean-position estimate ``x_hat`` and its weight."""
        if target.shape != self._positions.shape:
            raise ValueError(
                f"prior target shape {tuple(target.shape)} != {tuple(self._positions.shape)}"
            )
        self._prior_target = target.detach().to(self._positions)
        self._prior_lambda = float(lam)

    def clear_prior_target(self) -> None:
        self._prior_target = None
        self._prior_lambda = 0.0

    def refresh_neighbors(self, min_distance: float | None = None) -> None:
        """Cache the atom pairs closer than ``min_distance`` (default: the constraint value)."""
        from scipy.spatial import cKDTree

        r = float(self.constraints.min_distance if min_distance is None else min_distance)
        pairs = cKDTree(self.positions).query_pairs(r, output_type="ndarray")
        self._pair_idx = torch.as_tensor(pairs, dtype=torch.long, device=self.device).reshape(
            -1, 2
        )

    # endregion

    # region --- densification ---
    def _rebuild(
        self,
        keep_idx: torch.Tensor | None,
        new_pos: torch.Tensor | None = None,
        new_amp: torch.Tensor | None = None,
        new_sig: torch.Tensor | None = None,
    ) -> None:
        """Replace the per-atom Parameters by ``kept rows (+ appended new rows)`` and remap the
        optimizer state accordingly."""
        from quantem.core.ml.splat.densify import remap_optimizer_state

        def _new(param: nn.Parameter, extra: torch.Tensor | None) -> nn.Parameter:
            kept = param.detach() if keep_idx is None else param.detach()[keep_idx]
            if extra is not None:
                kept = torch.cat((kept, extra.to(kept)), dim=0)
            return nn.Parameter(kept.clone())

        n_new = 0 if new_pos is None else int(new_pos.shape[0])
        p_new = _new(self._positions, new_pos)
        a_new = _new(self._log_amplitudes, None if new_amp is None else torch.log(new_amp))
        s_new = _new(self._log_sigmas, None if new_sig is None else torch.log(new_sig))
        for old, new in (
            (self._positions, p_new),
            (self._log_amplitudes, a_new),
            (self._log_sigmas, s_new),
        ):
            remap_optimizer_state(self.optimizer, old, new, keep_idx, n_new)
        self._positions, self._log_amplitudes, self._log_sigmas = p_new, a_new, s_new
        self._prior_target = None
        self._pair_idx = None

    def prune(self, keep_mask: torch.Tensor) -> int:
        """Remove the atoms where ``keep_mask`` is False; returns the number removed."""
        keep_idx = torch.nonzero(keep_mask.to(self.device)).reshape(-1)
        removed = self.num_atoms - int(keep_idx.numel())
        if removed:
            self._rebuild(keep_idx)
        return removed

    def add_atoms(
        self,
        positions: torch.Tensor | NDArray,
        amplitude: float | torch.Tensor | None = None,
        sigma: float | torch.Tensor | None = None,
    ) -> int:
        """Append atoms (default amplitude / sigma = current medians); returns how many."""
        pos = torch.as_tensor(
            np.asarray(positions) if not isinstance(positions, torch.Tensor) else positions
        )
        pos = pos.to(self._positions).reshape(-1, 3)
        n = int(pos.shape[0])
        if n == 0:
            return 0
        amp = (
            self.amplitudes.detach().median() if amplitude is None else torch.as_tensor(amplitude)
        )
        sig = self.sigmas.detach().median() if sigma is None else torch.as_tensor(sigma)
        amp = amp.to(self._positions).reshape(-1).expand(n)
        sig = sig.to(self._positions).reshape(-1).expand(n)
        self._rebuild(torch.arange(self.num_atoms, device=self.device), pos, amp, sig)
        return n

    def merge(self, min_distance: float) -> int:
        """Merge atoms closer than ``min_distance`` Angstrom; returns the number removed."""
        from quantem.core.ml.splat.densify import merge_close_atoms

        keep_idx, pos, amp, sig = merge_close_atoms(
            self._positions, self.amplitudes, self.sigmas, min_distance
        )
        removed = self.num_atoms - int(keep_idx.numel())
        if removed:
            self._rebuild(keep_idx)
            with torch.no_grad():
                self._positions.copy_(pos)
                self._log_amplitudes.copy_(torch.log(amp))
                self._log_sigmas.copy_(torch.log(sig))
        return removed

    # endregion

    # region --- state ---
    def _store_init_state(self) -> None:
        self._init_state = {
            "positions": self._positions.detach().cpu().clone(),
            "log_amplitudes": self._log_amplitudes.detach().cpu().clone(),
            "log_sigmas": self._log_sigmas.detach().cpu().clone(),
            "background": None
            if self._background is None
            else self._background.detach().cpu().clone(),
        }

    def reset(self) -> None:
        st = self._init_state
        dev = self.device
        self._positions = nn.Parameter(st["positions"].to(dev).clone())
        self._log_amplitudes = nn.Parameter(st["log_amplitudes"].to(dev).clone())
        self._log_sigmas = nn.Parameter(st["log_sigmas"].to(dev).clone())
        if st["background"] is not None:
            self._background = nn.Parameter(st["background"].to(dev).clone())
        self.clear_prior_target()
        self._pair_idx = None
        if self.optimizer is not None:
            self.set_optimizer(self.optimizer_params)

    def to(self, device: str | torch.device):  # type: ignore[override]
        dev = torch.device(device)
        self._positions = nn.Parameter(self._positions.detach().to(dev))
        self._log_amplitudes = nn.Parameter(self._log_amplitudes.detach().to(dev))
        self._log_sigmas = nn.Parameter(self._log_sigmas.detach().to(dev))
        if self._background is not None:
            self._background = nn.Parameter(self._background.detach().to(dev))
        self._box_half = self._box_half.to(dev)
        if self._prior_target is not None:
            self._prior_target = self._prior_target.to(dev)
        if self._pair_idx is not None:
            self._pair_idx = self._pair_idx.to(dev)
        self._rng_to_device(dev)
        self.reconnect_optimizer_to_parameters()
        return self

    def sites_dict(self) -> dict[str, np.ndarray]:
        """Plain-numpy snapshot ``{positions, amplitudes, sigmas, background}``."""
        return {
            "positions": self.positions,
            "amplitudes": self.amplitudes_np,
            "sigmas": self.sigmas_np,
            "background": None
            if self._background is None
            else self._background.detach().cpu().numpy(),
        }

    # endregion


ObjectModelType = ObjectPixelated | ObjectINR | ObjectTensorDecomp | ObjectGaussianSplat
