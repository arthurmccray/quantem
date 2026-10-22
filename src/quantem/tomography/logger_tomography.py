import matplotlib.pyplot as plt
import numpy as np
import torch

from quantem.core.ml.logger import LoggerBase
from quantem.tomography.dataset_models import DatasetModelType
from quantem.tomography.object_models import ObjectModelType


class LoggerTomography(LoggerBase):
    """
    Logger for ML-based tomography reconstructions.
    """

    def __init__(
        self,
        log_dir: str,
        run_prefix: str,
        run_suffix: str = "",
        log_images_every: int = 10,
    ):
        super().__init__(log_dir, run_prefix, run_suffix, log_images_every)

    def log_epoch(self, epoch: int, loss: float, tilt_series_loss: float, soft_loss: float):
        self.log_scalar("loss/total", loss, epoch)
        self.log_scalar("loss/tilt_series", tilt_series_loss, epoch)
        self.log_scalar("loss/soft", soft_loss, epoch)

    def log_iter(
        self,
        object_model: ObjectModelType,
        iter: int,
        consistency_loss: float,
        total_loss: float,
        learning_rates: dict[str, float],
        num_samples_per_ray: int,
        val_loss: float | None = None,
    ):
        self.log_scalar("loss/consistency", consistency_loss, iter)
        self.log_scalar("loss/total", total_loss, iter)
        self.log_scalar("loss/soft", object_model._soft_constraint_losses[-1], iter)
        self.log_scalar("num_samples_per_ray", num_samples_per_ray, iter)
        for param_name, lr_value in learning_rates.items():
            self.log_scalar(f"learning_rate/{param_name}", float(lr_value), iter)
        if val_loss is not None:
            self.log_scalar("loss/val", val_loss, iter)

    def log_iter_images(
        self,
        pred_volume: np.ndarray,
        dataset_model: DatasetModelType,
        iter: int,
        logger_cmap: str = "turbo",
    ):
        with torch.no_grad():
            z1_vals = dataset_model.z1_params.detach().cpu().numpy()
            z3_vals = dataset_model.z3_params.detach().cpu().numpy()
            shifts_vals = dataset_model.shifts_params.detach().cpu().numpy()

        for channel in range(pred_volume.shape[0]):
            self.log_image(
                f"volume/sum_z_{channel}", pred_volume[channel].sum(axis=0), iter, logger_cmap
            )
            self.log_image(
                f"volume/sum_y_{channel}", pred_volume[channel].sum(axis=1), iter, logger_cmap
            )
            self.log_image(
                f"volume/sum_x_{channel}", pred_volume[channel].sum(axis=2), iter, logger_cmap
            )

        # Plotting z1 and z3 vals
        fig, ax = plt.subplots()
        ax.plot(z1_vals, label="Z1")
        ax.plot(z3_vals, label="Z3")
        ax.legend()
        ax.set_title("Z1 and Z3 Angles")
        ax.set_xlabel("Tilt Image")
        ax.set_ylabel("Degree")
        self.log_figure("z1_z3_angles", fig, iter)
        plt.close(fig)

        # Plotting shifts
        fig, ax = plt.subplots()
        ax.plot(shifts_vals[:, 0], label="Shifts X")
        ax.plot(shifts_vals[:, 1], label="Shifts Y")
        ax.legend()
        ax.set_title("Shifts")
        ax.set_xlabel("Tilt Image")
        ax.set_ylabel("Pixel")
        self.log_figure("shifts", fig, iter)
        plt.close(fig)


class LoggerTomographySplat(LoggerTomography):
    """
    Logger for Gaussian-splat reconstructions: scalar histories per group, atom statistics,
    prior diagnostics, measured / rendered / residual panels and the pose plots of the parent.
    """

    def log_iter_splat(
        self, recon, iter: int, total_loss: float, data_loss: float, soft_loss: float
    ):
        obj = recon.obj_model
        self.log_scalar("loss/total", total_loss, iter)
        self.log_scalar("loss/data", data_loss, iter)
        self.log_scalar("loss/soft", soft_loss, iter)
        for name, val in obj.soft_constraint_loss.items():
            self.log_scalar(f"loss/soft_{name}", float(val), iter)
        self.log_scalar("atoms/count", obj.num_atoms, iter)
        with torch.no_grad():
            amp = obj.amplitudes.detach().cpu().numpy()
            sig = obj.sigmas.detach().cpu().numpy()
        self.log_scalar("atoms/amplitude_median", float(np.median(amp)), iter)
        self.log_scalar(
            "atoms/amplitude_spread", float(np.std(amp) / max(np.mean(amp), 1e-30)), iter
        )
        self.log_scalar("atoms/sigma_median", float(np.median(sig)), iter)
        if obj.optimizer is not None:
            for key, group in zip(
                obj.get_optimization_parameters().keys(), obj.optimizer.param_groups
            ):
                self.log_scalar(f"learning_rate/{key}", float(group["lr"]), iter)
        if recon.dset.optimizer is not None:
            self.log_scalar(
                "learning_rate/pose", float(recon.dset.optimizer.param_groups[0]["lr"]), iter
            )
        if recon.prior_history and recon.prior_history[-1]["iteration"] == iter:
            for k, v in recon.prior_history[-1].items():
                if k != "iteration":
                    self.log_scalar(f"prior/{k}", float(v), iter)

    def log_iter_images_splat(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dataset_model,
        obj_model,
        iter: int,
        logger_cmap: str = "magma",
    ):
        tilts = dataset_model.tilt_angles.detach().cpu().numpy()
        sel = {
            "min": int(np.argmin(tilts)),
            "ref": int(dataset_model.reference_tilt_idx),
            "max": int(np.argmax(tilts)),
        }
        pred_np, target_np = pred.cpu().numpy(), target.cpu().numpy()
        for tag, k in sel.items():
            fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
            vmin, vmax = (
                float(min(target_np[k].min(), pred_np[k].min())),
                float(max(target_np[k].max(), pred_np[k].max())),
            )
            for ax, img, title in zip(axs, (target_np[k], pred_np[k]), ("measured", "rendered")):
                im = ax.imshow(img, cmap=logger_cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{title} tilt {tilts[k]:+.0f}")
                ax.axis("off")
            fig.colorbar(im, ax=axs[:2], shrink=0.8)
            res = target_np[k] - pred_np[k]
            im = axs[2].imshow(res, cmap=logger_cmap)
            axs[2].set_title("residual")
            axs[2].axis("off")
            fig.colorbar(im, ax=axs[2], shrink=0.8)
            self.log_figure(f"projections/{tag}", fig, iter)
            plt.close(fig)
        self.log_histogram("atoms/amplitudes", obj_model.amplitudes.detach().cpu(), iter)
        self.log_histogram("atoms/sigmas", obj_model.sigmas.detach().cpu(), iter)
        self.log_iter_images(
            pred_volume=obj_model.obj_view,
            dataset_model=dataset_model,
            iter=iter,
            logger_cmap=logger_cmap,
        )
        if getattr(dataset_model, "learn_tilt_offset", False):
            fig, ax = plt.subplots()
            ax.plot(dataset_model.dtilt_params.detach().cpu().numpy(), label="dtilt")
            ax.legend()
            ax.set_title("Tilt-angle offsets")
            ax.set_xlabel("Tilt image")
            ax.set_ylabel("Degree")
            self.log_figure("dtilt", fig, iter)
            plt.close(fig)
