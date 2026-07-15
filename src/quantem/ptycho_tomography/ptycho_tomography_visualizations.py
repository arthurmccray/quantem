"""Visualization mixin for joint ptycho-tomography reconstructions.

Mirrors ``PtychographyVisualizations``: ``visualize`` shows the loss/LR curves (via the inherited
``plot_losses``), central volume cross-sections, and the centered complex probe. Mixed into
:class:`~quantem.ptycho_tomography.ptycho_tomography.PtychoTomography` ahead of ``Ptychography``
so these override the 2D-object versions.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from quantem.core.visualization import show_2d


class PtychoTomographyVisualizations:
    """Volume-aware visualization methods (mixin; expects the PtychoTomography interface)."""

    def _volume_sections(self, slab_frac: float = 0.1, cubic: bool = False) -> list[np.ndarray]:
        """Slab-averaged central cross-sections [(y,x), (z,x), (z,y)] of the cropped volume.

        ``cubic=True`` uses ``volume_cropped_cubic`` (center-cropped to a cube) so depth sections
        are framed consistently with the projection instead of showing z vacuum headroom.
        """
        vol = self.volume_cropped_cubic if cubic else self.volume_cropped  # pyright: ignore[reportAttributeAccessIssue] -- host-class attr
        secs = []
        for ax in range(3):
            n = vol.shape[ax]
            half = max(1, int(round(n * slab_frac / 2)))
            lo, hi = max(0, n // 2 - half), min(n, n // 2 + half)
            secs.append(vol.take(range(lo, hi), axis=ax).mean(axis=ax))
        return secs

    def show_volume_sections(
        self,
        slab_frac: float = 0.1,
        cmap: str = "magma",
        axsize: tuple[float, float] = (4.0, 4.0),
        returnfig: bool = False,
        cubic: bool = False,
    ):
        """Central slab-averaged cross-sections of the reconstructed volume.

        The (z, x) and (z, y) panels are drawn with the true physical aspect
        (``z_voxel / lateral sampling``) so depth sections read to scale. ``cubic=True``
        center-crops to a cube first (consistent framing; no z vacuum headroom).
        """
        secs = self._volume_sections(slab_frac, cubic=cubic)
        titles = ["volume (y, x)", "volume (z, x)", "volume (z, y)"]
        fig, axs = plt.subplots(1, 3, figsize=(3 * axsize[0], axsize[1]))
        z_aspect = self.z_sampling / float(np.mean(self.sampling))  # pyright: ignore[reportAttributeAccessIssue] -- host-class attr
        for i, (ax, sec, title) in enumerate(zip(axs, secs, titles)):
            im = ax.imshow(sec, cmap=cmap, aspect=(z_aspect if i > 0 else 1.0))
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        if returnfig:
            return fig, axs
        plt.show()

    def visualize(
        self,
        cbar: bool = True,
        return_fig: bool = False,
        *,
        cmap: str = "magma",
        cubic: bool = False,
    ):
        """Losses + learning rates, central volume cross-sections, and the centered complex probe.

        Mirrors the ptychography ``visualize`` layout: the top panel reuses the inherited
        ``plot_losses`` (loss + LR curves); the probe is shown centered (fftshift) as a complex
        image, matching the ptychography probe display. ``cubic=True`` center-crops the volume
        to a cube for the cross-sections (consistent framing; no z vacuum headroom).
        """
        fig = plt.figure(figsize=(13, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.35)
        ax_top = fig.add_subplot(gs[0])
        iter_losses = self._iter_losses  # pyright: ignore[reportAttributeAccessIssue] -- host-class attr
        if len(iter_losses):
            self.plot_losses(figax=(fig, ax_top))  # pyright: ignore[reportAttributeAccessIssue] -- host-class attr
        else:
            ax_top.text(0.5, 0.5, "no iterations yet", ha="center", va="center")
            ax_top.set_axis_off()

        gs_bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1])
        axs = np.array([fig.add_subplot(gs_bot[0, i]) for i in range(4)])
        secs = self._volume_sections(cubic=cubic)
        titles = ["volume (y, x)", "volume (z, x)", "volume (z, y)"]
        z_aspect = self.z_sampling / float(np.mean(self.sampling))  # pyright: ignore[reportAttributeAccessIssue] -- host-class attr
        for i, (sec, title) in enumerate(zip(secs, titles)):
            im = axs[i].imshow(sec, cmap=cmap, aspect=(z_aspect if i > 0 else 1.0))
            axs[i].set_title(title)
            if cbar:
                plt.colorbar(im, ax=axs[i], fraction=0.046)

        probe = self.probe  # pyright: ignore[reportAttributeAccessIssue] -- host-class attr
        probe0 = probe.sum(0) if probe.ndim == 3 else probe
        show_2d(
            np.fft.fftshift(probe0),  # centered, complex (amplitude+phase rendering)
            figax=(fig, axs[3]),
            title="Probe",
            cbar=cbar,
            scalebar={"sampling": float(self.sampling[0]), "units": "Å"},  # pyright: ignore[reportAttributeAccessIssue] -- host-class attr
        )
        if len(iter_losses):
            title = f"Final loss: {iter_losses[-1]:.3e} | Iters: {len(iter_losses)}"
            plt.suptitle(title, fontsize=14, y=0.97)
        if return_fig:
            return fig, (ax_top, axs)
        plt.show()
