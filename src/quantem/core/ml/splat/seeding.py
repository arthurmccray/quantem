"""Seeding volumes for Gaussian-splat tomography, in the renderer's own convention.

The crude seed volume is the unfiltered back-projection of the tilt images
(:func:`quantem.core.ml.splat.renderer.backproject_images`), which is blurred by the ``1/r``
point-spread function of parallel-beam back-projection. A filtered back-projection (FBP) applies
the ramp filter along the detector direction perpendicular to the tilt axis before back-projecting
and undoes that blur. Doing it with the renderer's adjoint rather than ``radon_torch`` keeps the
pixel-centre convention of the renderer (pixel ``(i, j)`` at ``((i - (H-1)/2) s, (j - (W-1)/2) s)``;
``radon_torch`` centres at ``N // 2`` and the pixel dataset negates the tilt angles, so a SIRT
volume from that path carries a half-pixel offset and an axis permutation — GOTCHAS 4).

The tilt axis is ``x`` = detector columns, so the filter runs along the rows (``H`` axis).
"""

import math
from typing import TYPE_CHECKING

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

__all__ = ["filter_images", "fbp_volume"]


def filter_images(images: "torch.Tensor", filter_name: str | None = "ramp") -> "torch.Tensor":
    """Apply a 1D frequency-domain filter along the rows of every tilt image.

    Parameters
    ----------
    images : Tensor
        ``(T, H, W)`` tilt images; the tilt axis is along ``W`` (columns), so the filter acts on
        the ``H`` axis.
    filter_name : str or None
        ``"ramp"`` (plain FBP), ``"shepp-logan"``, ``"cosine"``, ``"hamming"``, ``"hann"`` (the ramp
        apodised toward the Nyquist frequency, which suppresses the shot noise the ramp amplifies at
        low dose) or ``None`` (no filtering).

    Returns
    -------
    Tensor
        ``(T, H, W)`` filtered images (same dtype and device).
    """
    from quantem.tomography.radon.radon import get_fourier_filter_torch

    if filter_name is None:
        return images
    T, H, W = (int(v) for v in images.shape)
    n = max(64, int(2 ** math.ceil(math.log2(2 * H))))
    f = get_fourier_filter_torch(n, filter_name, device=images.device, dtype=torch.float32)
    f = f.reshape(1, n, 1).to(images.dtype)
    padded = torch.nn.functional.pad(images, (0, 0, 0, n - H))  # pad rows
    spec = torch.fft.fft(padded, dim=1) * f
    return torch.real(torch.fft.ifft(spec, dim=1))[:, :H, :]


@torch.no_grad()
def fbp_volume(
    images: "torch.Tensor",
    rotations: "torch.Tensor",
    shifts: "torch.Tensor",
    sampling: float,
    volume_shape: tuple[int, int, int],
    voxel_size: float,
    filter_name: str | None = "ramp",
) -> "torch.Tensor":
    """Filtered back-projection on the renderer's voxel grid (see :func:`backproject_images` for
    the voxel convention). ``filter_name=None`` gives the unfiltered back-projection."""
    from quantem.core.ml.splat.renderer import backproject_images

    return backproject_images(
        filter_images(images, filter_name), rotations, shifts, sampling, volume_shape, voxel_size
    )
