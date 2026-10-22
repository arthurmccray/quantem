"""Gaussian-splat tools: analytic projection of isotropic 3D Gaussians ("atoms") onto tilt
images, back-projection for seeding, optimizer-safe densification helpers, and the learned
atomic prior wrapper. The tomography classes that use these live in :mod:`quantem.tomography`.
"""

from quantem.core.ml.splat.densify import (
    merge_close_atoms as merge_close_atoms,
    prune_mask as prune_mask,
    remap_optimizer_state as remap_optimizer_state,
)
from quantem.core.ml.splat.renderer import (
    backproject_images as backproject_images,
    project_positions as project_positions,
    render_projections as render_projections,
    window_radius_for as window_radius_for,
)
