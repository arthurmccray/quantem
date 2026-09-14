"""Tests for the optional per-axis coordinate warp of the ptycho-tomography object models.

The warp gives the specimen box a larger share of the backend's ``[-1, 1]`` range and squeezes
the support margin into the ends. It must be exactly off by default (bit-identical results), be
a continuous monotone map when on, leave the materialized volume and the crops meaning what they
meant before, and keep gradients flowing (it runs inside checkpointed regions).
"""

import pytest
import torch
import torch.nn as nn

from quantem.ptycho_tomography.geometry import PtychoTomoPatchData, rot_beam_to_spec
from quantem.ptycho_tomography.object_models import ObjectKPlanesTomo, ObjectVoxelTomo

# lateral specimen box 8 Å at 0.5 Å/px (16 px), 6 Å of y margin (12 px) -> c_y = 8/20 = 0.4
BOX_A = (8.0, 8.0)
SAMPLING = (0.5, 0.5)
MARGIN_Y = (0.0, 6.0, 0.0)


def _perturb(obj: ObjectKPlanesTomo, seed: int = 0) -> None:
    """Give the model deterministic non-zero weights (from_uniform starts at exact vacuum)."""
    gen = torch.Generator().manual_seed(seed)
    groups = obj.model.get_params()
    with torch.no_grad():
        for key in ("grids", "sigma_net"):
            for p in groups[key]:
                p.copy_(torch.rand(p.shape, generator=gen, dtype=p.dtype) * 0.2 - 0.1)


def make_kplanes_obj(
    warp_box_frac=None,
    box_margin_A=MARGIN_Y,
    lateral_box_A=BOX_A,
    resolution=(16, 24, 24),
) -> ObjectKPlanesTomo:
    """Small CPU tilted-K-Planes object with the geometry handshake already run."""
    torch.manual_seed(0)
    obj = ObjectKPlanesTomo.from_uniform(
        thickness_A=8.0,
        num_slices=4,
        num_z_voxels=9,
        M_features=8,
        resolution=resolution,
        multiscale_res_multipliers=(0.5, 1.0),
        tilted=True,
        T=1,
        rng=0,
    )
    _perturb(obj)
    obj.set_geometry(
        lateral_box_A=lateral_box_A,
        sampling=SAMPLING,
        box_margin_A=box_margin_A,
        warp_box_frac=warp_box_frac,
    )
    return obj


def make_voxel_obj(warp_box_frac=None, box_margin_A=MARGIN_Y) -> ObjectVoxelTomo:
    obj = ObjectVoxelTomo.from_uniform(thickness_A=8.0, num_slices=4, num_z_voxels=9)
    obj.set_geometry(
        lateral_box_A=BOX_A,
        sampling=SAMPLING,
        box_margin_A=box_margin_A,
        warp_box_frac=warp_box_frac,
    )
    return obj


def payload(lateral: int = 13, tilt: float = 35.0, batch: int = 2) -> PtychoTomoPatchData:
    """Beam-frame patch coordinates in Å over roughly the specimen box."""
    ax = torch.linspace(-1.0, 1.0, lateral) * 4.0
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    coords = torch.stack([gy, gx], dim=-1)[None].expand(batch, -1, -1, -1)
    rots = rot_beam_to_spec(0.0, torch.full((batch,), tilt), 0.0)
    return PtychoTomoPatchData(coords, rots, torch.zeros(batch, dtype=torch.long))


class TestOffByDefault:
    """Every "no warp" spelling must reproduce the pre-warp numbers bit for bit."""

    @pytest.mark.parametrize("spelling", ["none", "all_none", "identity_knots"])
    def test_identity_spellings_are_bit_identical(self, spelling):
        base = make_kplanes_obj()
        if spelling == "none":
            warp = None
        elif spelling == "all_none":
            warp = (None, None, None)
        else:  # the knots of an unwarped model: a == c on every axis
            warp = tuple(a for _, a in base.warp_knots)
        obj = make_kplanes_obj(warp_box_frac=warp)
        assert obj.warp_box_frac is None
        pd = payload()
        assert torch.equal(obj.forward(pd), base.forward(pd))
        assert torch.equal(obj.obj, base.obj)

    def test_helper_returns_the_same_object_when_off(self):
        obj = make_kplanes_obj()
        pts = torch.rand(32, 3) * 2.0 - 1.0
        assert obj._to_backend_coords(pts) is pts

    def test_warp_knots_report_identity_when_off(self):
        obj = make_kplanes_obj()
        (cz, az), (cy, ay), (cx, ax) = obj.warp_knots
        assert (cz, cy, cx) == pytest.approx((1.0, 0.4, 1.0))
        assert (az, ay, ax) == pytest.approx((cz, cy, cx))


class TestTheMap:
    def test_dense_map_is_continuous_and_monotone(self):
        # y margin chosen so the box share is exactly c = 8 / 80 = 0.1
        obj = make_voxel_obj(warp_box_frac=(None, 0.3, None), box_margin_A=(0.0, 36.0, 0.0))
        c, a = obj.warp_knots[1]
        assert (c, a) == pytest.approx((0.1, 0.3))

        n = 10001
        u = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
        pts = torch.stack([torch.zeros_like(u), u, torch.zeros_like(u)], dim=-1)
        w = obj._to_backend_coords(pts)[:, 1]

        steps = w[1:] - w[:-1]
        assert (steps > 0).all(), "the map must be strictly increasing"
        max_slope = max(a / c, (1.0 - a) / (1.0 - c))
        assert steps.max().item() <= 2.0 * max_slope * (2.0 / (n - 1))

        assert w[(n - 1) // 2].item() == pytest.approx(0.0, abs=1e-12)
        assert w[0].item() == pytest.approx(-1.0, abs=1e-6)
        assert w[-1].item() == pytest.approx(1.0, abs=1e-6)

        # the knots themselves land exactly on +-a
        knots = torch.tensor([[0.0, -c, 0.0], [0.0, c, 0.0]], dtype=torch.float64)
        wk = obj._to_backend_coords(knots)[:, 1]
        assert wk[0].item() == pytest.approx(-a, abs=1e-12)
        assert wk[1].item() == pytest.approx(a, abs=1e-12)

    def test_untouched_axes_pass_through(self):
        obj = make_voxel_obj(warp_box_frac=(None, 0.7, None))
        pts = torch.rand(64, 3, dtype=torch.float64) * 2.0 - 1.0
        out = obj._to_backend_coords(pts)
        assert torch.equal(out[:, 0], pts[:, 0])
        assert torch.equal(out[:, 2], pts[:, 2])
        assert not torch.equal(out[:, 1], pts[:, 1])

    def test_gradcheck(self):
        obj = make_voxel_obj(warp_box_frac=(0.8, 0.7, 0.7), box_margin_A=(3.0, 6.0, 6.0))
        gen = torch.Generator().manual_seed(3)
        # keep every point clear of the knots (+-c) and of the ends, where the map has a corner
        mag = 0.05 + 0.25 * torch.rand(24, 3, generator=gen, dtype=torch.float64)  # inner branch
        outer = 0.75 + 0.15 * torch.rand(24, 3, generator=gen, dtype=torch.float64)
        sign = torch.where(torch.rand(48, 3, generator=gen, dtype=torch.float64) < 0.5, -1.0, 1.0)
        pts = (torch.cat([mag, outer], dim=0) * sign).requires_grad_(True)
        assert torch.autograd.gradcheck(obj._to_backend_coords, (pts,))


class TestWarpedModel:
    def test_obj_matches_the_backend_on_the_warped_grid(self):
        obj = make_kplanes_obj(warp_box_frac=(None, 0.7, None))
        d, hh, ww = obj.volume_shape
        zs = torch.linspace(-1.0, 1.0, d)
        ys = torch.linspace(-1.0, 1.0, hh)
        xs = torch.linspace(-1.0, 1.0, ww)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        pts = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)
        with torch.no_grad():
            expected = obj._model(obj._to_backend_coords(pts)).reshape(d, hh, ww)
        assert torch.equal(obj.obj, expected)

    def test_warp_changes_the_result(self):
        base = make_kplanes_obj()
        obj = make_kplanes_obj(warp_box_frac=(None, 0.7, None))
        assert obj.warp_box_frac == (None, 0.7, None)
        assert not torch.equal(obj.obj, base.obj)

    def test_crops_are_unchanged(self):
        base = make_kplanes_obj()
        obj = make_kplanes_obj(warp_box_frac=(None, 0.7, None))
        assert obj.crop_slices == base.crop_slices
        assert obj.volume_shape == base.volume_shape
        assert obj.obj[obj.crop_slices].shape == base.obj[base.crop_slices].shape

    def test_voxel_volume_is_materialized_through_the_warp(self):
        # the voxel fast path (parameter == specimen volume) is only valid without a warp
        obj = make_voxel_obj(warp_box_frac=(None, 0.7, None))
        model = obj._model
        assert isinstance(model, nn.Module)
        with torch.no_grad():
            gen = torch.Generator().manual_seed(1)
            vol = torch.rand(obj.volume_shape, generator=gen)
            getattr(model, "volume").copy_(vol)
        assert not torch.equal(obj.obj, vol)
        assert obj.obj.shape == vol.shape

    def test_soft_constraints_still_differentiable(self):
        obj = make_kplanes_obj(warp_box_frac=(None, 0.7, None))
        obj.constraints = {"tv_weight": 0.1, "positivity_weight": 0.5, "sparsity_weight": 0.1}
        loss = obj.apply_soft_constraints()
        assert torch.isfinite(loss) and loss > 0
        loss.backward()
        grads = [p.grad for p in obj.model.get_params()["grids"]]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


class _CountingModel(nn.Module):
    """Wraps a backend and records how many coordinates it was asked for."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.count = 0

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        self.count += int(pts.shape[0])
        return self.inner(pts)


class TestPointBudget:
    def test_forward_queries_the_same_number_of_points(self):
        counts = []
        for warp in (None, (None, 0.7, None)):
            obj = make_kplanes_obj(warp_box_frac=warp)
            counter = _CountingModel(obj._model)
            obj._model = counter
            obj.forward(payload(tilt=40.0))
            counts.append(counter.count)
        assert counts[0] > 0
        assert counts[0] == counts[1]


class TestValidation:
    def test_rejects_share_below_the_current_one(self):
        with pytest.raises(ValueError, match="smaller than the share"):
            make_voxel_obj(warp_box_frac=(None, 0.2, None))  # c_y = 0.4

    def test_rejects_share_at_or_above_one(self):
        with pytest.raises(ValueError, match="must be < 1"):
            make_voxel_obj(warp_box_frac=(None, 1.0, None))
        with pytest.raises(ValueError, match="must be < 1"):
            make_voxel_obj(warp_box_frac=(None, 1.5, None))

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="3 \\(z, y, x\\) components"):
            make_voxel_obj(warp_box_frac=(0.7, 0.7))

    def test_setting_a_warp_invalidates_the_cached_volume(self):
        obj = make_kplanes_obj()
        before = obj.obj.clone()
        obj.set_geometry(
            lateral_box_A=BOX_A,
            sampling=SAMPLING,
            box_margin_A=MARGIN_Y,
            warp_box_frac=(None, 0.7, None),
        )
        assert obj._obj_cache is None
        assert not torch.equal(obj.obj, before)
