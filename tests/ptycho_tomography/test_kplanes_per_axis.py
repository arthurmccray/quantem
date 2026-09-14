"""Per-axis K-Planes feature planes (``per_axis_planes``) for the tilted backend.

Change B of the 04_kplanes_padding task. With the flag off the model must be exactly what it was
before (same parameter shapes, same state_dict keys, same numbers); with it on, each multiscale
level is stored as three tensors, one per plane type, each shaped for the two axes that plane type
is really sampled on. The axis probe here is the unit-test version of the CPU check that found the
old mix-up (dev_logs 04_kplanes_padding/logs/check_resz_plane_axes.out).
"""

from typing import Any, cast

import pytest
import torch
from torch import nn

from quantem.core.io.serialize import load as autoserialize_load
from quantem.core.ml.models.kplanes import (
    TILTED_PLANE_AXES,
    KPlanesTILTED,
    interpolate_ms_features_tilted,
    per_axis_plane_shapes,
)
from quantem.ptycho_tomography.geometry import PtychoTomoPatchData, rot_beam_to_spec
from quantem.ptycho_tomography.object_models import ObjectKPlanesTomo

RES_ANISO = (8, 16, 32)  # (z, y, x)
Triple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # one level's three plane types


def make_model(per_axis_planes: bool = False, resolution=RES_ANISO, **kwargs) -> KPlanesTILTED:
    """A small tilted model with a picklable activation and a fixed seed."""
    defaults: dict[str, Any] = dict(
        M_features=3,
        resolution=resolution,
        multiscale_res_multipliers=[0.5, 1.0],
        T=2,
        tau_init="identity",
        density_activation=nn.Identity(),
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return KPlanesTILTED(per_axis_planes=per_axis_planes, **defaults)


def random_pts(n: int = 11) -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    return torch.rand(n, 3, generator=g) * 1.8 - 0.9


# --------------------------------------------------------------------------------------
# (a) flag off: nothing changes
# --------------------------------------------------------------------------------------


def test_flag_off_is_unchanged():
    """Passing ``per_axis_planes=False`` must be the same model as not passing it at all."""
    torch.manual_seed(0)
    plain = KPlanesTILTED(
        M_features=3,
        resolution=RES_ANISO,
        multiscale_res_multipliers=[0.5, 1.0],
        T=2,
        tau_init="identity",
        density_activation=nn.Identity(),
    )
    flagged = make_model(per_axis_planes=False)

    assert list(plain.state_dict().keys()) == list(flagged.state_dict().keys())
    assert list(plain.state_dict().keys())[:2] == ["grids.0", "grids.1"]
    for key, value in plain.state_dict().items():
        assert torch.equal(value, flagged.state_dict()[key]), key

    # the old single stacked tensor per level, (3T, C, res_y, res_z)
    assert [tuple(p.shape) for p in plain.grids] == [(6, 3, 8, 4), (6, 3, 16, 8)]

    pts = random_pts()
    assert torch.equal(plain.forward(pts), flagged.forward(pts))
    assert torch.equal(plain.get_densities(pts), flagged.get_densities(pts))
    assert flagged.per_axis_planes is False
    # the flat list still reads as one tensor per level
    assert all(isinstance(level, torch.Tensor) for level in flagged.ms_grid_levels())


# --------------------------------------------------------------------------------------
# (b) flag on: the three plane shapes
# --------------------------------------------------------------------------------------


def test_per_axis_plane_shapes():
    """With resolution (z, y, x) = (8, 16, 32) the three planes get their own two axes."""
    model = make_model(per_axis_planes=True, multiscale_res_multipliers=[1.0])
    shapes = [tuple(p.shape) for p in model.grids]
    assert shapes == [(2, 3, 16, 8), (2, 3, 8, 32), (2, 3, 32, 16)]
    assert shapes == [tuple(s) for s in per_axis_plane_shapes(2, 3, RES_ANISO)]
    assert list(model.state_dict().keys())[:3] == ["grids.0", "grids.1", "grids.2"]
    assert model.per_axis_planes is True

    # multiscale: three planes per level, level l plane type p at flat index 3*l + p
    multi = make_model(per_axis_planes=True, multiscale_res_multipliers=[0.5, 1.0])
    assert len(multi.grids) == 6
    levels = multi.ms_grid_levels()
    assert len(levels) == 2
    assert all(isinstance(level, tuple) and len(level) == 3 for level in levels)
    assert [tuple(p.shape) for p in cast(Triple, levels[0])] == [
        tuple(s) for s in per_axis_plane_shapes(2, 3, [4, 8, 16])
    ]
    assert multi.forward(random_pts()).shape == (11, 1)


# --------------------------------------------------------------------------------------
# (c) the empirical axis probe: every plane type now sits on the right two axes
# --------------------------------------------------------------------------------------


def _effective_resolution(model: KPlanesTILTED, plane_type: int, axis: int, along: str) -> int:
    """Sign changes of a +1/-1 checkerboard, swept along one coordinate axis.

    A ramp cannot tell resolutions apart (a straight line is a straight line at any spacing), so
    the plane gets a checkerboard along the axis under test and we count sign changes; with N
    control points along that axis there are N-1 of them. The other two plane types are held at a
    flat 1 so the product across plane types passes the signal through untouched.
    """
    levels = model.ms_grid_levels()
    planes = cast(Triple, levels[0])
    with torch.no_grad():
        for level in levels:
            for p in level if isinstance(level, tuple) else [level]:
                p.fill_(1.0)
        target = planes[plane_type]
        _T, C, H, W = target.shape
        n = W if along == "W" else H
        checker = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
        pattern = (
            checker.view(1, 1, n).expand(C, H, W)
            if along == "W"
            else checker.view(1, n, 1).expand(C, H, W)
        )
        target[:] = pattern.unsqueeze(0).expand(_T, C, H, W)

    sweep = torch.linspace(-1.0, 1.0, 4001)
    pts = torch.zeros(4001, 3)
    pts[:, axis] = sweep
    with torch.no_grad():
        feats = interpolate_ms_features_tilted(
            pts=pts, ms_grids=[planes], rotation_matrices=model.so3.as_matrix()
        )
    signs = torch.sign(feats[:, 0])
    signs[signs == 0] = 1
    return int((signs[1:] != signs[:-1]).sum().item()) + 1


def test_axis_probe_every_plane_type_correct():
    """Each plane type's W axis resolves its first coordinate and its H axis the second."""
    model = make_model(per_axis_planes=True, multiscale_res_multipliers=[1.0], M_features=1)
    model.eval()
    assert torch.allclose(model.so3.as_matrix(), torch.eye(3).expand(2, 3, 3), atol=1e-5)

    res = RES_ANISO  # (z, y, x) = axis 0, 1, 2
    for plane_type, (w_axis, h_axis) in enumerate(TILTED_PLANE_AXES):
        for along, axis in (("W", w_axis), ("H", h_axis)):
            measured = _effective_resolution(model, plane_type, axis, along)
            assert measured == res[axis], (plane_type, along, axis, measured, res[axis])
        # the third axis does not move this plane type at all
        other = ({0, 1, 2} - {w_axis, h_axis}).pop()
        assert _effective_resolution(model, plane_type, other, "W") == 1


# --------------------------------------------------------------------------------------
# (d) split equivalence: the triple path is the stacked path
# --------------------------------------------------------------------------------------


def test_split_equals_stacked():
    """Splitting a flag-off level into its three plane types gives bit-identical features."""
    model = make_model(per_axis_planes=False, resolution=(16, 16, 16))
    pts = random_pts(23)
    R = model.so3.as_matrix()
    stacked = interpolate_ms_features_tilted(
        pts=pts, ms_grids=list(model.grids), rotation_matrices=R
    )
    # the stacked first dim is rotation-major: index 3*t + plane_type
    triples: list[Triple] = [
        (g[0::3].contiguous(), g[1::3].contiguous(), g[2::3].contiguous()) for g in model.grids
    ]
    split = interpolate_ms_features_tilted(pts=pts, ms_grids=triples, rotation_matrices=R)
    assert torch.equal(stacked, split)


def test_flag_on_matches_flag_off_with_equal_shapes():
    """An isotropic resolution makes the two layouts the same model, so forward must agree."""
    off = make_model(per_axis_planes=False, resolution=(16, 16, 16))
    on = make_model(per_axis_planes=True, resolution=(16, 16, 16))
    with torch.no_grad():
        for level, stacked in zip(on.ms_grid_levels(), off.grids):
            for p, plane in enumerate(cast(Triple, level)):
                plane.copy_(stacked[p::3])
        on.sigma_net.load_state_dict(off.sigma_net.state_dict())
        cast(torch.Tensor, on.so3.M).copy_(cast(torch.Tensor, off.so3.M))
    pts = random_pts(17)
    assert torch.equal(off.forward(pts), on.forward(pts))


# --------------------------------------------------------------------------------------
# (e) the ptycho-tomo object model: forward, obj, plane TV, save/load
# --------------------------------------------------------------------------------------


def make_obj(per_axis_planes: bool) -> ObjectKPlanesTomo:
    return ObjectKPlanesTomo.from_uniform(
        thickness_A=8.0,
        num_slices=4,
        num_z_voxels=8,
        M_features=4,
        resolution=RES_ANISO,
        multiscale_res_multipliers=(0.5, 1.0),
        tilted=True,
        T=2,
        per_axis_planes=per_axis_planes,
        rng=0,
    )


def payload(lateral: int, tilt: float, batch: int = 2, sampling: float = 0.5):
    ax = torch.linspace(-1, 1, lateral) * ((lateral - 1) / 2.0 * sampling)
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    coords = torch.stack([gy, gx], dim=-1)[None].expand(batch, -1, -1, -1)
    rots = rot_beam_to_spec(0.0, torch.full((batch,), tilt), 0.0)
    return PtychoTomoPatchData(coords, rots, torch.zeros(batch, dtype=torch.long))


def test_object_model_per_axis_runs():
    """forward / obj / plane-TV all work on an anisotropic per-axis model."""
    obj = make_obj(per_axis_planes=True)
    lateral = 17
    obj._initialize_obj((obj.num_slices, lateral, lateral), sampling=(0.5, 0.5))
    model = cast(KPlanesTILTED, obj.model)
    assert [tuple(p.shape) for p in model.grids] == [
        tuple(s) for s in per_axis_plane_shapes(2, 4, [4, 8, 16])
    ] + [tuple(s) for s in per_axis_plane_shapes(2, 4, RES_ANISO)]

    out = obj.forward(payload(lateral, 10.0))
    assert out.shape == (obj.num_slices, 2, lateral, lateral)
    assert torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()
    assert obj.obj.shape == obj.volume_shape
    tv = obj._plane_tv_loss(1.0)
    assert tv.ndim == 0 and torch.isfinite(tv) and float(tv.detach()) > 0.0
    tv.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.grids)

    # reset() reloads the pretrained weights through a strict state_dict load
    obj.reset()


def test_object_model_plane_tv_matches_stacked_when_isotropic():
    """With an isotropic resolution the triple TV is the stacked TV, value for value."""
    off = ObjectKPlanesTomo.from_uniform(
        thickness_A=8.0,
        num_slices=4,
        num_z_voxels=8,
        M_features=4,
        resolution=(16, 16, 16),
        multiscale_res_multipliers=(1.0,),
        tilted=True,
        T=2,
        rng=0,
    )
    on = ObjectKPlanesTomo.from_uniform(
        thickness_A=8.0,
        num_slices=4,
        num_z_voxels=8,
        M_features=4,
        resolution=(16, 16, 16),
        multiscale_res_multipliers=(1.0,),
        tilted=True,
        T=2,
        per_axis_planes=True,
        rng=0,
    )
    with torch.no_grad():
        stacked = cast(KPlanesTILTED, off.model).grids[0]
        for p, plane in enumerate(cast(Triple, cast(KPlanesTILTED, on.model).ms_grid_levels()[0])):
            plane.copy_(stacked[p::3])
    assert torch.equal(off._plane_tv_loss(2.5), on._plane_tv_loss(2.5))


def test_per_axis_planes_rejected_without_tilted():
    with pytest.raises(ValueError, match="only available for the tilted"):
        ObjectKPlanesTomo.from_uniform(thickness_A=8.0, tilted=False, per_axis_planes=True)


def test_object_model_save_load_roundtrip(tmp_path):
    """The three plane shapes and the numbers survive an AutoSerialize round trip."""
    obj = make_obj(per_axis_planes=True)
    lateral = 17
    obj._initialize_obj((obj.num_slices, lateral, lateral), sampling=(0.5, 0.5))
    before = obj.forward(payload(lateral, 10.0)).detach().clone()
    shapes = [tuple(p.shape) for p in cast(KPlanesTILTED, obj.model).grids]

    path = tmp_path / "kplanes_per_axis.zip"
    obj.save(path, mode="o")
    loaded = cast(ObjectKPlanesTomo, autoserialize_load(path))
    model = cast(KPlanesTILTED, loaded.model)
    assert [tuple(p.shape) for p in model.grids] == shapes
    assert model.per_axis_planes is True
    assert torch.allclose(loaded.forward(payload(lateral, 10.0)), before, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(loaded._plane_tv_loss(1.0))


# --------------------------------------------------------------------------------------
# (f) a model pickled before the flag existed
# --------------------------------------------------------------------------------------


def test_old_pickle_without_the_flag_still_runs(tmp_path):
    """An instance saved before ``_per_axis_planes`` existed must load and give the same output."""
    model = make_model(per_axis_planes=False)
    pts = random_pts(13)
    expected = model.forward(pts).detach().clone()

    # emulate a pre-change pickle: the attribute simply is not in the instance dict
    delattr(model, "_per_axis_planes")
    assert "_per_axis_planes" not in model.__dict__
    path = tmp_path / "old_model.pt"
    torch.save(model, path)
    reloaded = cast(KPlanesTILTED, torch.load(path, weights_only=False))
    assert "_per_axis_planes" not in reloaded.__dict__
    assert reloaded.per_axis_planes is False
    assert [tuple(p.shape) for p in reloaded.grids] == [(6, 3, 8, 4), (6, 3, 16, 8)]
    assert torch.equal(reloaded.forward(pts), expected)

    # and an old state_dict still loads strictly into a freshly built flag-off model
    fresh = make_model(per_axis_planes=False)
    fresh.load_state_dict(model.state_dict())
    assert torch.equal(fresh.forward(pts), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
