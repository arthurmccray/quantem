"""Tests for the voxel-grid ptycho-tomography object backend.

The synthetic rotated-phantom tests are the in-repo "M0": they validate the full rotated-coordinate
forward (physical-Å rotation, per-axis normalization, slab centers, box masking) against analytic
references, independent of any data files.
"""

import pytest
import torch

from quantem.ptycho_tomography.geometry import PtychoTomoPatchData, rot_beam_to_spec
from quantem.ptycho_tomography.object_models import ObjectVoxelTomo, VoxelGrid


def _axis(n: int) -> torch.Tensor:
    return torch.zeros(1) if n == 1 else torch.linspace(-1.0, 1.0, n)


def make_initialized_voxel_obj(
    volume: torch.Tensor | None = None,
    volume_shape=(9, 17, 17),
    thickness_A: float = 8.0,
    num_slices: int = 4,
    lateral_full: int = 17,
    sampling: float = 0.5,
    obj_type: str = "potential",
) -> ObjectVoxelTomo:
    """Build an ObjectVoxelTomo and run the preprocess-time geometry handshake manually."""
    if volume is not None:
        obj = ObjectVoxelTomo.from_array(
            volume, thickness_A=thickness_A, num_slices=num_slices, obj_type=obj_type
        )
    else:
        obj = ObjectVoxelTomo.from_uniform(
            volume_shape, thickness_A=thickness_A, num_slices=num_slices, obj_type=obj_type
        )
    obj._initialize_obj((num_slices, lateral_full, lateral_full), sampling=(sampling, sampling))
    return obj


def full_fov_payload(
    lateral_full: int, tilt_deg: float = 0.0, batch: int = 1, sampling: float = 0.5
) -> PtychoTomoPatchData:
    """One patch spanning the full padded lateral grid (rows/cols at exact grid coords, Å).

    Physical-units refactor: ``coords_yx_A`` carries beam-frame Å (origin at the box center),
    so the normalized [-1, 1] grid is scaled by the lateral half-extent
    ``h = (lateral_full - 1) / 2 * sampling`` (== the object's ``_box_half_extents`` laterals).
    """
    ax = _axis(lateral_full) * ((lateral_full - 1) / 2.0 * sampling)
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    coords = torch.stack([gy, gx], dim=-1)[None].expand(batch, -1, -1, -1)
    rots = rot_beam_to_spec(0.0, torch.full((batch,), float(tilt_deg)), 0.0)
    return PtychoTomoPatchData(
        coords_yx_A=coords, rotations=rots, tilt_indices=torch.zeros(batch, dtype=torch.long)
    )


class TestVoxelGrid:
    def test_voxel_center_query_matches_indexing(self):
        torch.manual_seed(0)
        vol = torch.randn(5, 6, 7)
        vg = VoxelGrid(vol.clone())
        zz, yy, xx = torch.meshgrid(_axis(5), _axis(6), _axis(7), indexing="ij")
        pts = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)
        vals = vg(pts).reshape(5, 6, 7)
        assert torch.allclose(vals, vol, atol=1e-6)

    def test_midpoint_average(self):
        vol = torch.zeros(3, 3, 3)
        vol[1, 1, 1], vol[1, 1, 2] = 2.0, 4.0
        vg = VoxelGrid(vol)
        # halfway between voxels (1,1,1) and (1,1,2): x between 0 and 1 normalized
        pt = torch.tensor([[0.0, 0.0, 0.5]])
        assert torch.allclose(vg(pt), torch.tensor([[3.0]]), atol=1e-6)

    def test_outside_is_vacuum(self):
        vg = VoxelGrid(torch.ones(4, 4, 4))
        pts = torch.tensor([[1.8, 0.0, 0.0], [0.0, -1.8, 0.0], [0.0, 0.0, 1.8]])
        assert torch.allclose(vg(pts), torch.zeros(3, 1), atol=1e-7)

    def test_gradient_flows(self):
        vg = VoxelGrid(torch.zeros(4, 4, 4))
        pts = torch.rand(10, 3) * 1.6 - 0.8
        vg(pts).sum().backward()
        assert vg.volume.grad is not None
        assert torch.isfinite(vg.volume.grad).all() and vg.volume.grad.abs().sum() > 0


class TestObjectVoxelTomoForward:
    def test_payload_contract_and_vacuum_init(self):
        obj = make_initialized_voxel_obj()
        pd = full_fov_payload(17, tilt_deg=35.0, batch=3)
        out = obj.forward(pd)
        assert out.shape == (4, 3, 17, 17)
        assert out.is_complex()
        assert torch.allclose(out, torch.ones_like(out), atol=1e-7)  # vacuum -> unit transmission

    def test_hot_plane_lands_in_correct_slice(self):
        """A single hot z-plane must contribute only to the geometrically matching multislice
        slab (D=9 voxels / S=4 slabs: plane 1 sits at slab 0's center, plane 7 at slab 3's).
        Catches slab-start vs slab-center convention bugs."""
        density = 0.3
        for hot_plane, hot_slice in [(1, 0), (7, 3)]:
            vol = torch.zeros(9, 17, 17)
            vol[hot_plane] = density
            obj = make_initialized_voxel_obj(volume=vol)
            out = obj.forward(full_fov_payload(17, tilt_deg=0.0))
            center = torch.angle(out)[:, 0, 8, 8]  # (S,)
            assert center[hot_slice] > 0.1 * density * obj.slab_thickness_A
            others = torch.cat([center[:hot_slice], center[hot_slice + 1 :]])
            assert others.abs().max() < 1e-6

    def test_uniform_density_gives_exact_slab_phase(self):
        """A uniform density must give phase = density * slab_thickness for every slab,
        independent of the quadrature count (slab-binning normalization invariant)."""
        density = 0.25
        vol = torch.full((8, 17, 17), density)
        for spp in (None, 1, 3, 7):
            obj = make_initialized_voxel_obj(volume=vol, volume_shape=(8, 17, 17))
            obj.samples_per_slab = spp
            out = obj.forward(full_fov_payload(17, tilt_deg=0.0))
            center = torch.angle(out)[:, 0, 8, 8]
            expected = density * obj.slab_thickness_A
            assert torch.allclose(center, torch.full_like(center, expected), atol=1e-6), (
                f"samples_per_slab={spp}"
            )

    def test_slab_binning_integrates_subslab_structure(self):
        """Rotate-then-bin: with quadrature points at z-voxel spacing (trilinear partition of
        unity), a single hot plane contributes phase = density * z_voxel to its slab regardless
        of where it sits within the slab — the midpoint rule (K=1) misses off-center planes."""
        density = 0.5
        thickness = 8.0
        d = 8  # z_voxel = 1.0 A
        s = 2  # slab 4.0 A -> auto K = 4 at z-voxel spacing
        for hot_plane in (1, 2, 5):
            vol = torch.zeros(d, 17, 17)
            vol[hot_plane] = density
            obj = make_initialized_voxel_obj(
                volume=vol, volume_shape=(d, 17, 17), thickness_A=thickness, num_slices=s
            )
            with torch.no_grad():
                out = obj.forward(full_fov_payload(17, tilt_deg=0.0))
            center = torch.angle(out)[:, 0, 8, 8]
            expected = density * obj.z_voxel_A
            assert center.sum().item() == pytest.approx(expected, rel=0.15), f"plane {hot_plane}"
            hot_slab = 0 if hot_plane < d // 2 else 1
            assert center[hot_slab].item() == pytest.approx(center.sum().item(), rel=1e-5)

    def test_zero_tilt_projection_matches_volume_projection(self):
        """At 0 deg the summed slice phases reproduce the volume's z-projection (cubic box)."""
        torch.manual_seed(1)
        n = 17
        # smooth random volume, vacuum at borders
        vol = torch.zeros(n, n, n)
        vol[4:-4, 4:-4, 4:-4] = torch.rand(n - 8, n - 8, n - 8)
        thickness = 8.0
        sampling = thickness / (n - 1)  # cubic box: h_z == h_y == h_x
        obj = make_initialized_voxel_obj(
            volume=vol,
            thickness_A=thickness,
            num_slices=n,  # slab centers densely sample z
            lateral_full=n,
            sampling=sampling,
        )
        out = obj.forward(full_fov_payload(n, tilt_deg=0.0, sampling=sampling))
        proj = torch.angle(out).sum(0)[0]  # (H, W)
        # Riemann sum of trilinear-interpolated volume at slab centers vs direct voxel sum:
        # compare against the same interpolation evaluated densely -> use voxel sum with tolerance
        ref = vol.sum(0) * (thickness / (n - 1)) * ((n - 1) / n)  # slab sum ~ trapezoid-ish
        corr = torch.corrcoef(torch.stack([proj.flatten(), ref.flatten()]))[0, 1]
        assert corr > 0.999

    def test_rotated_projection_matches_analytic_reference(self):
        """Synthetic M0: forward projections through the rotated voxel object match the same
        analytic phantom evaluated at rotated coordinates (asymmetric -> catches sign flips)."""
        n = 33
        thickness = 16.0
        sampling = thickness / (n - 1)
        half = thickness / 2.0

        # asymmetric 3-gaussian phantom (positions in Å, specimen frame, no symmetry)
        centers = torch.tensor([[-3.0, 1.5, 4.0], [2.5, -3.5, -1.0], [4.5, 4.0, -4.5]])
        amps = torch.tensor([1.0, 0.7, 1.3])
        sigma = 1.6

        def analytic_density(pts_A: torch.Tensor) -> torch.Tensor:
            # pts_A (..., 3) physical specimen coords -> density
            d2 = ((pts_A[..., None, :] - centers) ** 2).sum(-1)  # (..., 3)
            return (amps * torch.exp(-d2 / (2 * sigma**2))).sum(-1)

        # rasterize at 0 deg on the volume grid
        ax = _axis(n) * half
        zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing="ij")
        vol0 = analytic_density(torch.stack([zz, yy, xx], dim=-1))

        obj = make_initialized_voxel_obj(
            volume=vol0,
            thickness_A=thickness,
            num_slices=n,
            lateral_full=n,
            sampling=sampling,
        )

        lat = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), dim=-1)  # (n, n, 2) physical
        zc = obj._slab_z_centers_t  # (n,)
        for tilt in [0.0, 30.0, -30.0, 90.0, 70.0]:
            out = obj.forward(full_fov_payload(n, tilt_deg=tilt, sampling=sampling))
            proj = torch.angle(out).sum(0)[0]

            # analytic reference: evaluate the phantom at the SAME rotated beam-frame points
            R = rot_beam_to_spec(0.0, tilt, 0.0)[0]
            beam = torch.cat(
                [
                    zc.view(-1, 1, 1, 1).expand(n, n, n, 1),
                    lat.view(1, n, n, 2).expand(n, n, n, 2),
                ],
                dim=-1,
            )  # (S, H, W, 3) physical (z, y, x)
            spec = torch.einsum("ij,shwj->shwi", R, beam)
            ref = (analytic_density(spec)).sum(0) * obj.slab_thickness_A

            corr = torch.corrcoef(torch.stack([proj.flatten(), ref.flatten()]))[0, 1]
            assert corr > 0.995, f"tilt {tilt}: corr {corr:.4f}"

    def test_90deg_projection_is_volume_y_sum(self):
        """Fully independent 90-deg check: projecting the object tilted by 90 deg about x equals
        summing the 0-deg volume over its y axis (rows of the projection = specimen z)."""
        torch.manual_seed(2)
        n = 21
        thickness = 10.0
        sampling = thickness / (n - 1)
        vol = torch.zeros(n, n, n)
        vol[5:-5, 5:-5, 5:-5] = torch.rand(n - 10, n - 10, n - 10)
        # smooth it so trilinear interpolation error is small
        vol = torch.nn.functional.avg_pool3d(vol[None, None], 3, stride=1, padding=1)[0, 0]
        obj = make_initialized_voxel_obj(
            volume=vol, thickness_A=thickness, num_slices=n, lateral_full=n, sampling=sampling
        )
        out = obj.forward(full_fov_payload(n, tilt_deg=90.0, sampling=sampling))
        proj = torch.angle(out).sum(0)[0]
        ref = vol.sum(1) * obj.slab_thickness_A  # rows = specimen z, cols = x
        corr = torch.corrcoef(torch.stack([proj.flatten(), ref.flatten()]))[0, 1]
        assert corr > 0.99

    def test_forward_gradient_reaches_volume(self):
        obj = make_initialized_voxel_obj()
        out = obj.forward(full_fov_payload(17, tilt_deg=35.0))
        out.imag.sum().backward()
        model = obj._model
        assert isinstance(model, VoxelGrid)
        assert model.volume.grad is not None and model.volume.grad.abs().sum() > 0


class TestObjectVoxelTomoStateAndConstraints:
    def test_materialized_obj_and_volume(self):
        vol = torch.rand(5, 7, 7)
        obj = make_initialized_voxel_obj(volume=vol, volume_shape=(5, 7, 7))
        assert torch.allclose(obj.obj, vol, atol=1e-7)
        assert torch.allclose(obj.volume, vol, atol=1e-7)

    def test_reset_restores_initial_volume(self):
        vol = torch.rand(5, 7, 7)
        obj = make_initialized_voxel_obj(volume=vol)
        model = obj._model
        assert isinstance(model, VoxelGrid)
        with torch.no_grad():
            model.volume.add_(1.0)
        assert not torch.allclose(obj.volume, vol)
        obj.reset()
        assert torch.allclose(obj.volume, vol, atol=1e-7)

    def test_soft_constraints_finite_and_differentiable(self):
        vol = torch.rand(6, 8, 8) - 0.2  # some negative values for positivity
        obj = make_initialized_voxel_obj(volume=vol, volume_shape=(6, 8, 8))
        obj.constraints = {"tv_weight": 0.1, "positivity_weight": 0.5}
        loss = obj.apply_soft_constraints()
        assert torch.isfinite(loss) and loss > 0
        loss.backward()
        model = obj._model
        assert isinstance(model, VoxelGrid)
        assert model.volume.grad is not None and torch.isfinite(model.volume.grad).all()

    def test_zero_constraints_zero_loss(self):
        obj = make_initialized_voxel_obj()
        loss = obj.apply_soft_constraints()
        assert loss.item() == 0.0

    def test_voxel_has_no_pretrain(self):
        # warm starts go through set_volume; pretraining is the K-Planes backend's concern
        obj = make_initialized_voxel_obj(volume_shape=(5, 9, 9))
        assert not hasattr(obj, "pretrain")

    def test_box_half_extents(self):
        obj = make_initialized_voxel_obj(
            thickness_A=8.0, lateral_full=17, sampling=0.5, num_slices=4
        )
        h_z, h_y, h_x = obj._box_half_extents
        assert h_z == pytest.approx(4.0)
        assert h_y == pytest.approx(4.0)  # (17-1)/2*0.5
        assert h_x == pytest.approx(4.0)

    def test_requires_initialize(self):
        obj = ObjectVoxelTomo.from_uniform((5, 7, 7), thickness_A=8.0, num_slices=4)
        with pytest.raises(ValueError, match="preprocess"):
            _ = obj._box_half_extents

    def test_set_geometry_rejects_anisotropic_sampling(self):
        """set_geometry requires equal y/x sampling (cubic voxels)."""
        obj = ObjectVoxelTomo.from_uniform(thickness_A=8.0, num_slices=4)
        with pytest.raises(ValueError, match="equal y/x sampling"):
            obj.set_geometry(lateral_box_A=(8.0, 8.0), sampling=(0.5, 0.6))

    def test_set_geometry_matches_legacy_initialize(self):
        """set_geometry and the legacy pixel handshake produce identical box half-extents."""
        obj_legacy = ObjectVoxelTomo.from_uniform(thickness_A=8.0, num_slices=4)
        obj_legacy._initialize_obj((4, 17, 17), sampling=(0.5, 0.5))
        obj_new = ObjectVoxelTomo.from_uniform(thickness_A=8.0, num_slices=4)
        # 17 grid points at 0.5 Å/px span (17 - 1) * 0.5 = 8 Å (point-grid convention)
        obj_new.set_geometry(lateral_box_A=(8.0, 8.0), sampling=(0.5, 0.5))
        assert obj_new._box_half_extents == obj_legacy._box_half_extents
        # lateral grids identical; z uses the point convention (round(box/s) + 1 points) while
        # the legacy pixel handshake used the voxel convention (round(box/s)) — one more point
        assert obj_new.volume_shape[1:] == obj_legacy.volume_shape[1:]
        assert obj_new.volume_shape[0] == obj_legacy.volume_shape[0] + 1


class TestZPadding:
    def test_box_extends_and_crop_target_unchanged(self):
        obj = ObjectVoxelTomo.from_uniform(thickness_A=8.0, num_slices=4)
        obj.set_z_padding_A(2.0)
        assert obj.box_thickness_A == pytest.approx(12.0)
        assert obj.thickness_A == pytest.approx(8.0)  # specimen thickness unchanged
        assert obj.slab_thickness_A == pytest.approx(3.0)  # slabs span the padded box
        assert obj.slice_thicknesses is not None
        assert torch.allclose(obj.slice_thicknesses, torch.full((3,), 3.0))
        obj._initialize_obj((4, 17, 17), sampling=(0.5, 0.5))
        h_z, _hy, _hx = obj._box_half_extents
        assert h_z == pytest.approx(6.0)
        # cubic default z count follows the padded box
        assert obj.volume_shape[0] == round(12.0 / 0.5)

    def test_padded_box_covers_rotated_lateral_probes(self):
        """At high tilt, points near the lateral edge rotate to |z| > thickness/2; with z padding
        they stay inside the box (density there is representable, not force-masked to vacuum)."""
        thickness, lat_half = 8.0, 8.0  # h_y = 8 at 70 deg -> |z_spec| up to ~8.4 > 4
        obj_pad = ObjectVoxelTomo.from_uniform(thickness_A=thickness, num_slices=4)
        obj_pad.set_z_padding_A(6.0)  # box half-z = 10 > 8.4
        obj_pad._initialize_obj((4, 33, 33), sampling=(2 * lat_half / 32, 2 * lat_half / 32))
        # uniform density: with full coverage every slab phase = density * slab_thickness even
        # for edge patches at 70 deg... only true where the rotated point stays in the box;
        # instead check the masked fraction shrinks vs the unpadded object
        obj_nopad = ObjectVoxelTomo.from_uniform(thickness_A=thickness, num_slices=4)
        obj_nopad._initialize_obj((4, 33, 33), sampling=(2 * lat_half / 32, 2 * lat_half / 32))

        def covered_fraction(obj, density=0.05):  # small density: slab phase stays << pi
            with torch.no_grad():
                obj.set_volume(torch.full(obj.volume_shape, density), set_as_initial=False)
                out = obj.forward(full_fov_payload(33, tilt_deg=70.0, sampling=2 * lat_half / 32))
                phase = torch.angle(out)
                expected = density * obj.slab_thickness_A  # full coverage value
                return (phase > 0.95 * expected).float().mean().item()

        assert covered_fraction(obj_pad) > covered_fraction(obj_nopad)


class TestHardPositivity:
    def test_project_parameters_clamps_potential(self):
        vol = torch.rand(5, 7, 7) - 0.5
        obj = make_initialized_voxel_obj(volume=vol, volume_shape=(5, 7, 7))
        obj.project_parameters()  # constraints.positivity defaults True
        assert (obj.volume >= 0).all()

    def test_opt_out_and_pure_phase_untouched(self):
        vol = torch.rand(5, 7, 7) - 0.5
        obj = make_initialized_voxel_obj(volume=vol, volume_shape=(5, 7, 7))
        obj.constraints = {"positivity": False}  # hard positivity now lives in the constraints
        obj.project_parameters()
        assert (obj.volume < 0).any()
        obj2 = make_initialized_voxel_obj(
            volume=vol, volume_shape=(5, 7, 7), obj_type="pure_phase"
        )
        obj2.project_parameters()
        assert (torch.as_tensor(obj2._model.volume.detach()) < 0).any()

    def test_shrink_quantile_pins_vacuum_planes(self):
        vol = torch.zeros(6, 10, 10)
        vol += 0.05  # uniform haze
        vol[2, 4:6, 4:6] = 1.0  # feature on plane 2
        obj = make_initialized_voxel_obj(volume=vol, volume_shape=(6, 10, 10))
        obj.constraints = {"shrink_quantile": 0.3}
        obj.project_parameters()
        out = obj.volume
        # haze-only planes pinned to 0; the feature survives (shifted by the plane floor)
        assert out[0].abs().max() < 1e-6
        assert out[2].max() > 0.9

    def test_constraint_keys_validated(self):
        obj = make_initialized_voxel_obj()
        with pytest.raises(KeyError, match="Invalid constraint key"):
            obj.constraints = {"identical_slices": True}  # a Raster key, not a Volume key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
