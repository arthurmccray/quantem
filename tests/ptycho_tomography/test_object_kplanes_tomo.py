"""Tests for the K-Planes ptycho-tomography object backend (M2).

PPLR wiring (per-parameter-group optimizers incl. the reset/disable sentinel) mirrors the
coverage of test_object_tensor_decomp.py; the rotated-forward contract is shared with the voxel
backend and tested through the same payload interface. A compact inverse-crime reconstruction
smoke exercises the PPLR fan-out through the full PtychoTomography loop.
"""

from typing import cast

import numpy as np
import pytest
import torch

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.ml import OptimizerParams
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffractive_imaging.detector_models import DetectorPixelated
from quantem.diffractive_imaging.probe_models import ProbePixelated
from quantem.ptycho_tomography.dataset_models import PtychoTomoDatasetRaster
from quantem.ptycho_tomography.geometry import PtychoTomoPatchData, rot_beam_to_spec
from quantem.ptycho_tomography.object_models import ObjectKPlanesTomo, ObjectVoxelTomo
from quantem.ptycho_tomography.ptycho_tomography import PtychoTomography

PPLR = {
    "grids": OptimizerParams.Adam(lr=1e-2),
    "sigma_net": OptimizerParams.Adam(lr=1e-3),
}


def make_kplanes_obj(**kwargs) -> ObjectKPlanesTomo:
    defaults = dict(
        thickness_A=8.0,
        num_slices=4,
        num_z_voxels=8,
        M_features=8,
        resolution=(16, 16, 16),
        multiscale_res_multipliers=(0.5, 1.0),
        rng=0,
    )
    defaults.update(kwargs)
    return ObjectKPlanesTomo.from_uniform(**defaults)


def initialized(obj: ObjectKPlanesTomo, lateral: int = 17, sampling: float = 0.5):
    obj._initialize_obj((obj.num_slices, lateral, lateral), sampling=(sampling, sampling))
    return obj


def payload(lateral: int, tilt: float, batch: int = 2) -> PtychoTomoPatchData:
    ax = torch.linspace(-1, 1, lateral)
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    coords = torch.stack([gy, gx], dim=-1)[None].expand(batch, -1, -1, -1)
    rots = rot_beam_to_spec(0.0, torch.full((batch,), tilt), 0.0)
    return PtychoTomoPatchData(coords, rots, torch.zeros(batch, dtype=torch.long))


class TestConstructionAndForward:
    def test_vacuum_init_unit_transmission(self):
        obj = initialized(make_kplanes_obj())
        out = obj.forward(payload(17, 35.0))
        assert out.shape == (4, 2, 17, 17)
        assert torch.allclose(out, torch.ones_like(out), atol=1e-6)

    def test_forward_gradients_flow(self):
        obj = initialized(make_kplanes_obj())
        out = obj.forward(payload(17, 35.0))
        out.imag.sum().backward()
        # at vacuum init the decoder is zeroed, so grids get no gradient yet (chain rule);
        # the decoder itself must
        sigma_grads = [p.grad for p in obj.model.get_params()["sigma_net"]]
        assert any(g is not None and g.abs().sum() > 0 for g in sigma_grads)
        # after perturbing the decoder, gradient reaches the feature grids
        obj.zero_grad()
        with torch.no_grad():
            final = obj.model.sigma_net
            final = final[-1] if isinstance(final, torch.nn.Sequential) else final
            final.weight.normal_(0, 0.1)
        out = obj.forward(payload(17, 35.0))
        out.imag.sum().backward()
        grid_grads = [p.grad for p in obj.model.get_params()["grids"]]
        assert any(g is not None and g.abs().sum() > 0 for g in grid_grads)

    def test_from_model_rejects_non_kplanes(self):
        with pytest.raises(TypeError, match="KPlanes"):
            ObjectKPlanesTomo.from_model(torch.nn.Linear(3, 1), thickness_A=8.0)

    def test_tilted_variant_param_keys(self):
        obj = make_kplanes_obj(tilted=True, T=2)
        assert "so3" in obj.model.param_keys
        groups = obj.get_optimization_parameters()
        assert set(groups) == set(obj.model.param_keys)


class TestPPLRWiring:
    def test_param_groups_match_model(self):
        obj = make_kplanes_obj()
        groups = obj.get_optimization_parameters()
        assert set(groups) == {"grids", "sigma_net"}
        assert all(len(v) > 0 for v in groups.values())

    def test_set_optimizer_pplr_dict(self):
        obj = make_kplanes_obj()
        obj.set_optimizer(dict(PPLR))
        assert obj.optimizer is not None

    def test_single_optimizer_spec_rejected(self):
        obj = make_kplanes_obj()
        with pytest.raises(TypeError, match="param_keys"):
            obj.set_optimizer(OptimizerParams.Adam(lr=1e-2))
        with pytest.raises(TypeError, match="param_keys"):
            obj.set_optimizer({"name": "adam", "lr": 1e-2})

    def test_wrong_keys_rejected(self):
        obj = make_kplanes_obj()
        with pytest.raises(ValueError, match="param_keys"):
            obj.set_optimizer({"grids": OptimizerParams.Adam(lr=1e-2)})

    def test_disable_sentinel_accepted(self):
        obj = make_kplanes_obj()
        obj.set_optimizer(dict(PPLR))
        obj.set_optimizer({"default": OptimizerParams.NoneOptimizer()})  # reset_optimizer replay
        assert not obj.has_optimizer()


class TestPretrainAndState:
    def test_pretrain_fits_smooth_volume(self):
        obj = initialized(make_kplanes_obj(num_z_voxels=8))
        d, hh, ww = obj.volume_shape
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, d),
            torch.linspace(-1, 1, hh),
            torch.linspace(-1, 1, ww),
            indexing="ij",
        )
        target = torch.exp(-(zz**2 + yy**2 + xx**2) / 0.3)
        obj.pretrain(
            pretrain_target=target, num_iters=200, optimizer_params=dict(PPLR), show=False
        )
        vol = obj.volume
        corr = torch.corrcoef(torch.stack([vol.flatten(), target.flatten()]))[0, 1]
        assert corr > 0.95, f"pretrain correlation too low: {corr:.3f}"
        # pretrained weights are the reset state
        sd_before = {k: v.clone() for k, v in obj.model.state_dict().items()}
        with torch.no_grad():
            for p in obj.model.parameters():
                p.add_(0.5)
        obj.reset()
        for k, v in obj.model.state_dict().items():
            assert torch.allclose(v, sd_before[k], atol=1e-6)

    def test_soft_constraints_finite_and_differentiable(self):
        obj = initialized(make_kplanes_obj())
        # perturb decoder so output is nonzero / signable
        with torch.no_grad():
            final = obj.model.sigma_net
            final = final[-1] if isinstance(final, torch.nn.Sequential) else final
            final.weight.normal_(0, 0.1)
        obj.constraints = {"tv_weight_z": 0.1, "tv_weight_xy": 0.1, "positivity_weight": 0.5}
        loss = obj.apply_soft_constraints()
        assert torch.isfinite(loss)
        loss.backward()


# --------------------------------------------------------------------------- #
# integration smoke: PPLR through the full PtychoTomography loop
# --------------------------------------------------------------------------- #
N = 32
Q_MAX = 0.5
PROBE_ENERGY = 300e3
C10 = 30.0
SCAN_GPTS = (5, 5)
TILTS = [-45.0, 0.0, 45.0]
THICKNESS_A = 12.0


def _probe_array() -> np.ndarray:
    sampling = 1 / Q_MAX / 2
    reciprocal_sampling = 2 * Q_MAX / N
    qx = qy = np.fft.fftfreq(N, sampling)
    q = np.sqrt(qx[:, None] ** 2 + qy[None, :] ** 2)
    aperture = np.sqrt(np.clip((Q_MAX / 2 - q) / reciprocal_sampling + 0.5, 0, 1))
    chi = q**2 * electron_wavelength_angstrom(PROBE_ENERGY) * np.pi * C10
    pf = aperture * np.exp(-1j * chi)
    pf /= np.sqrt(np.sum(np.abs(pf) ** 2))
    return (np.fft.ifft2(pf) * N).astype(np.complex64)


def _make_ptycho(arrays, obj):
    recip = 2 * Q_MAX / N
    dsets = [
        Dataset4dstem.from_array(
            array=a.astype(np.float32),
            sampling=(2.0, 2.0, recip, recip),
            units=("A", "A", "A^-1", "A^-1"),
        )
        for a in arrays
    ]
    wrapper = PtychoTomoDatasetRaster.from_dataset4dstem_list(dsets, TILTS, verbose=0)
    probe = ProbePixelated.from_array(
        num_probes=1,
        probe_params={
            "energy": PROBE_ENERGY,
            "C10": C10,
            "semiangle_cutoff": electron_wavelength_angstrom(PROBE_ENERGY) * Q_MAX / 2 * 1e3,
        },
        probe_array=_probe_array(),
    )
    pt = PtychoTomography.from_models(
        wrapper, obj, probe, DetectorPixelated(), rng=0, verbose=False
    )
    pt.preprocess(obj_padding_px=(10, 10))
    return pt


@pytest.fixture(scope="module")
def kplanes_inverse_crime():
    rng = np.random.default_rng(0)
    placeholder = [
        rng.uniform(0.5, 1.0, size=(*SCAN_GPTS, N, N)).astype(np.float32) for _ in TILTS
    ]
    gt_obj = ObjectVoxelTomo.from_uniform(
        thickness_A=THICKNESS_A, num_slices=6, num_z_voxels=6, rng=0
    )
    pt0 = _make_ptycho(placeholder, gt_obj)
    d, hh, ww = gt_obj.volume_shape
    h_z, h_y, h_x = gt_obj._box_half_extents
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, d) * h_z,
        torch.linspace(-1, 1, hh) * h_y,
        torch.linspace(-1, 1, ww) * h_x,
        indexing="ij",
    )
    centers = torch.tensor([[-2.5, 1.5, 3.0], [2.0, -2.5, -1.0]])
    d2 = ((torch.stack([zz, yy, xx], -1)[..., None, :] - centers) ** 2).sum(-1)
    gt = (torch.tensor([0.4, 0.5]) * torch.exp(-d2 / (2 * 2.0**2))).sum(-1)
    gt_obj.set_volume(gt)
    preds = []
    with torch.no_grad():
        n = pt0.dset.num_gpts
        for s in range(0, n, 64):
            idx = torch.arange(s, min(s + 64, n))
            pd, _p, frac, desc = pt0.dset.forward(idx, pt0.obj_padding_px)
            probes = pt0.probe_model.forward(frac)
            patches = pt0.obj_model.forward(pd)
            _, ov = pt0.forward_operator(patches, probes, desc)
            preds.append(pt0.detector_model.forward(ov).cpu())
    preds = torch.cat(preds).numpy()
    npos = int(np.prod(SCAN_GPTS))
    arrays = [
        preds[i * npos : (i + 1) * npos].reshape((*SCAN_GPTS, N, N)) for i in range(len(TILTS))
    ]
    return arrays


class TestReconstructIntegration:
    def test_pplr_reconstruct_smoke_and_reset(self, kplanes_inverse_crime):
        obj = ObjectKPlanesTomo.from_uniform(
            thickness_A=THICKNESS_A,
            num_slices=6,
            num_z_voxels=6,
            M_features=8,
            resolution=(16, 24, 24),
            multiscale_res_multipliers=(0.5, 1.0),
            rng=0,
        )
        pt = _make_ptycho(kplanes_inverse_crime, obj)
        pt.reconstruct(
            num_iters=8,
            optimizer_params={"object": dict(PPLR)},
            batch_size=75,
        )
        losses = np.asarray(pt._iter_losses)
        assert np.isfinite(losses).all()
        assert losses[-1] < losses[0]
        # reset=True replays the disable sentinel through reset_optimizer -- must not raise
        pt.reconstruct(
            num_iters=2,
            reset=True,
            optimizer_params={"object": dict(PPLR)},
            batch_size=75,
        )

    def test_save_load_roundtrip(self, kplanes_inverse_crime, tmp_path):
        obj = ObjectKPlanesTomo.from_uniform(
            thickness_A=THICKNESS_A,
            num_slices=6,
            num_z_voxels=6,
            M_features=8,
            resolution=(16, 16, 16),
            rng=0,
        )
        pt = _make_ptycho(kplanes_inverse_crime, obj)
        pt.reconstruct(num_iters=2, optimizer_params={"object": dict(PPLR)}, batch_size=75)
        vol_before = pt.volume.copy()
        path = tmp_path / "kplanes_tomo.zip"
        pt.save(path, mode="o")  # raw data excluded by default -> reattach via from_file(dset=)
        wrapper2 = cast(PtychoTomoDatasetRaster, pt.dset)  # same wrapper, as the notebook flow
        loaded = PtychoTomography.from_file(path, dset=wrapper2)
        np.testing.assert_allclose(loaded.volume, vol_before, rtol=1e-4, atol=1e-5)
        assert loaded.dset.implicit_object is True  # re-synced by the from_file override
        loaded.reconstruct(num_iters=2, optimizer_params={"object": dict(PPLR)}, batch_size=75)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
