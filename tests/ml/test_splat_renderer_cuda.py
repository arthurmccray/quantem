"""The ``backend`` switch of ``render_projections``: the fused ``quantem.cuda`` stamp kernel
against the torch path of ``test_splat_renderer.py``. The GPU tests skip without a CUDA device or
without a ``quantem.cuda`` wheel that has ``splat_stamp``; the fallback test runs on the CPU."""

import sys
import types
import warnings

import pytest
import torch

from quantem.core.ml.splat import renderer as R
from quantem.core.ml.splat.renderer import render_projections, window_radius_for
from quantem.tomography.utils import pose_to_rotations


def _have_cuda_op() -> bool:
    try:
        from quantem.cuda.core.ml import splat_stamp  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


requires_cuda_op = pytest.mark.skipif(
    not _have_cuda_op(), reason="requires a CUDA device and quantem.cuda with splat_stamp"
)

SAMPLING = 0.5  # Angstrom / px
SHAPE = (32, 40)


@pytest.fixture(scope="module")
def scene():
    """60 atoms over the field of view (some projecting outside it at the steep tilts), per-atom
    widths 0.6-1.0 Angstrom, three tilts with in-plane angles and shifts; fp32 on the GPU."""
    gen = torch.Generator().manual_seed(3)
    pos = (torch.rand(60, 3, generator=gen) - 0.5) * torch.tensor([18.0, 14.0, 8.0])
    amp = 0.5 + torch.rand(60, generator=gen)
    sig = 0.6 + 0.4 * torch.rand(60, generator=gen)
    tilts = torch.tensor([-55.0, 10.0, 65.0])
    rot = pose_to_rotations(tilts, torch.tensor([2.0, 0.0, -3.0]), torch.tensor([-1.0, 0.0, 4.0]))
    shifts = torch.tensor([[0.4, -1.2], [0.0, 0.0], [-0.7, 0.9]])
    bg = torch.tensor([0.05, 0.1, 0.15])
    r = window_radius_for(float(sig.max()), SAMPLING)
    t = (pos, amp, sig, rot, shifts, bg)
    return tuple(x.to("cuda", torch.float32) for x in t) + (r,)


def _rel_err(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max())


@requires_cuda_op
def test_cuda_backend_matches_torch_images(scene, monkeypatch):
    """Keyword and environment-variable selection, tilt chunking, background and a broadcast
    (1,) sigma all give the torch image to 1e-5 of its maximum."""
    pos, amp, sig, rot, shifts, bg, r = scene
    args = (pos, amp, sig, rot, shifts, SAMPLING, SHAPE, r)
    ref = render_projections(*args, backend="torch")
    assert ref.abs().max() > 0
    monkeypatch.setenv(R.BACKEND_ENV, "cuda")
    cases = {
        "keyword": (render_projections(*args, backend="cuda"), ref),
        "env var": (render_projections(*args), ref),
        "tilt_chunk=1": (render_projections(*args, tilt_chunk=1, backend="cuda"), ref),
        "background": (
            render_projections(*args, background=bg, backend="cuda"),
            render_projections(*args, background=bg, backend="torch"),
        ),
        "sigma (1,)": (
            render_projections(pos, amp, sig[:1], rot, shifts, SAMPLING, SHAPE, r, backend="cuda"),
            render_projections(
                pos, amp, sig[:1], rot, shifts, SAMPLING, SHAPE, r, backend="torch"
            ),
        ),
    }
    for name, (img, expected) in cases.items():
        assert img.shape == expected.shape, name
        err = _rel_err(img, expected)
        assert err < 1e-5, f"{name}: relative image error {err:.2e}"
    # the env var is read at call time: back to torch without re-importing (torch's own CUDA
    # scatter_add is not bit-reproducible, so check the resolved backend, not bit equality)
    assert R.resolve_backend(None)[0] == "cuda"
    monkeypatch.setenv(R.BACKEND_ENV, "torch")
    assert R.resolve_backend(None) == ("torch", None)
    assert _rel_err(render_projections(*args), ref) < 1e-6


@requires_cuda_op
def test_cuda_backend_grads_match_fp64(scene):
    """Gradients to positions, amplitudes, sigmas, rotations and shifts: the CUDA backend's
    deviation from the fp64 torch reference is within 2x the fp32 torch path's own (or 1e-5 of
    the largest gradient entry)."""
    pos, amp, sig, rot, shifts, bg, r = scene
    upstream = torch.randn(
        3, *SHAPE, device="cuda", generator=torch.Generator("cuda").manual_seed(7)
    )

    def grads(dtype, backend):
        ins = tuple(
            t.detach().to(dtype).requires_grad_(True) for t in (pos, amp, sig, rot, shifts)
        )
        out = render_projections(*ins, SAMPLING, SHAPE, r, backend=backend)
        return torch.autograd.grad((out * upstream.to(dtype)).sum(), ins)

    g64 = grads(torch.float64, "torch")
    g_torch = grads(torch.float32, "torch")
    g_cuda = grads(torch.float32, "cuda")
    names = ("positions", "amplitudes", "sigmas", "rotations", "shifts")
    for name, gt, gc, gd in zip(names, g_torch, g_cuda, g64):
        err_torch = (gt.double() - gd).abs().max().item()
        err_cuda = (gc.double() - gd).abs().max().item()
        bar = max(2.0 * err_torch, max(1e-6, 1e-5 * gd.abs().max().item()))
        assert err_cuda <= bar, (
            f"grad {name}: cuda err {err_cuda:.3e}, torch fp32 err {err_torch:.3e}"
        )


@requires_cuda_op
def test_cuda_backend_rejects_fp64_and_cpu(scene):
    pos, amp, sig, rot, shifts, bg, r = scene
    with pytest.raises(TypeError, match="fp32-only"):
        render_projections(
            pos.double(),
            amp.double(),
            sig.double(),
            rot.double(),
            shifts.double(),
            SAMPLING,
            SHAPE,
            r,
            backend="cuda",
        )
    with pytest.raises(ValueError, match="needs CUDA tensors"):
        render_projections(
            pos.cpu(),
            amp.cpu(),
            sig.cpu(),
            rot.cpu(),
            shifts.cpu(),
            SAMPLING,
            SHAPE,
            r,
            backend="cuda",
        )


def test_env_cuda_is_a_preference_keyword_a_demand(monkeypatch):
    """With the backend from the environment, CPU / fp64 inputs take the torch path silently
    (no kernel call, no warning); the explicit keyword raises on the same inputs."""
    pos = torch.tensor([[1.3, -0.7, 2.1], [-2.2, 1.9, -1.4]], dtype=torch.float64)
    amp = torch.tensor([1.0, 0.7], dtype=torch.float64)
    sig = torch.full((2,), 0.8, dtype=torch.float64)
    rot = pose_to_rotations(torch.tensor([-20.0, 50.0]), torch.zeros(2), torch.zeros(2)).double()
    shifts = torch.zeros(2, 2, dtype=torch.float64)
    args = (pos, amp, sig, rot, shifts, SAMPLING, (16, 16), 6)
    ref = render_projections(*args, backend="torch")

    def stamp_stub(*a, **k):
        raise AssertionError("the kernel must not be called for CPU inputs")

    monkeypatch.setattr(R, "_cuda_stamp_or_none", lambda: stamp_stub)
    monkeypatch.setenv(R.BACKEND_ENV, "cuda")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = render_projections(*args)
    assert torch.equal(out, ref)
    assert R.resolve_backend(None, (pos, amp, sig)) == ("torch", None)
    assert R.resolve_backend(None, (pos.float(), amp.float(), sig.float()))[0] == "torch"  # CPU
    with pytest.raises((TypeError, ValueError)):
        render_projections(*args, backend="cuda")


def test_cuda_fallback_warns_once_and_uses_torch(monkeypatch):
    """Without quantem.cuda, or with an older wheel lacking the op, backend='cuda' renders with
    the torch path and warns once per process; unknown backends are rejected."""
    pos = torch.tensor([[1.3, -0.7, 2.1], [-2.2, 1.9, -1.4]], dtype=torch.float64)
    amp = torch.tensor([1.0, 0.7], dtype=torch.float64)
    sig = torch.full((2,), 0.8, dtype=torch.float64)
    rot = pose_to_rotations(torch.tensor([-20.0, 50.0]), torch.zeros(2), torch.zeros(2)).double()
    shifts = torch.zeros(2, 2, dtype=torch.float64)
    args = (pos, amp, sig, rot, shifts, SAMPLING, (16, 16), 6)
    ref = render_projections(*args, backend="torch")
    stale = types.ModuleType("quantem.cuda.core.ml")  # an older wheel: importable, no splat_stamp
    stale.__file__ = "/stale/quantem/cuda/core/ml/__init__.py"
    fake_pkg = types.ModuleType("quantem.cuda")
    fake_pkg.__version__ = "0.1.0+stale"
    fake_pkg.__file__ = "/stale/quantem/cuda/__init__.py"
    for name, modules, msg in (
        ("missing", {"quantem.cuda.core.ml": None}, "not importable"),
        (
            "older wheel",
            {"quantem.cuda.core.ml": stale, "quantem.cuda": fake_pkg},
            "no splat_stamp",
        ),
    ):
        for key, mod in modules.items():
            monkeypatch.setitem(sys.modules, key, mod)
        monkeypatch.setattr(R, "_cuda_fallback_warned", False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = render_projections(*args, backend="cuda")
            again = render_projections(*args, backend="cuda")
        assert torch.equal(out, ref) and torch.equal(again, ref), name
        msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(msgs) == 1 and msg in msgs[0], f"{name}: {msgs}"
        if name == "older wheel":
            assert "0.1.0+stale" in msgs[0] and "/stale/" in msgs[0]
    with pytest.raises(ValueError, match="unknown splat backend"):
        render_projections(*args, backend="numpy")
