"""Tests for ``quantem.tomography.candidate_classifier``: masking of candidate senders, scale invariance, checkpoint
round trip (CPU, tiny random-init models on a small gold octahedron)."""

import numpy as np
import pytest
import torch

from quantem.tomography.candidate_classifier import CandidateClassifier, ClassifierConfig

A0 = 4.078  # Angstrom, gold
D1 = A0 / np.sqrt(2.0)
TINY = dict(dim=16, num_convs=3, d1=D1)


@pytest.fixture(scope="module", autouse=True)
def _few_threads():
    # tiny graphs; 4 threads avoid the login node's oversubscription
    n = torch.get_num_threads()
    torch.set_num_threads(min(4, n))
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def structure() -> tuple[torch.Tensor, torch.Tensor]:
    """Atoms of a 44-atom octahedron with three atoms removed (their sites are the first three candidates) plus
    one candidate outside the surface."""
    from ase.cluster import Octahedron

    x = Octahedron("Au", 4, latticeconstant=A0).positions
    centre = np.argmin(np.linalg.norm(x - x.mean(0), axis=1))
    removed = [centre, 0, 5]
    keep = np.setdiff1d(np.arange(len(x)), removed)
    outside = x[0] + 1.5 * (x[0] - x.mean(0)) / np.linalg.norm(x[0] - x.mean(0)) * D1
    cands = np.concatenate([x[removed], outside[None]])
    return torch.tensor(x[keep], dtype=torch.float32), torch.tensor(cands, dtype=torch.float32)


def _model(kind: str, seed: int = 0) -> CandidateClassifier:
    torch.manual_seed(seed)
    return CandidateClassifier(ClassifierConfig(kind=kind, **TINY)).eval()


def _run(model, atoms, cands):
    pos = torch.cat([atoms, cands])
    is_c = torch.zeros(len(pos), dtype=torch.bool)
    is_c[len(atoms) :] = True
    with torch.no_grad():
        return model(pos, is_c, return_hidden=True)


def test_candidates_receive_but_do_not_send(structure):
    atoms, cands = structure
    model = _model("add")
    n = len(atoms)
    logits0, h0 = _run(model, atoms, cands)
    moved = cands.clone()
    moved[0] += torch.tensor([0.3, -0.2, 0.1])
    variants = {"candidate moved": moved, "candidate deleted": cands[[0, 2, 3]]}
    for name, c in variants.items():
        _, h = _run(model, atoms, c)
        assert torch.equal(h[:n], h0[:n]), f"{name}: a real atom's state depends on a candidate"
    # a real atom next to candidate 0 (the central vacancy) moves by 0.5 A: the candidate's state and logit must change
    # (a random-init net is weakly sensitive; the thresholds are far above float32 round-off)
    j = int(torch.argmin((atoms - cands[0]).norm(dim=1)))
    a2 = atoms.clone()
    a2[j] += torch.tensor([0.5, 0.0, 0.0])
    logits2, h2 = _run(model, a2, cands)
    dh = float((h2[n] - h0[n]).norm() / h0[n].norm())
    dl = abs(float(logits2[n] - logits0[n]))
    assert dh > 1e-4 and dl > 1e-7, (
        f"candidate state ignores its real neighbour (relative change {dh:.1e}, logit {dl:.1e})"
    )
    # the same move of the candidate with the mask switched off would reach the real atoms (the test can fail)
    g0 = model.featurize(torch.cat([atoms, cands]), None)
    g1 = model.featurize(torch.cat([atoms, moved]), None)
    with torch.no_grad():
        _, hu0 = model.forward_graph(g0, return_hidden=True)
        _, hu1 = model.forward_graph(g1, return_hidden=True)
    assert not torch.equal(hu0[:n], hu1[:n]), (
        "unmasked control: moving a site did not reach the real atoms"
    )


def test_scale_invariance(structure):
    atoms, cands = structure
    model = _model("add")
    p = model.score_candidates(atoms, cands)
    f = 1.37
    q = model.score_candidates(atoms * f, cands * f, d1=D1 * f)
    lp, lq = torch.logit(p.double()), torch.logit(q.double())
    assert torch.allclose(lp, lq, atol=1e-5, rtol=0), (
        f"max logit change {float((lp - lq).abs().max()):.2e}"
    )
    r = model.score_candidates(atoms * f, cands * f)  # without the matching d1 the graph changes
    assert not torch.allclose(p, r, atol=1e-4), "scaling without d1 left the output unchanged"


def test_checkpoint_round_trip(structure, tmp_path):
    atoms, cands = structure
    for kind, seed in (("add", 1), ("remove", 2)):
        model = _model(kind, seed)
        path = tmp_path / f"{kind}.ckpt"
        model.save_checkpoint(path, extra={"note": kind})
        loaded = CandidateClassifier.from_checkpoint(path, device="cpu")
        assert loaded.kind == kind and loaded.d1 == pytest.approx(D1), kind
        if kind == "add":
            a, b = model.score_candidates(atoms, cands), loaded.score_candidates(atoms, cands)
        else:
            a, b = model.score_atoms(atoms), loaded.score_atoms(atoms)
        assert torch.equal(a, b), f"{kind}: logits differ after the round trip"
        assert float(a.std()) > 0, f"{kind}: constant output, the comparison is empty"
