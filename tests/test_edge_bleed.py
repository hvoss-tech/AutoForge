"""Edge bleed strength (--edge_bleed / webui "Edge bleed"): every composite
path - the torch composites and both fused kernels, forward and backward -
must use the configured strength, and agree with each other at it."""
import pytest
import torch

from autoforge.Helper import FusedComposite as fc
from autoforge.Helper import OptimizerHelper as oh
from autoforge.Helper.OptimizerHelper import composite_image_cont, get_edge_bleed, set_edge_bleed

cuda_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and fc._HAS_TRITON), reason="needs CUDA + Triton"
)


@pytest.fixture(autouse=True)
def _restore_strength():
    before = get_edge_bleed()
    yield
    set_edge_bleed(before)


def _inputs(seed=0, H=20, W=24, L=12, M=4):
    g = torch.Generator().manual_seed(seed)
    dev = "cuda"
    pl = (torch.randn(H, W, generator=g) * 3).to(dev)
    gl = torch.randn(L, M, generator=g).to(dev)
    mc = torch.rand(M, 3, generator=g).to(dev)
    td = (torch.rand(M, generator=g) * 4 + 0.5).to(dev)
    bg = torch.rand(3, generator=g).to(dev)
    ge = torch.empty(L, M).exponential_(generator=g).to(dev)
    return pl, gl, mc, td, bg, ge


def _cont(pl, gl, mc, td, bg, ge, fused, monkeypatch):
    monkeypatch.setattr(fc, "_HAS_TRITON", fused)
    x = pl.clone().requires_grad_(True)
    out = composite_image_cont(x, gl, 1.0, 1.0, 0.04, gl.shape[0], mc, td, bg, gumbel_exp=ge)
    (out * torch.linspace(0, 1, out.numel(), device=out.device).view_as(out)).sum().backward()
    return out.detach(), x.grad.detach()


@cuda_triton
@pytest.mark.parametrize("strength", [0.0, 0.1, 0.3])
def test_fused_training_composite_matches_torch_at_every_strength(strength, monkeypatch):
    if not getattr(oh, "_USE_FUSED_CONT", True):
        pytest.skip("fused training composite disabled")
    set_edge_bleed(strength)
    args = _inputs()
    out_f, g_f = _cont(*args, fused=True, monkeypatch=monkeypatch)
    out_t, g_t = _cont(*args, fused=False, monkeypatch=monkeypatch)
    assert torch.allclose(out_f, out_t, atol=0.05, rtol=1e-3)
    assert torch.allclose(g_f, g_t, atol=1e-3 * g_t.abs().max().item() + 1e-6, rtol=1e-2)


@cuda_triton
def test_strength_changes_the_composite(monkeypatch):
    args = _inputs(seed=1)
    set_edge_bleed(0.0)
    a, _ = _cont(*args, fused=True, monkeypatch=monkeypatch)
    set_edge_bleed(0.3)
    b, _ = _cont(*args, fused=True, monkeypatch=monkeypatch)
    assert (a - b).abs().max() > 1.0


@cuda_triton
@pytest.mark.parametrize("strength", [0.0, 0.1, 0.3])
def test_fused_discrete_composite_matches_torch_at_every_strength(strength):
    from autoforge.webui.helpers.slider_render import composite_from_slider_stack

    set_edge_bleed(strength)
    g = torch.Generator().manual_seed(2)
    L, M = 10, 4
    z = torch.randint(0, L + 1, (18, 22), generator=g).cuda()
    dg = torch.randint(0, M, (L,), generator=g).cuda()
    mc = torch.rand(M, 3, generator=g).cuda()
    td = (torch.rand(M, generator=g) * 4 + 0.5).cuda()
    bg = torch.rand(3, generator=g).cuda()
    fused = fc.composite_heights(z.to(torch.int32), fc.stack_params(dg, mc, td, bg, 0.04), fc.background_tensor(bg), 0.04)
    ref = composite_from_slider_stack(z, dg.cpu().numpy(), mc, td, bg, 0.04, L)
    fused = fused * 255.0 if fused.max() <= 1.5 else fused
    assert torch.allclose(fused.float(), ref.float(), atol=0.1)


def test_cli_flag_sets_the_process_strength():
    import sys

    from autoforge import auto_forge

    argv = sys.argv
    try:
        sys.argv = ["autoforge", "--input_image", "x.png", "--csv_file", "x.csv", "--edge_bleed", "0.25"]
        args = auto_forge.parse_args()
    finally:
        sys.argv = argv
    assert args.edge_bleed == pytest.approx(0.25)
