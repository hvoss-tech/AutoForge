"""Deterministic backwards for the two gathers of the trained height fields.

The per-cluster height offsets are gathered per pixel (``offsets[labels]``)
and the smooth height field is bilinearly upsampled; the stock CUDA backward
of both accumulates with float atomics, so the order of the additions - and
with it the gradient's last bits - changes from run to run. Through the
Gumbel sampling and the discrete checks that made whole runs diverge (the
same seed ended anywhere between 55 and 81 on a benchmark image). The
forwards here are the stock ones; the backwards sum in a fixed order.
Both are free of host syncs, so they work inside a captured CUDA graph.
"""
import torch
import torch.nn.functional as F


class _SegmentGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, labels, perm, ends):
        ctx.save_for_backward(perm, ends)
        return values[labels]

    @staticmethod
    def backward(ctx, grad):
        perm, ends = ctx.saved_tensors
        # Sum of each label's pixels: a float64 prefix sum over the pixels
        # sorted by label, differenced at the segment ends.
        cs = grad.reshape(-1)[perm].double().cumsum(0)
        cs = torch.cat([cs.new_zeros(1), cs])[ends]
        seg = torch.diff(cs, prepend=cs.new_zeros(1))
        return seg.to(grad.dtype), None, None, None


def gather_plan(labels: torch.Tensor, n: int):
    """(perm, ends) for ``segment_gather`` of ``n`` values by ``labels``."""
    flat = labels.reshape(-1).to(torch.long)
    perm = torch.argsort(flat, stable=True)
    ends = torch.cumsum(torch.bincount(flat, minlength=n), 0)
    return perm, ends


def segment_gather(values: torch.Tensor, labels: torch.Tensor, plan) -> torch.Tensor:
    """``values[labels]`` with a deterministic backward (see gather_plan)."""
    return _SegmentGather.apply(values, labels, *plan)


def _interp_matrix(n_out: int, n_in: int, device) -> torch.Tensor:
    """[n_out, n_in]: 1-D linear interpolation weights, as F.interpolate's
    bilinear mode (align_corners=False) applies them along one axis."""
    eye = torch.eye(n_in, device=device, dtype=torch.float32)
    return F.interpolate(eye[None, None], size=(n_out, n_in), mode="bilinear", align_corners=False)[0, 0]


class _BilinearUp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size, ah, aw):
        ctx.save_for_backward(ah, aw)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def backward(ctx, grad):
        ah, aw = ctx.saved_tensors
        g = grad.float()
        return (ah.t() @ g @ aw).to(grad.dtype), None, None, None


def bilinear_plan(size_in, size_out, device):
    """Interpolation matrices for ``bilinear_up`` from size_in to size_out."""
    return (
        _interp_matrix(size_out[0], size_in[0], device),
        _interp_matrix(size_out[1], size_in[1], device),
    )


def bilinear_up(x: torch.Tensor, size, plan) -> torch.Tensor:
    """F.interpolate(x, size, bilinear, align_corners=False) for x [1,1,h,w]
    with a deterministic backward."""
    return _BilinearUp.apply(x, tuple(size), *plan)
