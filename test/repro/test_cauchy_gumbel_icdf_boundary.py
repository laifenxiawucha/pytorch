# Repro for pytorch/pytorch#186824 (sandbox issue intel-sandbox/torch-xpu-ops-exp#1915).
# Cauchy.icdf(0) returned a large finite wrong-sign value in float32 and
# Gumbel.icdf(0/1) returned finite values; both supports are unbounded so the
# boundary quantiles must be -inf / +inf. Verified on xpu and cpu.
import torch
from torch.distributions import Cauchy, Gumbel


def _check(device):
    inf = float("inf")
    for ctor in (Cauchy, Gumbel):
        d = ctor(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        out = d.icdf(torch.tensor([0.0, 1.0], device=device))
        assert out[0].item() == -inf, (ctor.__name__, out.tolist())
        assert out[1].item() == inf, (ctor.__name__, out.tolist())


def test_icdf_boundary_cpu():
    _check("cpu")


def test_icdf_boundary_xpu():
    if not torch.xpu.is_available():
        import pytest

        pytest.skip("XPU not available")
    _check("xpu")


if __name__ == "__main__":
    test_icdf_boundary_cpu()
    if torch.xpu.is_available():
        test_icdf_boundary_xpu()
    print("OK")
