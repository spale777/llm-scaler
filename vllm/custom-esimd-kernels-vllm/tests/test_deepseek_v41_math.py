"""The DeepSeek V4.1 FP4 dequant and noaux_tc router arithmetic.

Both are pure arithmetic, so they are checked exactly on CPU against the
reference in deepseek-ai/DeepSeek-V4.1-Flash
inference/{model.py,kernel.py,convert.py}.
"""

import re
import struct
from pathlib import Path

import pytest

from srctext import code

_DS = Path(__file__).resolve().parents[1] / "csrc/deepseek_v41"
_DEQUANT = Path(__file__).resolve().parents[1] / "csrc/deepseek_v41/fp4_dequant.h"
_LUT = _DS / "fp4_dequant.h"
_TOPK = _DS / "topk_noaux_tc.h"
_KERNELS = Path(__file__).resolve().parents[1] / "csrc/xpu/deepseek_kernels.sycl"

# E2M1: 1 sign, 2 exponent, 1 mantissa. Reference clamps FP4 to +-6.0.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _f16_from_bits(b: int) -> float:
    return struct.unpack("e", struct.pack("H", b & 0xFFFF))[0]


def test_lut_matches_e2m1():
    src = _LUT.read_text()
    m = re.search(r"fp4_e2m1_lut\[16\]\s*=\s*\{(.*?)\};", src, re.S)
    assert m, "LUT not found"
    nums = [float(x) for x in re.findall(r"(-?\d+\.?\d*)f", m.group(1))]
    assert len(nums) == 16, f"expected 16 entries, got {len(nums)}"
    assert nums[:8] == E2M1, f"positive half wrong: {nums[:8]}"
    assert [abs(v) for v in nums[8:]] == E2M1, f"negative half wrong: {nums[8:]}"


def test_lut_is_not_eighth_scaled():
    """E2M1 / 8 would make every GEMM output 8x too small."""
    src = _LUT.read_text()
    assert "0.0625f" not in src, "LUT carries 1/8-scaled magnitudes"


def test_bit_trick_reproduces_the_lut():
    """The bit trick THE KERNEL SHIPS must reproduce the E2M1 table.

    The constants are read out of fp4_dequant.h: recomputing them in Python and
    comparing to a Python constant is a tautology that never opens the source.
    """
    src = code(_DEQUANT.read_text())
    m = re.search(r"simd<uint16_t, 8> res = (0x[0-9A-Fa-f]+) \+ \(\(m - (\d+)\) << (\d+)\)",
                  src)
    assert m, "the E2M1 bit trick is no longer recognisable in fp4_dequant.h"
    base, sub, shift = int(m.group(1), 16), int(m.group(2)), int(m.group(3))

    got = []
    for mm in range(8):
        if mm == 0:
            bits = 0x0000
        elif mm == 1:
            bits = 0x3800
        else:
            bits = base + ((mm - sub) << shift)
        got.append(_f16_from_bits(bits))
    assert got == E2M1, (
        f"the kernel's bit trick (0x{base:04X} + ((m - {sub}) << {shift})) "
        f"yields {got}, not the E2M1 table {E2M1}"
    )


def test_unpack_uses_the_e2m1_base():
    """0x2C00 encodes 0.0625 and yields the 1/8-scaled magnitudes."""
    # Case-insensitive both ways: hex literal case is not semantic.
    src = code(_LUT.read_text())
    assert re.search(r"0x3c00", src, re.I), "expected the 0x3C00 base for m >= 2"
    assert not re.search(r"0x2c00", src, re.I), (
        "0x2C00 encodes 0.0625, not 1.0: every magnitude comes out 8x small"
    )


def _sqrtsoftplus(x):
    import math

    return math.sqrt(math.log1p(math.exp(x)))


def route_reference(logits, bias, top_k, route_scale=1.5, eps=1e-20):
    """DeepSeek Gate.forward: transform ALL, select on +bias, gather unbiased."""
    scores = [_sqrtsoftplus(x) for x in logits]
    sel = [s + b for s, b in zip(scores, bias)]
    idx = sorted(range(len(sel)), key=lambda i: -sel[i])[:top_k]
    w = [scores[i] for i in idx]
    tot = sum(w) + eps
    return idx, [x / tot * route_scale for x in w]


def route_raw_selection(logits, bias, top_k, route_scale=1.5, eps=1e-5):
    """Selecting on the RAW logit + bias, transforming only the winner."""
    lg, bs = list(logits), list(bias)
    idx, w = [], []
    for _ in range(top_k):
        best = max(range(len(lg)), key=lambda i: lg[i] + bs[i])
        idx.append(best)
        w.append(_sqrtsoftplus(lg[best]))
        lg[best] = -65500.0
        bs[best] = -65500.0
    tot = sum(w) + eps
    return idx, [x / tot * route_scale for x in w]


def test_selection_diverges_when_bias_is_nonzero():
    """With bias != 0, raw-logit selection picks a different expert set."""
    import random

    random.seed(7)
    diff = 0
    trials = 400
    for _ in range(trials):
        logits = [random.gauss(0, 1.5) for _ in range(32)]
        bias = [random.gauss(0, 0.3) for _ in range(32)]
        a, _ = route_reference(logits, bias, 6)
        b, _ = route_raw_selection(logits, bias, 6)
        if set(a) != set(b):
            diff += 1
    assert diff > trials * 0.1, (
        f"only {diff}/{trials} differed; the test is not discriminating"
    )


def test_orders_agree_when_bias_is_zero():
    """sqrtsoftplus is monotonic, so with no bias both orders must match."""
    import random

    random.seed(11)
    for _ in range(50):
        logits = [random.gauss(0, 1.5) for _ in range(32)]
        zero = [0.0] * 32
        a, _ = route_reference(logits, zero, 6)
        b, _ = route_raw_selection(logits, zero, 6)
        assert set(a) == set(b)


def test_kernel_transforms_before_selecting():
    src = _TOPK.read_text()
    body = src[src.index("compute_noaux_tc_routing") :]
    tr = body.index("sqrt_softplus")
    # The property, not one spelling of the cast.
    m = re.search(r"scores\[i\]\s*\+\s*[^;]*bias\[i\]", body)
    assert m, "selection must be on scores + bias"
    assert tr < m.start(), (
        "sqrtsoftplus must be applied to all experts before selection"
    )
    assert "1e-20f" in body, "reference normalises by sum + 1e-20, not 1e-5"
    assert "float scores[NUM_EXPERTS]" in body, "accumulate in float, not fp16"


def test_expert_count_matches_the_config():
    if not _KERNELS.exists():
        pytest.skip("deepseek kernels not present")
    src = _KERNELS.read_text()
    assert "DeepSeekTopKKernel<384," in src, (
        "n_routed_experts is 384 in config.json; 256 matches no layer"
    )
    assert "case 6:" in src, "num_experts_per_tok is 6"
