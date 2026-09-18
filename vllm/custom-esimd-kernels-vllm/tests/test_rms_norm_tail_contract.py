"""Contract test for the fused add + RMSNorm tail handling, modelled in Python.

The two trees ship different kernels and so carry different obligations. The
vLLM tail block re-reads a full VL window ending at K, overlapping the last
aligned chunk, so its store must be lane-masked or the overlap gets h + 2r --
and residual is the inter-layer carry, so that compounds through every later
layer. It also needs K >= VL, since k_tail = K - VL underflows below the tensor
base. The sglang kernel has no tail path at all, walking exactly K / VL chunks,
so its host must reject an indivisible K rather than truncate it.
"""

from pathlib import Path

import pytest

from srctext import code, tokens

_VLLM = Path(__file__).resolve().parents[1] / "csrc/xpu/esimd_kernels/fused_add_rms_norm.h"
_SGL = (
    Path(__file__).resolve().parents[3]
    / "sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels/fused_add_rms_norm.h"
)


def pass1_masked(hidden, residual, vl):
    """Model of the vLLM pass 1: aligned loop + masked-store tail."""
    k = len(hidden)
    k_aligned = (k // vl) * vl
    out = list(residual)
    for base in range(0, k_aligned, vl):
        for i in range(vl):
            out[base + i] = hidden[base + i] + residual[base + i]
    if k_aligned < k and k >= vl:
        k_tail = k - vl
        overlap = k_aligned - k_tail
        for i in range(vl):
            if i < overlap:
                continue  # already written by the aligned loop
            idx = k_tail + i
            out[idx] = hidden[idx] + residual[idx]
    return out


def pass1_unmasked(hidden, residual, vl):
    """Counter-model: an unmasked tail that stores the whole window."""
    k = len(hidden)
    k_aligned = (k // vl) * vl
    out = list(residual)
    for base in range(0, k_aligned, vl):
        for i in range(vl):
            out[base + i] = hidden[base + i] + residual[base + i]
    if k_aligned < k:
        k_tail = k - vl
        for i in range(vl):
            idx = k_tail + i
            out[idx] = hidden[idx] + out[idx]  # re-reads the ALREADY-updated value
    return out


@pytest.mark.parametrize("k", [2816, 1044, 1056, 320, 192, 100, 65])
@pytest.mark.parametrize("vl", [64, 128, 256])
def test_residual_written_exactly_once(k, vl):
    if k < vl:
        pytest.skip("host rejects K < VL")
    hidden = [1.0] * k
    residual = [10.0] * k
    got = pass1_masked(hidden, residual, vl)
    assert got == [11.0] * k, "every element must be h + r exactly once"


@pytest.mark.parametrize("k,vl", [(1044, 64), (2816, 256), (100, 64), (1044, 128)])
def test_unmasked_model_corrupts_overlap(k, vl):
    """The unmasked model must differ, or the test above proves nothing: its
    tail re-adds hidden over lanes already holding h + r, reaching 2h + r."""
    if k % vl == 0:
        pytest.skip("no tail for this shape")
    hidden, residual = [1.0] * k, [10.0] * k
    good = pass1_masked(hidden, residual, vl)
    bad = pass1_unmasked(hidden, residual, vl)
    assert good != bad, "test is not exercising the bug"
    assert any(v == 12.0 for v in bad), "expected 2h + r in the overlap"


def test_vllm_tail_store_is_masked():
    c = code(_VLLM.read_text())
    assert c.count("k_aligned < K && K >= VL") == 6, (
        "every tail block must guard K >= VL; k_tail = K - VL underflows below it"
    )
    # Polarity, not presence: `lane < overlap` stores exactly the lanes the
    # main loop already wrote, which is the 2h + r corruption.
    t = tokens(_VLLM.read_text())
    assert "simd_mask<VL>keep=lane>=(uint32_t)overlap" in t, (
        "the pass-1 tail must keep lanes at or past the overlap"
    )
    assert "simd_mask<VL>keep=lane<(uint32_t)overlap" not in t, (
        "inverted mask: this stores the overlap instead of excluding it"
    )


def test_sglang_dispatcher_rejects_indivisible_k():
    if not _SGL.exists():
        pytest.skip("sglang tree not present")
    src = code(_SGL.read_text())
    start = src.index("inline void fused_add_rms_norm_host")
    body = src[start : start + 1600]
    assert "TORCH_CHECK(false" in body, (
        "the kernel has no tail path, so an indivisible K must be rejected, "
        "not silently truncated"
    )
    assert "K % 8 == 0" in body, "expected the VL ladder down to 8"
