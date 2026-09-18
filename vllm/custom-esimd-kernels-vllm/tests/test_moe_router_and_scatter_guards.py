"""Static guards over MoE properties that cannot be reproduced on CPU.

The output scatter in int4_nmajor_gemm.h is a gather-modify-scatter, not an
atomic: sorted_idxs is clamped to t1 - 1, so a partial final iteration carries
duplicate lanes on one token and all but one contribution is lost unless the
scatter is lane-masked. The narrow router kernel only writes four tokens, so it
must stay under an n_tokens <= 4 guard. The sglang patch's MoE branch must keep
its wrapper guard and name only attributes that exist.
"""

import re
from pathlib import Path

import pytest

from srctext import assert_single_write, code

_ROOT = Path(__file__).resolve().parents[3]
_NMAJOR = [
    Path(__file__).resolve().parents[1] / "csrc/moe_batch/int4_nmajor_gemm.h",
    _ROOT / "sglang/custom-esimd-kernels/csrc/moe_batch/int4_nmajor_gemm.h",
]
_MOE_SYCL = _ROOT / "sglang/custom-esimd-kernels/csrc/moe_batch/moe.sycl"
_SGL_PATCH = _ROOT / "sglang/patches/sglang_for_multi_arc.patch"


@pytest.mark.parametrize("path", _NMAJOR, ids=lambda p: p.parents[3].name)
def test_output_scatter_is_lane_masked(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    # code(), not raw text: a `/*ms_live*/` comment would satisfy the needle
    # over an unmasked scatter.
    src = code(path.read_text())
    m = re.search(r"scatter<IT, ?16>\(output,[^;]*;", src, re.S)
    assert m, "output scatter not found"
    assert "ms_live" in m.group(0), (
        "the read-modify-write scatter must be masked; duplicate clamped lanes "
        "otherwise lose contributions"
    )
    # ...and derived from the predicate, not a constant carrying the name.
    assert re.search(r"simd_mask<16> ms_live = lane_live\.", src), (
        "ms_live is no longer sliced out of lane_live, so every lane scatters "
        "and duplicate clamped lanes lose contributions"
    )
    # LAST WRITE: the slice keeps its exact text if `lane_live = 1;` is
    # inserted above it, so pin that lane_live is assigned exactly once.
    writes = re.findall(r"\blane_live\s*=(?!=)", src)
    assert len(writes) == 1, (
        f"lane_live is assigned {len(writes)} times; a later write makes every "
        "lane live and the masked scatter stops masking"
    )
    m = re.search(r"\blane_live\s*=\s*([^;]+);", src)
    assert m, "lane_live is no longer assigned"
    rhs = " ".join(m.group(1).split())
    # The predicate must bound against the ROW COUNT: `(lane_id + m_base) <
    # 0xFFFFFFFFu` compares and is vacuously true, so pin what it compares to.
    # Scoped per kernel, since each of the two variants legitimately declares
    # t1 and m_base. The write counts and the const bindings cover different
    # halves: const protects the binding, not what is bound.
    for m2 in re.finditer(r"simd_mask<MAX_M> lane_live", src):
        beg = src.rfind("SYCL_ESIMD_KERNEL", 0, m2.start())
        end = src.find("SYCL_ESIMD_KERNEL", m2.start())
        scope = src[max(0, beg):end if end > 0 else len(src)]
        for operand in ("t1", "m_base"):
            assert_single_write(scope, operand,
                                f"int4_nmajor_gemm.h: {operand}")
    assert re.search(r"const uint32_t m_row = \(uint32_t\)m_base;", src), (
        "the lane bound is read straight from the mutable m_base; bind a const "
        "copy so a rewrite around the mask is a compile error, not a silent "
        "all-true mask"
    )
    assert re.search(r"const uint32_t t_end = \(uint32_t\)t1;", src), (
        "the row count is read straight from the mutable t1"
    )
    assert rhs == "(lane_id + m_row) < t_end", (
        f"lane_live no longer bounds the lane index against the token count "
        f"t1; a comparison against a constant is vacuously true: {rhs!r}"
    )


@pytest.mark.parametrize("path", _NMAJOR, ids=lambda p: p.parents[3].name)
def test_no_atomic_claim_without_atomic(path):
    """A comment must not claim an atomic the scatter does not perform."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    assert "// Atomic add for accumulate across experts" not in src, (
        "comment claims an atomic that the code does not perform"
    )


def test_router_fallback_respects_the_narrow_wide_split():
    if not _MOE_SYCL.exists():
        pytest.skip("sglang tree not present")
    src = _MOE_SYCL.read_text()
    # Every call to the narrow kernel must sit under an n_tokens <= 4 test.
    for m in re.finditer(r"moe_router_forward_e4m3_kernel\(", src):
        window = src[max(0, m.start() - 600) : m.start()]
        assert "n_tokens <= 4" in window, (
            "narrow router kernel called without the n_tokens <= 4 guard; it "
            "only writes four tokens"
        )


def test_fused_router_dispatcher_is_not_stubbed():
    if not _MOE_SYCL.exists():
        pytest.skip("sglang tree not present")
    src = _MOE_SYCL.read_text()
    m = re.search(
        r"dispatch_moe_router_topk_fused_e4m3\([^)]*\)\s*\{\s*return false;\s*\}",
        src,
        re.S,
    )
    assert m is None, "dispatcher is stubbed to `return false`, forcing the fallback"
    assert "dispatch_moe_router_topk_fused_e4m3_orig" not in src, (
        "leftover _orig shadow of the real dispatcher"
    )


def test_patch_moe_branch_keeps_its_guard():
    if not _SGL_PATCH.exists():
        pytest.skip("sglang patch not present")
    src = _SGL_PATCH.read_text()
    assert "_maybe_esimd_moe_silu_fused" in src, "guarded MoE wrapper call removed"
    assert "quant_info.weight_scale" not in src, (
        "references a quant_info attribute that does not exist"
    )
    assert "dispatch_output.expert_idx" not in src, (
        "references a dispatch_output attribute that does not exist"
    )
