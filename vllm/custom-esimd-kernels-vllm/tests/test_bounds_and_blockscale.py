"""Bounds and block-scale guards over the prefill and GEMM kernels.

Four limits: the KV surface height must be the real extent, or the hardware Y
clamp is defeated; gather() byte offsets are uint32, which
token_index * hidden_size * 2 exhausts at 262144 tokens and hidden 8192;
prefill SLM must fit 128 KB; the N-block divisor is a parameter, not 128.
"""

import re
from pathlib import Path

import pytest

from srctext import assert_single_write, code, tokens

_ROOT = Path(__file__).resolve().parents[3]
_VLLM = Path(__file__).resolve().parents[1] / "csrc"
_SGL = _ROOT / "sglang/custom-esimd-kernels/csrc"

_PREFILL = _SGL / "xpu/esimd_kernels/prefill_dpas.h"
_NMAJOR = [_VLLM / "moe_batch/int4_nmajor_gemm.h", _SGL / "moe_batch/int4_nmajor_gemm.h"]
_BLOCKSCALE = [
    _VLLM / "xpu/esimd_kernels/fp8_GEMM_blockscale.h",
    _SGL / "xpu/esimd_kernels/fp8_GEMM_blockscale.h",
]
# The MoE kernels are the ones the built module ships; they take block_n at
# runtime rather than as a template parameter.
_MOE_BLOCKSCALE = [
    _VLLM / "xpu/esimd_kernels/fp8_moe_gemm_blockscale.h",
    _SGL / "xpu/esimd_kernels/fp8_moe_gemm_blockscale.h",
]


def _ids(p):
    return f"{p.parents[3].name}/{p.name}"


def test_kv_surface_height_is_the_real_extent():
    if not _PREFILL.exists():
        pytest.skip("prefill_dpas.h not present")
    src = code(_PREFILL.read_text())
    assert not re.search(r"0x3[Ff]{5}[Uu]?", src), (
        "a fabricated surface height defeats the hardware bounds clamp on the "
        "KV cache"
    )
    assert "num_kv_blocks" in src, "the physical KV extent must reach the kernel"
    m = re.search(r"kv_surf_h\s*=\s*([^;]+);", code(src))
    assert m, "kv_surf_h not found"
    expr = m.group(1)
    assert all(k in expr for k in ("num_kv_blocks", "phys_rows_per_block",
                                   "block_size")), (
        "the height must be the largest Y KV_PHYS_Y can emit: "
        "(num_kv_blocks - 1) * phys_rows_per_block + block_size - 1, "
        f"got: {expr}"
    )
    # A literal dodges the clamp in either base, and a ternary can name the
    # variables while ignoring their values.
    assert not re.search(r"0x[0-9A-Fa-f]{5,}", expr), (
        f"a hex literal surface height defeats the bounds clamp: {expr}"
    )
    assert not re.search(r"\b\d{6,}\b", expr), (
        f"a decimal literal surface height defeats the bounds clamp: {expr}"
    )
    assert "?" not in expr, (
        f"the height must be a plain expression, not a conditional: {expr}"
    )


def test_block_table_clamp_is_still_present():
    """A separate mechanism from the surface height: this bounds the block
    table read, and without it the device faults."""
    if not _PREFILL.exists():
        pytest.skip("prefill_dpas.h not present")
    src = code(_PREFILL.read_text())
    # A bare identifier survives a body reduced to ((int32_t)(idx)), so pin
    # the comparison, which cannot.
    assert "#define BLK_LOGICAL_CLAMP" in src, "the clamp macro is gone"
    body = src[src.index("#define BLK_LOGICAL_CLAMP"):][:220]
    assert "max_valid_blk_idx" in body, (
        "BLK_LOGICAL_CLAMP no longer mentions max_valid_blk_idx -- the macro "
        "has been reduced to an identity and clamps nothing"
    )
    assert "< max_valid_blk_idx ?" in body and ": max_valid_blk_idx" in body, (
        f"BLK_LOGICAL_CLAMP is not a clamp against max_valid_blk_idx: {body[:120]}"
    )
    # The WHOLE right-hand side: a prefix match leaves `| (1 << 30)` free to
    # be appended within the same single assignment.
    m = re.search(r"max_valid_blk_idx\s*=\s*([^;]+);", src)
    assert m, "max_valid_blk_idx is no longer assigned"
    rhs = " ".join(m.group(1).split())
    assert rhs == "(seq_len - 1) >> block_size_shift", (
        f"the block bound is no longer exactly (seq_len - 1) >> "
        f"block_size_shift; any extra term can widen it back to inert: {rhs!r}"
    )
    # ...and exactly one write of it, since a later one replaces the bound.
    writes = re.findall(r"max_valid_blk_idx\s*=(?!=)", src)
    assert len(writes) == 1, (
        f"max_valid_blk_idx is assigned {len(writes)} times; a later write "
        "replaces the bound the clamp reads"
    )
    # ...and of what the RHS reads, which can be widened around the pinned
    # line and restored after, leaving every assertion above satisfied.
    for operand in ("seq_len", "block_size_shift"):
        assert_single_write(src, operand, f"prefill_dpas.h: {operand}")


def test_slm_request_is_bounded_at_compile_time():
    if not _PREFILL.exists():
        pytest.skip("prefill_dpas.h not present")
    src = _PREFILL.read_text()
    assert "static_assert(PF_TOTAL_SLM" in src, (
        "the SLM request should fail the build, not the launch"
    )
    m = re.search(r"PF_TOTAL_SLM\s*=\s*(0x[0-9A-Fa-f]+)", src)
    assert m, "PF_TOTAL_SLM not found"
    assert int(m.group(1), 16) <= 128 * 1024


@pytest.mark.parametrize("path", _NMAJOR, ids=_ids)
def test_gather_offset_range_is_guarded(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    n = len(re.findall(r"TORCH_CHECK\(\s*\(\(int64_t\)total_seqlen", c))
    assert n >= 2, (
        f"both host launchers must reject wrapping token offsets; found {n} "
        "TORCH_CHECK guards"
    )
    assert "0xFFFFFFFFLL" in c, "the guard must bound the uint32 range"


def test_offset_bound_math():
    """The guard threshold must be where uint32 actually wraps."""
    limit = 0xFFFFFFFF
    # 262144 x 8192 x 2 is exactly 2**32: the first shape that wraps.
    assert 262144 * 8192 * 2 > limit, "this shape must be rejected"
    assert 262143 * 8192 * 2 <= limit, "one token fewer must still be accepted"
    assert 131072 * 8192 * 2 <= limit, "a typical prefill must be accepted"


@pytest.mark.parametrize("path", _BLOCKSCALE, ids=_ids)
def test_n_block_is_parameterised(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    t = tokens(path.read_text())
    assert "(n/128)" not in t and "(n/BK)" not in t, (
        "the N index must be divided by an N-block, not a literal or the K-block"
    )
    assert "intBN" in t, "BN should be a template parameter"
    assert "(n/BN)" in t


@pytest.mark.parametrize("path", _BLOCKSCALE, ids=_ids)
def test_block_n_32_is_reachable(path):
    """DeepSeek V4.1 uses 32x32 weight blocks."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    assert "block_n == 128 || block_n == 32" in code(src), (
        "the host still rejects every block_n except 128"
    )
    assert "dispatch_gemv_block_bmg<32," in src, "no BN=32 instantiation"


@pytest.mark.parametrize("path", _MOE_BLOCKSCALE, ids=_ids)
def test_moe_n_block_is_indexed_by_block_n(path):
    """BK is the K-block: using it for the N index conflates the two axes."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    t = tokens(path.read_text())
    assert "(n/BK)" not in t and "(n/128)" not in t, (
        "the N index must be divided by block_n, not the K-block or a literal"
    )
    assert "n_start/128" not in t, "the prefill N index is still a literal"
    assert "(n/block_n)" in t and "n_start/block_n" in t
