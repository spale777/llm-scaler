"""Static guard against the resadd+norm+GEMV cross-work-group residual race.

The fused kernels launch nd_range<1>(N, 1), so every work-group reads
residual_ptr with no ordering between them. If work-group 0 writes the updated
value back in place, a group scheduled after it reads h + r instead of r and
computes h + (h + r) -- nondeterministic and silent. Two shapes rule that out:
a single-work-item pre-pass on the same in-order queue, which settles
residual_ptr before the grid starts, or writing to a separate `nr` buffer that
must not alias `res`. The race needs a GPU to reproduce, so what is asserted
here is the structural property.
"""

import re
from pathlib import Path

import pytest

from srctext import code, tokens

_VLLM_DIR = Path(__file__).resolve().parents[1] / "csrc/xpu/esimd_kernels"
_SGL_DIR = (
    Path(__file__).resolve().parents[3]
    / "sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels"
)

# Files that launch a multi-work-group grid over a shared residual buffer.
_FUSED = [
    _VLLM_DIR / "resadd_norm_gemv_fused.h",
    _SGL_DIR / "resadd_norm_gemv_fused.h",
    _SGL_DIR / "resadd_norm_gemv_int4.h",
]


def _ids(p):
    return f"{p.parents[4].name}/{p.name}" if len(p.parents) > 4 else p.name


@pytest.mark.parametrize("path", _FUSED, ids=_ids)
def test_no_guarded_residual_store(path):
    """`if (n == 0) { block_store(residual_ptr...) }` is the race, verbatim."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = tokens(path.read_text())
    pat = re.compile(
        r"if\(n==0\)\{[^}]*block_store<[^>]*>\(residual_ptr", re.S
    )
    hit = pat.search(src)
    assert hit is None, (
        f"{path.name}: work-group 0 stores residual_ptr while other groups read "
        "it — restore the single-work-item pre-pass instead"
    )


@pytest.mark.parametrize("path", _FUSED, ids=_ids)
def test_prepass_is_present_and_launched(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    assert "ResAddResidualOnly" in src, (
        f"{path.name}: residual pre-pass kernel missing"
    )
    # It must actually be submitted, not merely defined.
    assert re.search(r"nd_range<1>\(1, 1\),\s*\n?\s*ResAddResidualOnly", src), (
        f"{path.name}: pre-pass is defined but never launched"
    )


@pytest.mark.parametrize("path", _FUSED, ids=_ids)
def test_prepass_handles_non_multiple_k(path):
    """K is not guaranteed to be a multiple of the 512-wide vector."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    start = src.index("struct ResAddResidualOnly")
    body = src[start : start + 1200]
    assert "for (; k < K; ++k)" in body, (
        f"{path.name}: pre-pass has no scalar remainder loop, so K % 512 "
        "elements would never be added"
    )


# Hosts with one work-group per output column: every block reads the whole
# residual row while only block 0 writes it, so `nr` aliasing `res` is the race.
_ALIAS_ENFORCED = [
    _SGL_DIR / "resadd_norm_gemv_kq.h",
    _SGL_DIR / "resadd_norm_gemv_kq_mt.h",
    _SGL_DIR / "resadd_norm_gemv_q4k_silu.h",
    _SGL_DIR / "resadd_norm_gemv_q4k_silu_mt.h",
    _SGL_DIR / "norm_add_norm_gemv_q4k_gelu.h",
]


@pytest.mark.parametrize("path", _ALIAS_ENFORCED, ids=lambda p: p.name)
def test_alias_contract_is_enforced_not_merely_declared(path):
    """The host must refuse nr == res. Asserted over `code()`, so the prose
    declaring the contract cannot stand in for enforcing it."""
    if not path.exists():
        pytest.skip("sglang tree not present")
    c = code(path.read_text())
    assert re.search(r"if \(nr == res\) return false;", c), (
        f"{path.name}: the alias contract is declared but not enforced; "
        "aliasing races block 0's residual store against every other block's "
        "load, giving a per-work-group rstd and silently wrong output"
    )


def test_gelu_fp8_binding_refuses_an_aliasing_residual():
    """norm_add_norm_gemv_gelu's host returns void, so the check is at the op."""
    path = (
        Path(__file__).resolve().parents[3]
        / "sglang/custom-esimd-kernels/csrc/xpu/esimd_kernel.sycl"
    )
    if not path.exists():
        pytest.skip("sglang tree not present")
    c = code(path.read_text())
    assert re.search(
        r"TORCH_CHECK\(residual_output\.data_ptr\(\) != residual_input\.data_ptr\(\),",
        c), (
        "esimd_norm_add_norm_gemv_gelu_fp8 must refuse an aliasing residual: "
        "1056 work-groups read the row, only WG 0 writes it, and loop 3 reads "
        "it again after the guarded store"
    )
