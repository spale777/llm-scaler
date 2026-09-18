"""Static guards for the performance-critical kernel shapes.

None of these can be measured without a GPU, so each test pins the structural
property the performance depends on, not its effect.
"""

import re
from pathlib import Path

import pytest

from srctext import code, tokens

_ROOT = Path(__file__).resolve().parents[3]
_VLLM = Path(__file__).resolve().parents[1] / "csrc"
_SGL = _ROOT / "sglang/custom-esimd-kernels/csrc"

_MOE_OPS = [_VLLM / "xpu/esimd_kernels/moe_ops.h", _SGL / "xpu/esimd_kernels/moe_ops.h"]
_MOE_SYCL = [_VLLM / "moe_batch/moe.sycl", _SGL / "moe_batch/moe.sycl"]
_DECODE = [_VLLM / "moe_batch/moe_decode_gemv.h", _SGL / "moe_batch/moe_decode_gemv.h"]
_OCCUPANCY = [
    _VLLM / "xpu/esimd_kernels/fp8_GEMV_bmg.h",
    _VLLM / "xpu/esimd_kernels/fp8_GEMM_blockscale.h",
    _SGL / "xpu/esimd_kernels/fp8_GEMV_bmg.h",
    _SGL / "xpu/esimd_kernels/fp8_GEMM_blockscale.h",
]


def _ids(p):
    return f"{p.parents[3].name}/{p.name}"


@pytest.mark.parametrize("path", _MOE_OPS, ids=_ids)
def test_row_parallel_kernels_use_a_wide_work_group(path):
    """A work-group of one work-item uses one lane of a SIMD16 engine."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    c = code(src)
    assert re.search(r"MOE_WG\s*=\s*\d+", c), "MOE_WG must be a real constant"
    assert "{MOE_WG}" in c, "the launches must use MOE_WG as the local range"
    # The sequential prefix scan is the one legitimate single-item launch.
    singles = re.findall(r"nd_range<1>\(\{[^}]*\}, \{1\}\)", src)
    assert len(singles) <= 1, (
        f"{path.name}: {len(singles)} single-work-item launches; only the "
        "sequential prefix scan should remain"
    )


@pytest.mark.parametrize("path", _MOE_OPS, ids=_ids)
def test_widened_kernels_index_globally_and_bound_check(path):
    """A wider group makes get_group(0) the wrong index."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    src = code(src)
    assert "item.get_group(0)" not in src, (
        f"{path.name}: get_group(0) indexes work-groups, not rows"
    )
    n_idx = len(re.findall(r"item\.get_global_id\(0\)", src))
    n_guard = len(re.findall(r"if \(\w+ >= \w+\) return;", src))
    assert n_guard >= n_idx, (
        f"{path.name}: {n_idx} global-id reads but only {n_guard} bounds guards; "
        "a rounded-up global range overruns without one"
    )


@pytest.mark.parametrize("path", _OCCUPANCY, ids=_ids)
def test_occupancy_target_is_named_and_correct(path):
    """B70 is 32 Xe cores x 8 engines x 8 threads."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    c = code(src)
    assert re.search(r"BMG_HW_THREADS\s*=\s*\d+", c), (
        "occupancy target should be a named constant"
    )
    assert not re.search(r"[<>]=\s*6[34]\d\b", c), (
        "640 is a 20-core part in large-GRF mode, not B70"
    )
    m = re.search(r"BMG_HW_THREADS\s*=\s*(\d+)", c)
    if m:
        assert int(m.group(1)) == 2048, f"expected 2048, got {m.group(1)}"


@pytest.mark.parametrize("path", _DECODE, ids=_ids)
def test_weight_streams_are_prefetched(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    assert code(src).count("lsc_prefetch") >= 3, (
        f"{path.name}: expected prefetch on the gate, up and down weight streams"
    )


@pytest.mark.parametrize("path", _MOE_SYCL, ids=_ids)
def test_wide_router_streams_each_weight_row_once(path):
    """A (token, expert) grid re-reads each expert row n_tokens times."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    seen = 0
    for name in ("MoeRouterForwardE4M3Wide", "MoeRouterForwardE5M2Wide"):
        i = src.find(name)
        if i < 0:
            continue
        seen += 1
        window = src[i : i + 400]
        assert "sycl::range<2>" not in window, (
            f"{path.name}: {name} still uses a (token, expert) grid"
        )
    assert re.search(r"WIDE_TOK\s*=\s*\d+", code(src)), (
        "token-blocking constant missing"
    )
    assert seen, (
        "neither wide-router kernel was found; the anchor moved and this test stopped checking the grid")


@pytest.mark.parametrize("path", _MOE_SYCL, ids=_ids)
def test_submit_kernel_does_not_type_erase(path):
    """std::function heap-allocates on every launch."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    m = re.search(r"submit_kernel\(\s*\n?\s*([^,]+),", src)
    assert m, "submit_kernel not found"
    assert "std::function" not in m.group(1), (
        f"{path.name}: submit_kernel takes std::function by value"
    )
    assert "template <typename KernelFn>" in src


@pytest.mark.parametrize("path", _MOE_SYCL, ids=_ids)
def test_wide_router_accumulates_in_float(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    # The router declares xv* as simd<fp16,...>, so xv * wv rounds before
    # reaching the accumulator; elsewhere the operands are already float, hence
    # matching the router's own variable names. Matching the accumulate
    # SPELLING is a losing game, so assert the positive property: every xv*wv
    # product sits inside a convert<float>(...), which travels with it.
    t = tokens(src)
    prods = [m.start() for m in re.finditer(r"xv\w*\*wv\w*", t)]
    assert prods, "no xv*wv products found -- re-derive this test"
    unconverted = []
    for at in prods:
        # Walk back over the enclosing call, if any.
        head = t[:at]
        k, depth = len(head) - 1, 0
        while k >= 0:
            if head[k] == ")":
                depth += 1
            elif head[k] == "(":
                if depth == 0:
                    break
                depth -= 1
            k -= 1
        if k < 0 or not head[:k].endswith("convert<float>"):
            unconverted.append(t[max(0, at - 30):at + 12])
    # Wrapping alone is not sufficient: rounding can be reinstated in a
    # sibling statement under either spelling, so the scope is the enclosing
    # BLOCK and both `fp16` and `half` are banned. At H=7168 that rounding
    # swaps one of the top-8 experts.
    for at in prods:
        beg = t.rfind("{", 0, at) + 1
        end = t.find("}", at)
        stmt = t[beg:end if end > 0 else len(t)]
        # fp16 OPERANDS are legitimate; only a conversion of the ACCUMULATOR.
        assert not re.search(r"convert<(?:fp16|half)>\(\s*acc", stmt), (
            f"{path.name}: the accumulate rounds back to fp16, reinstating the "
            f"divergence this test is named for: {stmt[:110]}"
        )
    assert not unconverted, (
        f"{path.name}: an xv*wv product is not wrapped in convert<float>, so it "
        f"rounds to fp16 before reaching the accumulator and the wide router "
        f"disagrees with the narrow kernel across the dispatch boundary: "
        f"{unconverted[:2]}"
    )
