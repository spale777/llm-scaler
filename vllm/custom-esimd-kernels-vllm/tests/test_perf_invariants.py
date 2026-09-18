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


# Every file that DEFINES submit_kernel, not just the two moe.sycl ones: the
# definition was templated there while moe_int4.sycl and eagle.sycl kept the
# by-value std::function in both trees, so the guard passed over 105 launch
# sites that still allocated.
_SUBMIT_KERNEL_FILES = _MOE_SYCL + [
    _VLLM / "moe_batch/moe_int4.sycl", _SGL / "moe_batch/moe_int4.sycl",
    _VLLM / "eagle/eagle.sycl", _SGL / "eagle/eagle.sycl",
]


@pytest.mark.parametrize("path", _SUBMIT_KERNEL_FILES, ids=_ids)
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


@pytest.mark.parametrize("path", _SUBMIT_KERNEL_FILES, ids=_ids)
def test_no_std_function_scaffolding_around_a_single_submit(path):
    """A `std::function` assigned and submitted inside the same switch arm is
    a second heap allocation buying nothing.

    Only the arm-local form is banned. Where one handle is assigned across
    several branches and submitted once afterwards the type erasure is doing
    real work (carrying a branch-selected kernel out of the switch), and
    removing it means duplicating the submit into every arm.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    # Only std::function-typed handles. `auto cgf = [&](sycl::handler&){...}`
    # binds the closure by its own type and erases nothing, so flagging it is a
    # false positive -- and both moe.sycl files are written that way.
    erased = set(re.findall(
        r"std::function<void\(sycl::handler&\)>\s+(\w+)", src))
    if not erased:
        return
    # A lambda body contains semicolons, so the span from the assignment to the
    # submit cannot be matched with a character class -- brace-walk instead.
    bad = []
    for m in re.finditer(r"\b(\w+)\s*=\s*\[[&=]\]\s*\(\s*sycl::handler", src):
        name = m.group(1)
        if name not in erased:
            continue
        ob = src.find("{", m.end())
        if ob < 0:
            continue
        depth, j = 0, ob
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        # What follows the lambda, up to the next statement of substance.
        tail = src[j:j + 200]
        if re.search(rf"submit_kernel\(\s*{re.escape(name)}\s*,", tail):
            bad.append(name)
    assert not bad, (
        f"{path.name}: {sorted(set(bad))} is assigned and submitted within one "
        "arm; pass the lambda straight to submit_kernel instead"
    )


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


_INT4_MOE = [_VLLM / "moe_batch/moe_int4.sycl", _SGL / "moe_batch/moe_int4.sycl"]


@pytest.mark.parametrize("path", _INT4_MOE, ids=_ids)
def test_int4_scale_is_not_reloaded_every_k_step(path):
    """A scale group spans 16 kp, so loading it per kp re-reads it 16 times.

    The weight load in these loops is the useful traffic; on the 64-wide up
    kernel four scale vectors of 64 bytes each sat beside it, so a third of
    that kernel's bytes were the same scale lines fetched again. The value is
    a pure function of kp / 16, so reloading only when that changes is
    bit-identical -- simulated over every shape and stride with no divergence.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = code(path.read_text())
    # Every `kg = kp / 16` must be followed by a change guard rather than by a
    # bare load.
    sites = [m.end() for m in re.finditer(r"(?:const )?int kg = kp / 16;", src)]
    assert sites, "the scale-group index is gone -- re-derive this test"
    ungated = []
    for at in sites:
        window = src[at:at + 120]
        if not re.search(r"if \(kg != kg_cur\)", window):
            ungated.append(window[:70])
    assert not ungated, (
        f"{path.name}: {len(ungated)} scale-group site(s) load unconditionally "
        f"inside the k loop: {ungated}"
    )
    # ...and the cursor must start outside the valid range, or the first group
    # is never loaded at all.
    for m in re.finditer(r"int kg_cur = (-?\d+);", src):
        assert int(m.group(1)) < 0, (
            f"{path.name}: kg_cur starts at {m.group(1)}, so group "
            f"{m.group(1)} would be skipped"
        )
