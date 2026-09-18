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

# The thread target and its fallback live in one shared header per tree; the
# dispatchers include it rather than each carrying a copy.
_OCCUPANCY_HDR = [
    _VLLM / "xpu/esimd_kernels/bmg_occupancy.h",
    _SGL / "xpu/esimd_kernels/bmg_occupancy.h",
]


def _occ_text(path):
    """The dispatcher's source plus the occupancy header it includes."""
    src = path.read_text()
    hdr = path.parent / "bmg_occupancy.h"
    if hdr.exists() and "bmg_occupancy.h" in src:
        src = hdr.read_text() + src
    return src


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
    c = code(_occ_text(path))
    assert re.search(r"BMG_HW_THREADS\s*=\s*\d+", c), (
        "occupancy target should be a named constant"
    )
    assert not re.search(r"[<>]=\s*6[34]\d\b", c), (
        "640 is a 20-core part in large-GRF mode, not B70"
    )
    m = re.search(r"BMG_HW_THREADS\s*=\s*(\d+)", c)
    if m:
        assert int(m.group(1)) == 2048, f"expected 2048, got {m.group(1)}"


@pytest.mark.parametrize("path", _OCCUPANCY, ids=_ids)
def test_thread_target_is_read_from_the_device(path):
    """B60 and B70 differ in Xe core count, so the target cannot be a constant.

    2048 is 32 Xe cores x 8 engines x 8 threads, which is B70. Applied to a B60
    the dispatcher believes it has more parallelism than it does and stops
    splitting K too early; the two disagree on the split for 672 of the first
    4096 N. max_compute_units reports the core count, so the device answers
    rather than a table, and the constant stays only as the fallback when the
    driver will not say.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(_occ_text(path))
    assert "max_compute_units" in c, (
        f"{path.name}: the thread target is not derived from the device, so a "
        "B60 is dispatched as though it were a B70"
    )
    assert "bmg_hw_threads(q)" in c, (
        f"{path.name}: the queried target is never passed to a dispatcher"
    )
    # No live comparison may still use the constant: that is the B70 assumption.
    live = re.findall(r"<=\s*BMG_HW_THREADS", c)
    assert not live, (
        f"{path.name}: {len(live)} dispatch comparison(s) still use the "
        "constant instead of the device's own thread count"
    )


@pytest.mark.parametrize("path", _DECODE, ids=_ids)
def test_weight_streams_are_prefetched(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    assert code(src).count("lsc_prefetch") >= 3, (
        f"{path.name}: expected prefetch on the gate, up and down weight streams"
    )


@pytest.mark.parametrize("path", _MOE_SYCL, ids=_ids)
def test_router_and_shared_weight_streams_are_prefetched(path):
    """Every fp8 weight stream in moe.sycl, not just the routed up/down pair.

    The router reads one [E, hidden] row per token block and the shared-expert
    kernels read the largest non-routed matrices at decode; both streamed with
    no prefetch while the routed kernels beside them had it.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    # Each fp8 weight pointer that is loaded in a k loop must be prefetched.
    checked = 0
    for ptr in ("wrow", "gw", "uw", "dw"):
        loads = len(re.findall(
            rf"block_load<uint8_t, 64>\({ptr} \+ k\)", c))
        if not loads:
            continue
        checked += 1
        pf = len(re.findall(rf"lsc_prefetch<[^>]*>\({ptr} \+ k \+ 64\)", c))
        assert pf >= loads, (
            f"{path.name}: {loads} load(s) of `{ptr}` in a k loop but only "
            f"{pf} prefetch(es); this stream is read once and never reused"
        )
    # Renaming every pointer in both trees would otherwise empty the loop and
    # pass having examined nothing.
    assert checked >= 4, (
        f"{path.name}: examined {checked} of the 4 weight streams; the "
        "anchors moved and this test stopped checking them"
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


_POOLED_SCRATCH = [
    (_VLLM / "moe_batch/moe_int4.sycl",
     "moe_forward_gelu_tanh_int4_decode", "gemma_int4_decode_ws"),
    (_SGL / "xpu/esimd_kernel.sycl",
     "esimd_shared_expert_q8", "shared_expert_q8_ws"),
]


@pytest.mark.parametrize("path,fn,pool", _POOLED_SCRATCH,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_pooled_scratch_does_not_escape(path, fn, pool):
    """A decode op may reuse its internal buffers but must return a fresh one.

    Both of these run per token per layer and allocated every intermediate on
    entry, which is pure allocator dispatch on a host-bound path; the file each
    lives in already carried the pooling idiom. Reuse is only sound while the
    returned tensor is not itself pooled -- it escapes to python and has to
    stay live past the call, and the next token would otherwise overwrite it.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    i = src.find(fn + "(")
    assert i >= 0, f"{fn} not found -- re-derive this test"
    # Brace-match the body.
    ob = src.index("{", i)
    depth, j = 0, ob
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    body = src[i:j + 1]
    assert pool in body, f"{fn} no longer uses the {pool} pool"
    returned = set(re.findall(r"return\s+(\w+)\s*;", body))
    assert returned, f"{fn} returns nothing -- re-derive this test"
    for name in returned:
        assert not re.search(rf"auto&\s+{name}\s*=\s*ws\.", body), (
            f"{fn}: `{name}` is returned but bound to the pool; the next "
            "call overwrites a tensor the caller still holds"
        )


_NMAJOR = [
    _VLLM / "moe_batch/int4_nmajor_gemm.h",
    _SGL / "moe_batch/int4_nmajor_gemm.h",
]


@pytest.mark.parametrize("path", _NMAJOR, ids=_ids)
def test_nmajor_scale_reads_are_not_scalar_strided(path):
    """N-major scales for one k-group are a strided run, so they load as one.

    `s_base[(n_start + ni) * K_groups + kg]` walked by a scalar `ni` loop is one
    message per element, and consecutive ni are K_groups*2 bytes apart -- a
    separate 64-byte line each, for two useful bytes. The up kernel does this
    16 times per group and the down kernel 32, on both scale streams.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = code(path.read_text())
    scalar = re.findall(r"\w+\[ni\]\s*=\s*s_base\[\(", src)
    assert not scalar, (
        f"{path.name}: {len(scalar)} scalar strided scale read(s) remain; "
        "a strided gather<fp16, N> issues the run as one message"
    )
    gathers = re.findall(r"gather<fp16,\s*N>\(\w+\s*\+\s*kg,\s*\w+\)", src)
    assert len(gathers) >= 3, (
        f"{path.name}: expected the gate, up and down scale streams to gather; "
        f"found {len(gathers)}"
    )


_PREFILL_DPAS = _SGL / "xpu/esimd_kernels/prefill_dpas.h"


def test_prefill_dpas_checks_the_device_slm_budget():
    """104 KB per work-group is a request, not a guarantee.

    The static_assert covers the 128 KB architectural per-core budget, but the
    amount a driver exposes to a single work-group is a device property and is
    not the same number on every Battlemage part. Without the runtime check the
    kernel fails to launch and says nothing about which resource was short.
    """
    if not _PREFILL_DPAS.exists():
        pytest.skip(f"{_PREFILL_DPAS} not present")
    src = code(_PREFILL_DPAS.read_text())
    i = src.find("sdp_paged_prefill_dpas_host")
    assert i >= 0, "host launcher not found -- re-derive this test"
    body = src[i:]
    assert "local_mem_size" in body, (
        "the host launcher must query the device SLM size before launching"
    )
    assert "PF_TOTAL_SLM" in body, (
        "the check must compare against the kernel's own SLM request"
    )


_BLOCKSCALE = [
    _VLLM / "xpu/esimd_kernels/fp8_moe_gemm_blockscale.h",
    _SGL / "xpu/esimd_kernels/fp8_moe_gemm_blockscale.h",
]


@pytest.mark.parametrize("path", _BLOCKSCALE, ids=_ids)
def test_dpas_prefill_requests_large_grf_per_kernel(path):
    """The register file is picked per kernel, not per translation unit.

    moe_gemm_block_prefill_kernel holds acc[8] of simd<float,128> -- 4 KB
    before its operands -- so it needs the 256-register file. The kernels it
    shares a module with (topk, scatter, silu, gather) hold under 300 B, and a
    module-wide -doubleGRF halves their threads per vector engine from 8 to 4
    to buy them nothing. grf_size<256> on the one kernel that needs it keeps
    both correct.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    i = src.find("struct moe_gemm_block_prefill_kernel")
    assert i >= 0, "prefill kernel not found -- re-derive this test"
    j = src.find("\nstruct ", i + 10)
    body = src[i:] if j < 0 else src[i:j]
    assert "grf_size<256>" in body, (
        "the DPAS prefill kernel must request the large register file itself; "
        "otherwise it depends on a module-wide flag that penalises its "
        "translation-unit siblings"
    )


def test_mixed_moe_module_does_not_force_large_grf_on_every_kernel():
    """-doubleGRF is module-wide; the MoE module is not single-purpose.

    esimd_kernel_moe.sycl pulls in both moe_ops.h (light, row-parallel) and
    fp8_moe_gemm_blockscale.h (heavy DPAS). Building the whole module large-GRF
    halves occupancy for the light majority. The genuinely single-purpose
    modules (GDN conv, grouped GGUF, prefill DPAS) keep the flag.
    """
    setups = [
        Path(__file__).resolve().parents[1] / "setup.py",
        Path(__file__).resolve().parents[1] / "setup_sycl.py",
        _ROOT / "sglang/custom-esimd-kernels/setup.py",
    ]
    checked = 0
    for setup in setups:
        if not setup.exists():
            continue
        src = setup.read_text()
        i = src.find("esimd_kernel_moe.sycl")
        if i < 0:
            continue
        checked += 1
        # The compile args for the module that owns this source.
        seg = src[i:i + 1200]
        end = seg.find("ext_modules.append")
        if end > 0:
            seg = seg[:end]
        assert "doubleGRF" not in seg, (
            f"{setup.name}: the module containing esimd_kernel_moe.sycl is "
            "built -doubleGRF, which halves occupancy for its light kernels; "
            "the DPAS kernel requests grf_size<256> for itself instead"
        )
    assert checked >= 2, (
        f"only {checked} setup file(s) declared the MoE module; this test "
        "would pass without examining the flag it exists to pin"
    )


_MOE_BLOCKSCALE = [
    _VLLM / "xpu/esimd_kernels/fp8_moe_gemm_blockscale.h",
    _SGL / "xpu/esimd_kernels/fp8_moe_gemm_blockscale.h",
]


@pytest.mark.parametrize("path", _MOE_BLOCKSCALE, ids=_ids)
def test_moe_blockscale_uses_the_requested_k_block(path):
    """block_k reaches the kernel instead of being accepted and ignored.

    The decode launcher pinned `constexpr int BK = 128` while its host took
    block_k as a parameter, so a 32- or 64-wide K block was silently scaled as
    though it were 128: every weight past the first block gets the wrong scale
    and the output is plausible but wrong. DeepSeek V4.1 uses 32.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    i = c.find("inline void launch_moe_gemv_block")
    assert i >= 0, "the decode launcher was renamed; re-derive this test"
    body = c[i:c.find("inline void moe_gemm_fp8_blockscale_host", i)]
    assert not re.search(r"constexpr\s+int\s+BK\s*=\s*\d+", body), (
        "the decode launcher pins BK; block_k would be ignored"
    )
    assert "int BK," in c[c.find("template", i - 200):i + 80] or \
           "int VL, int BK, int MAX_M" in c, (
        "BK must be a template parameter of the decode launcher"
    )
    # The host must actually branch on the runtime value.
    host = c[c.find("inline void moe_gemm_fp8_blockscale_host"):]
    assert "block_k == 32" in host and "block_k == 64" in host, (
        "the host does not dispatch the narrower K blocks, so requesting one "
        "silently falls back to 128"
    )
