"""A Python admission gate must imply the TORCH_CHECKs of the kernel it calls.

Each test derives what a kernel demands from its own TORCH_CHECKs and checks the
gate against that, in both directions where the comparison is decidable from
source text. The too-strict direction matters as much as the too-loose one: a
gate declining work the kernel accepts costs performance and fails nothing.

Scope is narrow by construction. Source text does not support a symbolic
implication check over arbitrary Python conditions, so each test below encodes
one decidable shape of mismatch.
"""

import re
from pathlib import Path

import pytest

from srctext import (assert_single_write, code,
                     is_vacuous_predicate, patch_code)

_ROOT = Path(__file__).resolve().parents[3]
_VLLM = Path(__file__).resolve().parents[1] / "csrc"
_SGL = _ROOT / "sglang/custom-esimd-kernels/csrc"
_SGL_PATCH = _ROOT / "sglang/patches/sglang_for_multi_arc.patch"
_VLLM_PATCH = _ROOT / "vllm/patches/vllm_for_multi_arc.patch"


def _patch(p):
    if not p.exists():
        pytest.skip(f"{p.name} not present")
    return p.read_text()


def _balanced_rhs(text):
    """The right-hand side up to its own balanced close, joined to one line."""
    out, depth = [], 0
    for line in text.splitlines():
        line = line.lstrip("+").strip()
        if out and depth == 0 and ("=" in line and not line.startswith(("=", "=="))):
            break                      # the next statement
        out.append(line)
        depth += line.count("(") - line.count(")")
        depth += line.count("[") - line.count("]")
        if out and depth <= 0 and not line.endswith(("and", "or", "(", "[", ",")):
            break
    return " ".join(out).strip()

def _kernel_pins_exact_dtype(src, op, tensors):
    """True when the kernel pins every named tensor to one exact scalar type."""
    i = src.find("esimd_" + op) if not op.startswith("esimd_") else src.find(op)
    if i < 0:
        return None
    body = src[i:i + 4000]
    return all(
        re.search(rf"{t}\.scalar_type\(\) ==\s*at::(?:ScalarType::)?\w+", body)
        for t in tensors
    )


# `itemsize == 2` is true of float16 AND bfloat16, so where a kernel pins Half
# the proxy admits bf16 into a TORCH_CHECK abort.

def test_no_gate_proxies_a_dtype_by_its_width():
    """A gate must not use itemsize as a stand-in for an exact dtype."""
    # Strip comments first: an `itemsize ==` match in prose is not a gate. The
    # excusing clause must pin THIS object's dtype, or a neighbouring tensor's
    # dtype excuses a genuine proxy on a different one.
    offenders = []
    for name, text in (("sglang", patch_code(_patch(_SGL_PATCH))),
                       ("vllm", patch_code(_patch(_VLLM_PATCH)))):
        for m in re.finditer(
                r"^\+.*?([A-Za-z_][\w.]*?)(?:\.dtype)?\.itemsize\s*==\s*(\d+)",
                text, re.M):
            line = m.group(0).lstrip("+").strip()
            base = m.group(1).split(".")[0]
            window = text[max(0, m.start() - 1500):m.start() + 1500]
            if "_esimd_" not in window:
                continue
            if not re.search(
                    rf"^\+.*\b{re.escape(base)}[\w.]*\.dtype\s*==\s*torch\.\w+",
                    window, re.M):
                offenders.append(f"{name}: {line} (nothing pins {base}'s dtype)")
    assert not offenders, (
        "a width proxy is the only dtype condition guarding an ESIMD kernel; "
        "itemsize == 2 admits bfloat16 wherever the kernel pins Half: "
        f"{offenders}"
    )


# A gate whose comment quotes a kernel message the kernel no longer emits is
# stale in the too-strict direction: it declines work the kernel now accepts.

def test_gate_comments_do_not_quote_superseded_kernel_messages():
    """A gate justified by a message the kernel no longer emits is stale.

    Collect the quoted kernel-error fragments that gate comments rely on, and
    require each to still exist in the kernel sources.
    """
    text = _patch(_SGL_PATCH)
    sources = ""
    for tree in (_SGL, _VLLM):
        if not tree.exists():
            continue
        for ext in ("*.sycl", "*.h"):
            for f in tree.rglob(ext):
                sources += f.read_text()

    stale = []
    # Only messages from OUR kernels are checkable; a comment quoting upstream
    # sgl_kernel or torch is accurate and simply not in this repo, so scope by
    # attribution below. Double quotes within one line only: allowing single
    # quotes matches apostrophes in prose and isolates no message at all.
    for m in re.finditer(r'^\+\s*#[^\n]*?"([^"\n]{20,120})"', text, re.M):
        frag = m.group(1)
        if not re.search(r"must |requires |only |unsupported ", frag):
            continue
        # Attribute by the SAME comment block: walk back over contiguous `+ #`
        # lines. A character window catches `torch.` from neighbouring code and
        # excuses every fragment, including a genuinely stale one.
        block, at = [], m.start()
        while at > 0:
            ls = text.rfind("\n", 0, at) + 1
            line = text[ls:text.index("\n", ls) if "\n" in text[ls:] else len(text)]
            if not re.match(r"\+\s*#", line):
                break
            block.append(line)
            at = ls - 1
        if re.search(r"sgl_kernel|sgl_per_|upstream", " ".join(block)):
            continue
        if frag not in sources:
            stale.append(frag)
    assert not stale, (
        "a gate comment quotes a kernel message that no longer exists, so the "
        "condition it justifies may be stricter than the kernel: "
        f"{stale[:3]}"
    )


# A kernel pinning a literal numel/sizes needs a shape term in its gate, or a
# mismatched shape aborts instead of falling back.

def test_shape_pinned_kernels_have_a_shape_term_in_their_gate():
    """If a kernel pins a literal extent, its caller must pin it too."""
    if not _SGL.exists():
        pytest.skip("sglang tree not present")
    kern = (_SGL / "xpu/esimd_kernel.sycl")
    if not kern.exists():
        pytest.skip("esimd_kernel.sycl not present")
    ksrc = kern.read_text()
    text = _patch(_SGL_PATCH)

    # Ops whose kernel pins a literal numel/sizes. Scope bodies to the NEXT
    # function definition: a fixed character window runs into the following
    # op's literals and reports a dim-relative op as pinned.
    ents = [(m.group(1), m.start()) for m in re.finditer(
        r"(?:void|at::Tensor|std::vector<at::Tensor>)\s+(esimd_\w+)\s*\(", ksrc)]
    pinned = {}
    for i, (op, at) in enumerate(ents):
        end = ents[i + 1][1] if i + 1 < len(ents) else len(ksrc)
        lits = re.findall(r"numel\(\) == (\d{3,})", ksrc[at:end])
        if lits:
            pinned[op] = sorted(set(lits))

    # Two corrections. (1) text.find() landed on the module-level
    # `_esimd_x = None` initializer for every op -- 18,030 characters from the
    # real gate -- so the window covered the import block. Skip initializer and
    # import lines. (2) The `"shape" not in gate` fallback was satisfied by one
    # incidental `for expert_id in range(data.shape[0])`, which made every
    # possible failure unreachable: this test had never fired.
    tc = patch_code(text)
    missing = []
    for op, lits in pinned.items():
        short = op[len("esimd_"):]
        gi = None
        for m in re.finditer(rf"_esimd_{re.escape(short)}\b", tc):
            ls = tc.rfind("\n", 0, m.start()) + 1
            le = tc.find("\n", m.start())
            line = tc[ls:le if le > 0 else len(tc)]
            if re.search(r"=\s*None|import|as _esimd_", line):
                continue
            gi = m.start()
            break
        if gi is None:
            continue
        gate = tc[max(0, gi - 2500):gi + 1500]
        absent = [l for l in lits if l not in gate]
        if absent:
            missing.append(
                f"{op} pins {lits}; its gate does not constrain {absent}")
    assert not missing, (
        "a kernel pinning a literal extent whose caller does not pin it aborts "
        "the forward pass on any other shape; vllm's twin of the same op is "
        f"dim-relative, so the trees disagree: {missing}"
    )


# Where a kernel requires two tensors to share a scalar type, checking one of
# them is not enough: a quantised KV cache leaves q and the caches disagreeing.

def test_gate_covers_cross_tensor_dtype_equalities():
    """Where a kernel equates two tensors' dtypes, the gate must test both."""
    kern = _SGL / "xpu/esimd_kernel_prefill_dpas.sycl"
    if not kern.exists():
        pytest.skip(f"{kern} not present")
    ksrc = kern.read_text()
    text = _patch(_SGL_PATCH)

    # Which tensors the kernel equates to q.
    equated = set(re.findall(
        r"q\.scalar_type\(\) == (\w+)\.scalar_type\(\)", ksrc))
    assert equated, "the prefill kernel no longer equates any dtype to q's"

    gi = text.find('"SGL_XPU_PREFILL_DPAS"')
    assert gi >= 0, "the prefill-DPAS gate is gone"
    gate = text[gi:gi + 1800]
    assert "q.dtype == torch.float16" in gate, "the gate no longer pins q's dtype"
    qm = re.search(r"q\.dtype\s*==\s*(torch\.\w+)", gate)
    assert qm, "the gate no longer pins q's dtype"
    for t in sorted(equated):
        # The kernel requires EQUALITY with q, so pinning the cache to a
        # different dtype satisfies a name check and still aborts.
        tm = re.search(rf"{t}\.dtype\s*==\s*(torch\.\w+)", gate)
        if tm:
            assert tm.group(1) == qm.group(1), (
                f"{t}.dtype is pinned to {tm.group(1)} but q is {qm.group(1)}; "
                "the kernel requires equality, so this aborts on the first prefill"
            )
        assert f"{t}.dtype ==" in gate, (
            f"the kernel requires q.scalar_type() == {t}.scalar_type() but the "
            f"gate does not test {t}'s dtype: --kv-cache-dtype fp8_e4m3 aborts "
            "on the first prefill, and the flag ships enabled"
        )


# The twin trees must agree on an op's contract. One host dim-relative and the
# other pinned to a literal extent puts the pinned tree's caller at risk.

def test_twin_hosts_agree_on_relative_versus_pinned_extents():
    """The same op must not be dim-relative in one tree and literal in the other."""
    disagree = []
    for rel, op in (("xpu/esimd_kernel.sycl", "esimd_norm_gemv_norm_fp16"),):
        forms = {}
        for name, tree in (("vllm", _VLLM), ("sglang", _SGL)):
            path = tree / rel
            if not path.exists():
                continue
            src = path.read_text()
            i = src.find("void " + op + "(")
            if i < 0:
                continue
            body = src[i:i + 4000]
            literal = bool(re.search(r"residual\.numel\(\) == \d+", body))
            relative = bool(re.search(
                r"residual\.numel\(\) == (?:K|hidden)\b", body))
            forms[name] = "literal" if literal else (
                "relative" if relative else "none")
        # Both trees must have been examined. Two `continue`s can fire together
        # -- renaming the op in BOTH trees leaves forms empty, len(set()) == 0,
        # and the test passes having compared nothing. Proven: injecting a
        # literal extent on the vllm side fails, and the same injection with
        # the op renamed passes.
        assert len(forms) == 2, (
            f"{op}: examined {sorted(forms)} -- the anchor moved in at least "
            "one tree and there is nothing left to compare"
        )
        if len(set(forms.values())) > 1:
            disagree.append(f"{op}: {forms}")
    assert not disagree, (
        "twin hosts disagree about whether an extent is pinned or derived; the "
        "pinned side aborts on any shape its caller gate does not pre-filter: "
        f"{disagree}"
    )


# A divisibility the kernel TORCH_CHECKs must appear in the gate. Sharding makes
# these reachable: intermediate_size divides by the TP degree.

def test_moe_gate_tests_the_divisibility_the_kernel_enforces():
    """Every % N the MoE kernel enforces must appear in its eligibility gate."""
    kern = _VLLM / "moe_batch/moe.sycl"
    if not kern.exists():
        pytest.skip(f"{kern} not present")
    ksrc = kern.read_text()
    i = ksrc.find("moe_forward_full_fp8_block(")
    assert i >= 0, "moe_forward_full_fp8_block not found"
    body = ksrc[i:i + 3000]
    mods = set(re.findall(r"(?:hidden_size|intermediate_size) % (\d+) == 0", body))
    assert mods, "the kernel no longer pins a divisibility -- re-derive this test"

    # patch_code(), not raw: the needle "% 128 == 0" was satisfied by the COMMENT
    # documenting the check, so deleting the predicate line left this green.
    text = patch_code(_patch(_VLLM_PATCH))
    gi = text.find("_esimd_moe_weight_eligible = ")
    assert gi >= 0, "the MoE eligibility gate is gone"
    gate = text[max(0, gi - 2500):gi + 400]
    # Pin the two terms by the quantity each one bounds, not by the modulus alone.
    for mod in sorted(mods):
        assert re.search(rf"shape\[2\]\s*%\s*{mod}\s*==\s*0", gate), (
            f"the kernel enforces hidden_size % {mod} and the gate does not"
        )
        assert re.search(rf"shape\[1\]\s*//\s*2\)\s*%\s*{mod}\s*==\s*0", gate), (
            f"the kernel enforces intermediate_size % {mod} and the gate does not"
        )
    # A vacuous term would satisfy every assertion above.
    dm = re.search(r"_dims_ok = [^\n]*(?:\n\+[^\n]*){0,3}", gate)
    assert dm, "cannot locate the _dims_ok definition"
    # Evaluate it, do not pattern-match it. Banning `or`/`True` is too strict
    # -- legitimate disjunctions exist here -- and too weak: `any((<pred>, 1))`
    # is vacuous with neither token present.
    # Cut the RHS at its own balanced end, not at a fixed line count, or the
    # extracted text is a syntax error; a parse failure must not read as
    # "not vacuous".
    rhs = _balanced_rhs(dm.group(0).split("=", 1)[1])
    # BOTH checks, because they answer different questions. The syntactic ban
    # catches a TARGETED weakening -- `A and (B and C or True)` still refuses
    # when A is false, so it is not vacuous, but the divisibility subterm is
    # dead. The semantic check catches a WHOLESALE one -- `any((expr, 1))` is
    # true regardless and carries neither banned token. Replacing one with the
    # other loses half the coverage; I tried, and MUT A went green.
    assert not re.search(r"\bTrue\b|\bor\b", rhs), (
        f"_dims_ok has a vacuous term: {rhs[:140]}"
    )
    assert not is_vacuous_predicate(rhs), (
        f"_dims_ok is true with every leaf false, so it refuses nothing: "
        f"{dm.group(0)[:140]}"
    )
    # The use sites too: checking only the definition left `(_dims_ok or True)`
    # passing.
    for flag in ("_esimd_moe_is_block", "_esimd_moe_weight_eligible"):
        at = gate.find("self." + flag + " = ")
        assert at >= 0, f"{flag} is not assigned in the gate"
        stmt = gate[at:]
        nxt = stmt.find("self._esimd_moe_", len(flag) + 8)
        stmt = stmt[:nxt] if nxt > 0 else stmt[:400]
        assert "_dims_ok" in stmt, (
            f"{flag} ignores the divisibility predicate: {stmt.strip()[:110]}"
        )
        rhs2 = _balanced_rhs(stmt.split("=", 1)[1] if "=" in stmt else stmt)
        assert not re.search(r"_dims_ok\s+or\b|\bor\s+_dims_ok", rhs2), (
            f"{flag} softens _dims_ok with an `or`: {rhs2[:110]}"
        )
        assert not is_vacuous_predicate(rhs2), (
            f"{flag} is true with every leaf false: {stmt.strip()[:110]}"
        )
    for mod in sorted(mods):
        assert f"% {mod} == 0" in gate, (
            f"the kernel enforces % {mod} on hidden/intermediate but the gate "
            "does not, so a failing shape aborts the forward pass instead of "
            "falling back (Qwen3-Next-80B-A3B at TP=8 gives I=64)"
        )
    # Present is not enough: the predicate must be USED in the eligibility
    # expression. Dropping `and _dims_ok` from both assignments left the `% 128`
    # text sitting in the now-dead definition, and a grep for it still passed.
    m = re.search(r"_esimd_moe_weight_eligible = \(?\s*([^\n]*(?:\n\+[^\n]*){0,3})",
                  text[gi - 30:gi + 400])
    assert m, "cannot read the eligibility expression"
    expr = m.group(1)
    assert "_dims_ok" in expr, (
        "the divisibility predicate is defined but not used in the eligibility "
        f"expression: {expr.strip()[:110]}"
    )
    blk = re.search(r"_esimd_moe_is_block = ([^\n]*)", text[gi - 900:gi + 200])
    assert blk and "_dims_ok" in blk.group(1), (
        "the block-path flag ignores the divisibility predicate: "
        f"{blk.group(1).strip()[:90] if blk else 'not found'}"
    )


# A file reinterpret_cast-ing to a concrete float type needs a scalar_type test:
# reading bf16 through fp16's exponent/mantissa split is silent, not a throw.

def test_lgrf_entries_check_the_dtype_they_reinterpret():
    """A file that reinterprets storage must verify the storage type."""
    path = _SGL / "xpu/esimd_kernel_lgrf.sycl"
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    casts = len(re.findall(r"reinterpret_cast<(?:const )?fp16\*>", src))
    assert casts, "no fp16 reinterpret_cast found -- re-derive this test"
    checks = len(re.findall(r"scalar_type\(\) == at::kHalf", src))
    assert checks >= 5, (
        f"{casts} reinterpret_cast<fp16*> against only {checks} dtype checks; a "
        "bf16 tensor is read with fp16's exponent split, silently"
    )
    # Every entry point that casts must carry the check, not just one of them.
    entries = [m.start() for m in re.finditer(
        r"auto\* p_qkvz\s*= reinterpret_cast<const fp16\*>", src)]
    assert entries, "the qkvz cast moved -- re-derive this test"
    for at in entries:
        head = src[max(0, at - 1600):at]
        assert "scalar_type() == at::kHalf" in head, (
            "an lgrf entry point reinterprets its tensors as fp16 with no dtype "
            "check above the casts"
        )


# The page-attention kernel requires a power-of-two page_size with
# 6 <= log2(page_size) <= 10, so the gate needs a page_size term.

def test_page_attn_gate_tests_page_size():
    """The gate must bound page_size the way the kernel does."""
    kern = _SGL / "eagle/eagle.sycl"
    if not kern.exists():
        pytest.skip(f"{kern} not present")
    ksrc = kern.read_text()
    assert re.search(r"pageTableSizeLog2 >= 6", ksrc), (
        "the kernel no longer bounds page_size below -- re-derive this test"
    )
    text = _patch(_SGL_PATCH)
    gi = text.find("_pa_gqa_ok = (")
    assert gi >= 0, "the page-attention gate is gone"
    gate = text[max(0, gi - 1400):gi + 900]
    assert "_pa_page_ok" in gate, (
        "the gate does not bound page_size; the kernel requires a power of two "
        "in [64, 1024] and every shipped script passes exactly the minimum"
    )
    # The NAME is not the bound. Gutting the predicate to `_pa_page >= 1`,
    # dropping the lower bound, or dropping the power-of-two clause all kept
    # this test green while fully reintroducing the --page-size 32 abort.
    # Pin the three terms by their operators.
    di = gate.find("_pa_page_ok = (")
    assert di >= 0, "_pa_page_ok is no longer defined here"
    pred = gate[di:di + 300]
    assert "64 <= _pa_page" in pred, (
        f"the lower bound on page_size is gone: {pred[:120]}"
    )
    assert "_pa_page <= 1024" in pred, (
        f"the upper bound on page_size is gone: {pred[:120]}"
    )
    assert "& (_pa_page - 1)) == 0" in pred, (
        f"the power-of-two clause is gone; the kernel masks the page offset "
        f"with page_size - 1, so a non-power-of-two selects the wrong KV "
        f"token rather than failing: {pred[:120]}"
    )
    # ...and the predicate must be used, not merely defined.
    use = text[gi:gi + 900]
    assert "and _pa_page_ok" in use, (
        "_pa_page_ok is defined but not part of the gate's conjunction"
    )


# The int4 MoE gate must divide by the same constant the kernels stride expert
# scales by: K_groups = <axis> / 128 floors to 0 below 128, pointing every
# expert at expert 0's scales, and a remainder mis-strides them. Silent.

@pytest.mark.parametrize("tree", [_VLLM, _SGL], ids=["vllm", "sglang"])
def test_int4_moe_gate_matches_the_scale_stride_divisor(tree):
    kern = tree / "moe_batch/moe_int4.sycl"
    if not kern.exists():
        pytest.skip(f"{kern} not present")
    src = code(kern.read_text())

    # The divisor the kernels actually stride expert scales by.
    divisors = {int(d) for d in re.findall(
        r"K_groups = (?:intermediate_size|hidden_size) / (\d+)", src)}
    assert divisors, "K_groups is no longer derived by division -- re-derive this test"
    divisor = max(divisors)

    i = src.find("moe_forward_full_int4(")
    assert i >= 0, "moe_forward_full_int4 not found"
    body = src[i:i + 6000]
    # The shared axis carries the same arithmetic: K_groups_down =
    # shared_intermediate_size / 128 strides the per-shared-expert scale base.
    for axis in ("intermediate_size", "hidden_size"):
        mods = {int(m) for m in re.findall(
            rf"TORCH_CHECK\({axis} % (\d+) == 0", body)}
        assert mods, f"the int4 entry point no longer gates {axis} at all"
        assert max(mods) >= divisor, (
            f"{axis} is gated at % {max(mods)} but the kernels stride the "
            f"per-expert scale base by <axis> / {divisor}; values below "
            f"{divisor} collapse every expert onto expert 0's scales"
        )
    # shared_intermediate_size is gated PER BRANCH, and must be: only the int4
    # kernels divide it by 128, while the fp16 shared kernels have no /128 at
    # all -- their granularity is `k += 64` with a fixed block_load<fp16,64>.
    # Pinning %128 unconditionally here is what made the C++ refuse 64 values
    # in [64,8192] that the fp16 path serves exactly, S=64 among them, which is
    # what the repo's own 35B-A3B config gives at TP=8. Assert the selector and
    # both arms, not one literal.
    assert re.search(
        r"shared_mod = shared_is_int4 \? (\d+) : (\d+)", body), (
        "shared_intermediate_size is not gated per branch; a single modulus is "
        "either too weak for the int4 scale stride or too strict for fp16"
    )
    sm = re.search(r"shared_mod = shared_is_int4 \? (\d+) : (\d+)", body)
    i4, f16 = int(sm.group(1)), int(sm.group(2))
    assert i4 >= divisor, (
        f"the int4 arm gates % {i4} but the kernels stride the shared scale "
        f"base by S / {divisor}"
    )
    assert f16 == 64, (
        f"the fp16 arm gates % {f16}; those kernels load 64 lanes at a time "
        "with no tail, so 64 is both necessary and sufficient"
    )
    assert re.search(r"shared_intermediate_size % shared_mod == 0", body), (
        "the per-branch modulus is computed but never applied"
    )
    # The intermediates buffer row is sized by the ROUTED intermediate size
    # while the shared-up kernels write [0, shared_intermediate_size). A larger
    # shared size writes into the next row -- a different invariant from the
    # modulus, so it needs its own check.
    # The shared-up kernels write [0, shared_intermediate_size) into the
    # intermediates buffer, whose row was sized by the ROUTED size alone -- so
    # S > I wrote out of row. Gating S <= I fixed the corruption but refused
    # the whole Qwen shared-expert MoE family, where S > I at every TP. The
    # invariant that holds without refusing anything is that the row is sized
    # by the larger of the two.
    assert re.search(
        r"inter_row_width\s*=\s*std::max\(intermediate_size,\s*"
        r"shared_intermediate_size\)", src), (
        "the intermediates buffer row is not sized by max(intermediate, "
        "shared); a shared size larger than the routed one writes out of row"
    )
    # The allocation uses the GROWN width (max of the requested and the cached),
    # not the request, so that a later narrower shape cannot shrink a buffer a
    # captured graph still points at. Either name is acceptable; what must not
    # appear is the routed size alone.
    assert re.search(r"rows_per_token, (?:inter_row_width|row_w)\}", src), (
        "the intermediates buffer is still allocated with the routed "
        "intermediate size"
    )
    assert not re.search(r"rows_per_token, intermediate_size\}", src), (
        "the intermediates buffer is allocated by the routed size, so a larger "
        "shared expert writes out of row"
    )


# Every up kernel tiles N as intermediate_size / 16 and stores 16 at a time, so
# a remainder leaves tail columns of the intermediates buffer unwritten. Most
# moe_forward_* entries launch the tiled kernels directly rather than through
# moe_up_forward, so derive the requirement from the kernels each entry launches.

def _strip_comments_keep_lines(src):
    """Drop // and /* */ comments, preserving newlines."""
    out, i, n = [], 0, len(src)
    while i < n:
        if src[i] == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src[i] == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            seg = src[i:n if j < 0 else j + 2]
            out.append("\n" * seg.count("\n"))
            i = n if j < 0 else j + 2
        else:
            out.append(src[i]); i += 1
    return "".join(out)


def _template_widths(name, whole):
    """Concrete values a template parameter is instantiated with, file-wide.

    A stride like `k += VecWidth` has no value in the kernel body -- it is a
    template parameter chosen at the launch site
    (`launch_up.template operator()<64>()`). The binding requirement is the
    NARROWEST instantiation, since that is the smallest tile any caller can get.
    """
    vals = {int(x) for x in re.findall(r"operator\(\)<(\d+)>", whole)}
    return vals


def _fold_stride(expr, body, whole=None):
    """Resolve a symbolic loop stride to an int, or None.

    Handles a constant defined in the same body (`constexpr int C = 64;`, a
    `static const`, or a template parameter with a default) and simple products
    of such names and literals. Returns None when it genuinely cannot tell --
    the caller turns that into a loud failure rather than a silent zero.
    """
    consts = {}
    for m in re.finditer(
            r"(?:constexpr|static const|const)\s+\w+\s+(\w+)\s*=\s*(\d+)", body):
        consts[m.group(1)] = int(m.group(2))
    for m in re.finditer(r"(?:int|size_t|uint32_t)\s+(\w+)\s*=\s*(\d+)\s*;", body):
        consts.setdefault(m.group(1), int(m.group(2)))
    for m in re.finditer(r"template\s*<[^>]*int\s+(\w+)\s*=\s*(\d+)", body):
        consts.setdefault(m.group(1), int(m.group(2)))
    tokens = re.split(r"\s*\*\s*", expr.strip())
    total = 1
    for t in tokens:
        t = t.strip()
        if t.isdigit():
            total *= int(t)
        elif t in consts:
            total *= consts[t]
        elif whole is not None and re.search(rf"template\s*<\s*int {t}\s*>", whole):
            widths = _template_widths(t, whole)
            if not widths:
                return None
            total *= min(widths)     # narrowest instantiation binds
        else:
            return None
    return total


def _moe_kernel_divisors(src):
    """kernel name -> the set of extents it tiles hidden/intermediate by."""
    out = {}
    # `^void` alone misses `static void` dispatchers, and sglang routes the
    # gelu up/down kernels through two of them -- so their tiling was invisible
    # and four entries got a weaker gate than their vllm twins.
    # Body extent = up to the NEXT match, not the next non-indented `void `.
    # `find("\nvoid ")` ran past every function followed by `static void` or a
    # template, so one "body" was 1187 lines -- the rest of the file, including
    # the TORCH_LIBRARY block. Benign today only because the imported values
    # happened to be small; an over-run body can import a LARGER divisor from a
    # neighbour and manufacture a false over-strict demand.
    starts = [m for m in re.finditer(
        r"^(?:static\s+)?(?:inline\s+)?void (\w+)\(", src, re.M)]
    for idx, m in enumerate(starts):
        s = m.start()
        e = starts[idx + 1].start() if idx + 1 < len(starts) else len(src)
        body = src[s:e]
        # Per AXIS. Lumping them together makes the gate over-strict: hidden is
        # usually tiled by 64 and intermediate by 16, and a blanket %64 would
        # refuse gemma4-26B at TP=2/4/8 (hidden 2816, intermediate 352/176/88)
        # for a constraint only the hidden axis imposes. An over-tight gate is a
        # silent perf cliff -- the defect class on the other side of this one.
        per = {}
        for ax in ("intermediate_size", "hidden_size"):
            # Literal divisions AND loop strides. Reading only `axis / N` is
            # blind to `for (k = 0; k < axis; k += N)` with a fixed N-wide
            # block_load and no tail -- three kernels walk hidden that way, and
            # relaxing their entry to %32 on the strength of the /N literals
            # alone re-opened a 64-byte overread per row per shared expert.
            d = {int(x) for x in re.findall(rf"{ax} / (\d+)", body)}
            # Every loop over this axis, whatever the stride TOKEN. A regex that
            # matches only literal strides is blind to `k += VL`, `k += WG_SIZE`,
            # `k += 16 * GS` and `k += SG * HB` -- 38 such loops exist right now
            # across the four MoE files. A symbolic stride must RESOLVE or FAIL:
            # contributing nothing reads as "this kernel tiles by nothing",
            # which is how a gate gets relaxed below what its kernel needs.
            flat = re.sub(r"\s+", " ", body)
            # A COOPERATIVE loop -- `for (k = W*tid; k < axis; k += W*GS)` --
            # has GS threads each taking W-wide tiles, so the binding
            # granularity is the TILE width W, not the stride W*GS. Reading the
            # stride there manufactures a false /256 demand on kernels that
            # actually need /16. Take the init's multiplier when the loop starts
            # at a thread-dependent offset.
            for init, step in re.findall(
                    rf"for *\( *\w+ +\w+ *= *([^;]+?) *; *[^;]*< *{ax} *; "
                    rf"*\w+ *\+= *([^)]+?) *\)", flat):
                if re.search(r"\b(tid|lid|sg|local_id|item)\b", init) and "*" in init:
                    # `off = lid * HB` -> the tile width is the factor that is
                    # NOT the thread id. Taking the first factor blindly yields
                    # `lid`, which is runtime-varying and resolves to nothing.
                    factors = [f.strip() for f in init.split("*")]
                    widths = [f for f in factors
                              if not re.fullmatch(r"tid|lid|sg|local_id|item", f)]
                    if widths:
                        step = " * ".join(widths)
                step = step.strip()
                if step.isdigit():
                    d.add(int(step))
                    continue
                folded = _fold_stride(step, flat, src)
                assert folded is not None, (
                    f"{m.group(1)}: cannot resolve the stride {step!r} over "
                    f"{ax}. An unresolved stride silently contributes nothing "
                    "to the derived requirement -- define it as a literal, or "
                    "teach _fold_stride about it."
                )
                d.add(folded)
            if d:
                per[ax] = d
        # A kernel with a tail arm needs only the tail's granularity, not the
        # main tile's. moe_accumulate_kernel tiles hidden by 64 and then submits
        # a 32-lane tail for the remainder, so demanding %64 of its callers
        # rejects (288, 64) -- a shape the repo's own
        # test_canonical_decode_handles_32_element_tail_chunks names as
        # intended. Reading the /64 literal and stopping is how that happened.
        tail = re.search(r"tail_lanes\s*=\s*hidden_size\s*-\s*tail_base", body)
        if tail and "hidden_size" in per:
            # Scope to the TAIL arm, and take the WIDEST access in it -- loads
            # included.
            #
            # Every access in the tail must fit inside the remainder, so the
            # binding requirement is the WIDEST, not the narrowest: a
            # function-scoped `min` over stores alone lets a narrow store in
            # the main arm drag it down and never scans the load.
            tl = re.search(r"auto cgf_tail\s*=", body)
            region = body[tl.start():] if tl else body
            widths = {int(w) for w in re.findall(
                r"block_(?:store|load)<fp16, ?(\d+)>", region)}
            width = max(widths) if widths else None
            if width:
                per["hidden_size"] = {width}
        if per:
            out[m.group(1)] = per
    return out


@pytest.mark.parametrize("tree", [_VLLM, _SGL], ids=["vllm", "sglang"])
def test_every_moe_entry_gates_what_its_kernels_tile_by(tree):
    path = tree / "moe_batch/moe.sycl"
    if not path.exists():
        pytest.skip(f"{path} not present")
    # NOT code(): it collapses all whitespace onto one line, so the `^void`
    # scan below matches nothing and every entry point reads as clean. Comments
    # still have to go -- a commented-out TORCH_CHECK must not satisfy the gate
    # check -- so strip them without touching line structure.
    raw = _strip_comments_keep_lines(path.read_text())
    kdiv = _moe_kernel_divisors(raw)
    assert kdiv, "no tiled MoE kernels found -- re-derive this test"

    # std::vector<torch::Tensor> counts too: moe_forward_full_rtfused_norm
    # returns one and was skipped entirely by a `torch::Tensor `-anchored scan,
    # which is how it kept its sibling's guards from reaching it.
    entries = [(m.start(), m.group(1)) for m in
               re.finditer(r"(?:std::vector<)?torch::Tensor>? (moe_forward_\w+)\(", raw)]
    assert entries, "no moe_forward_* entry points found"

    offenders = []
    for k, (pos, name) in enumerate(entries):
        end = entries[k + 1][0] if k + 1 < len(entries) else len(raw)
        body = raw[pos:end]
        need = {}
        for kname, per in kdiv.items():
            if re.search(rf"\b{re.escape(kname)}\s*\(", body):
                for ax, d in per.items():
                    need.setdefault(ax, set()).update(d)
        for ax, divs in sorted(need.items()):
            mod = max(divs)
            have = {int(x) for x in re.findall(
                rf"TORCH_CHECK\({ax} % (\d+)", body)}
            if not have or max(have) < mod:
                offenders.append(
                    f"{name}: launches kernels tiling {ax} by /{mod} but gates "
                    f"{sorted(have) or 'nothing'}")
    assert not offenders, (
        "a MoE entry point admits shapes its own kernels truncate; the tail "
        "columns are left unwritten in a reused buffer, so this is silent: "
        + "; ".join(offenders)
    )


# The canonical decode path checks intermediate_size % 32 and the native one
# % 16. The call is bare, so an admitted-but-refused shape aborts on the first
# decode token rather than falling back.

def test_gemma4_moe_decode_gate_has_a_modulus_term():
    kern = _VLLM / "moe_batch/moe.sycl"
    if not kern.exists():
        pytest.skip(f"{kern} not present")
    ksrc = code(kern.read_text())
    mods = set()
    for fn in ("dispatch_moe_up_decode", "dispatch_moe_down_decode"):
        i = ksrc.find(fn + "(")
        assert i >= 0, f"{fn} not found -- re-derive this test"
        mods |= {int(m) for m in re.findall(
            r"TORCH_CHECK\(intermediate_size % (\d+)", ksrc[i:i + 2500])}
    assert mods, "the canonical decode path no longer pins intermediate_size"

    text = patch_code(_patch(_VLLM_PATCH))
    gi = text.find("_full_moe_ready = True")
    assert gi >= 0, "the gemma4 full-fused MoE gate is gone"
    gate = text[max(0, gi - 4000):gi]
    assert "native_layout" in gate, "the gate no longer distinguishes the layouts"
    assert re.search(r"%\s*_mod\s*!=\s*0|_inter\s*%", gate), (
        "the gate has no intermediate_size modulus term; the canonical kernel "
        f"requires % {max(mods)} and the call site is bare, so a failing shape "
        "aborts on the first decode token instead of falling back"
    )

    # The shape of the expression is not the contract. Asserting only that some
    # `% _mod` appears let three mutations through with the suite green:
    # swapping the two shape axes (which makes _inter = hidden/2 = 1408 at every
    # TP, so the gate admits unconditionally and reverts to the no-op this test
    # exists to prevent), swapping the two moduli, and widening _mod to 128
    # (which kills the fast path at every TP, including the TP=2 the fix cites
    # as proof the path works). Pin the axis and compare the moduli to the
    # kernel's own.
    #
    # _get_grouped_moe_weights defines canonical as w13.shape[2] == hidden and
    # native as w13.shape[1] == hidden, so intermediate is the OTHER axis in
    # each case -- shape[1] for canonical, shape[2] for native.
    assert re.search(
        r"_inter\s*=\s*w13\.shape\[2\]\s*//\s*2\s*if\s*native_layout"
        r"\s*else\s*w13\.shape\[1\]\s*//\s*2", gate), (
        "the gate reads intermediate_size off the wrong axis; on the canonical "
        "layout w13.shape[2] IS hidden_size, so this admits every shape"
    )
    down = ksrc.find("dispatch_moe_down_decode(")
    want_canonical = max(int(m) for m in re.findall(
        r"TORCH_CHECK\(intermediate_size % (\d+)", ksrc[down:down + 2500]))
    ni = ksrc.find("moe_forward_full_gelu_tanh_decode_native(")
    assert ni >= 0, "the native decode entry is gone -- re-derive this test"
    want_native = max(int(m) for m in re.findall(
        r"intermediate_size % (\d+) == 0", ksrc[ni:ni + 2500]))
    m = re.search(r"_mod\s*=\s*(\d+)\s*if\s*native_layout\s*else\s*(\d+)", gate)
    assert m, "the gate no longer selects a modulus per layout"
    got_native, got_canonical = int(m.group(1)), int(m.group(2))
    assert got_canonical == want_canonical, (
        f"the canonical arm gates % {got_canonical} but dispatch_moe_down_decode "
        f"requires % {want_canonical}"
    )
    assert got_native == want_native, (
        f"the native arm gates % {got_native} but the native entry requires "
        f"% {want_native}"
    )
    assert re.search(r"%\s*64\s*!=\s*0", gate), (
        "the gate has no hidden_size % 64 term"
    )


# The decode kernels require key_cache and value_cache to match q, or to be
# kHalf outright. The calls are bare, so a gate testing only q's dtype lets a
# quantised KV cache abort on the first decode token.

def test_sglang_decode_gates_check_the_kv_cache_dtypes():
    text = patch_code(_patch(_SGL_PATCH))
    for anchor, what in (("_use_esimd_pa = (", "page_attn decode"),
                         ("_splitk_decode_attention is not None", "split-K decode")):
        i = text.find(anchor)
        assert i >= 0, f"the {what} gate is gone"
        gate = text[i:i + 1600]
        # The right-hand side is the contract. `key_cache.dtype ==` alone was
        # satisfied by `key_cache.dtype == key_cache.dtype`, which reintroduces
        # the fp8-KV abort on gemma4's ten global layers with the suite green,
        # and by `== torch.bfloat16`, which refuses everything instead.
        # page_attn requires the caches to MATCH q (eagle.sycl:451-454);
        # split-K requires all four operands to be kHalf (:2243-2248).
        rhs = (r"q_reshaped\.dtype" if what == "page_attn decode"
               else r"torch\.float16")
        for cache in ("key_cache", "value_cache"):
            assert re.search(rf"{cache}\.dtype\s*==\s*{rhs}\b", gate), (
                f"the {what} gate does not tie {cache}.dtype to {rhs}; the "
                "kernel requires it and the call site is bare, so "
                "--kv-cache-dtype fp8_e4m3 aborts on the first decode token"
            )


def test_splitk_gate_does_not_borrow_page_attn_s_page_size_bound():
    """The 64..1024 bound belongs to page_attn_decode, a different kernel.

    splitk_decode_attention requires only that page_size be a power of two
    (eagle.sycl), and derives pageMask/pageSizeLog2 from key_cache.size(1), so
    it serves 8/16/32 correctly. Importing _pa_page_ok here would refuse
    --page-size 32 that the kernel handles -- the too-strict direction, which
    fails nothing and so is never noticed.
    """
    text = patch_code(_patch(_SGL_PATCH))
    i = text.find("_splitk_decode_attention is not None")
    if i < 0:
        pytest.skip("split-K gate not present")
    gate = text[i:i + 1600]
    assert "_pa_page_ok" not in gate, (
        "the split-K gate imported page_attn's 64..1024 page_size bound; "
        "split-K only needs a power of two, so this refuses --page-size 32 "
        "that the kernel serves exactly"
    )


# The fp8 linear kernel bounds N itself (`if (n >= N) return;`), so the gate
# needs no N modulus. K is M-dependent: at M==1 the GEMV selector falls through
# to a masked-tail arm taking any remainder; at M>=2 the GEMM narrow-N arm
# hard-asserts K % 128.

def test_fp8_linear_gate_is_m_conditional_and_has_no_n_term():
    text = patch_code(_patch(_VLLM_PATCH))
    i = text.find("def _try_esimd_fp8_linear(")
    assert i >= 0, "the fp8 linear adapter is gone"
    gate = text[i:i + 3000]

    assert not re.search(r"weight\.shape\[0\] % \w*ALIGNMENT", gate), (
        "the gate constrains N; fp8_GEMV_bmg.h bounds n >= N itself, so N only "
        "sizes the grid and every N is servable"
    )
    m = re.search(
        r"weight\.shape\[1\] % \((\d+) if x\.shape\[0\] == 1 else (\d+)\) != 0",
        gate)
    assert m, (
        "the K term is not M-conditional; one modulus is either too strict for "
        "the GEMV masked tail or too weak for the GEMM narrow-N arm"
    )
    m1, m2 = int(m.group(1)), int(m.group(2))
    assert m2 == 128, (
        f"the M>=2 arm gates % {m2}; GEMM_fp8_pert_dispatch hard-asserts "
        "K % 128 on the narrow-N path, so anything weaker aborts"
    )
    assert m1 <= 64, (
        f"the M==1 arm gates % {m1}; the GEMV masked-tail arm covers an "
        "arbitrary remainder, and this refuses K=1056 among others"
    )


# s_int4_intermediates is one row space, indexed by the routed kernels (row =
# token*rows_per_token + k_idx) and the shared ones (row = ... + top_k + sid).
# Allocation width and the routed kernels' stride must widen together, or two
# strides share one buffer and routed rows land inside earlier rows. Silent.

@pytest.mark.parametrize("tree", [_VLLM, _SGL], ids=["vllm", "sglang"])
def test_int4_intermediates_has_one_stride(tree):
    path = tree / "moe_batch/moe_int4.sycl"
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = code(path.read_text())

    # Only indexing into the INTERMEDIATES buffer. `row * hidden_size` is a
    # legitimate stride into routed_output, which is [rows, hidden] -- a
    # different buffer. Scope by the pointer being indexed.
    # Scope to the kernels that take buffer_row_width -- those are the ones
    # sharing s_int4_intermediates. Other kernels here index privately
    # torch::empty'd buffers of their own, with their own consistent stride, and
    # a file-wide scan flags those as violations.
    #
    # Within that scope, accept ANY index variable: requiring the literal names
    # routed_row|shared_row is a name test, and `const int rrow = routed_row;`
    # then `rrow * intermediate_size` evades it with the buffer back to two
    # strides.
    strides = set()
    for km in re.finditer(r"void (\w+)\(([^{]*?buffer_row_width[^{]*?)\)\s*\{",
                          src):
        ks = km.end()
        ke = src.find("void ", ks)
        body = src[ks:ke if ke > 0 else len(src)]
        strides |= set(re.findall(
            r"intermediates(?:\[|\s*\+\s*)\(size_t\)\s*\w+\s*\*\s*(\w+)",
            body))
    assert strides, "no intermediates row indexing found -- re-derive this test"
    assert strides == {"buffer_row_width"}, (
        f"the intermediates buffer is strided by {sorted(strides)}; routed and "
        "shared rows share one row space, so a second stride makes every row "
        "past the first land inside an earlier one"
    )
    # ...and the CALL SITES must bind that parameter to the allocator's width.
    # Grepping bodies alone is a name test: the parameter is uniformly called
    # buffer_row_width inside every kernel, so the body check returns
    # {'buffer_row_width'} whatever the callers pass. Threading the width into
    # the routed kernels while the shared calls still pass intermediate_size
    # moves the second stride rather than removing it.
    raw = path.read_text()
    bad = []
    for i, line in enumerate(raw.splitlines()):
        if i and "shared_intermediate_size," in raw.splitlines()[i - 1] \
                and line.strip() == "intermediate_size,":
            bad.append(i + 1)
    assert not bad, (
        f"{tree.name}: the argument after shared_intermediate_size is "
        f"intermediate_size at line(s) {bad}; that position is "
        "buffer_row_width, so the shared kernels stride by the routed size "
        "and their rows land inside the routed ones"
    )

    # LAST WRITE on the width. `int inter_row_width = std::max(...);` followed
    # by `inter_row_width = intermediate_size;` restores the pre-fix defect with
    # the max() still visible above it. Two declarations are legitimate (the
    # allocator and the entry, which re-reads the cached width), so check each
    # one's own scope rather than the file: every write must be a max() or the
    # cached value, never the routed size alone.
    for wm in re.finditer(r"inter_row_width\s*=\s*([^;]+);", src):
        rhs = " ".join(wm.group(1).split())
        assert ("std::max(intermediate_size, shared_intermediate_size)" in rhs
                or "s_int4_cached_row_w" in rhs), (
            f"{tree.name}: inter_row_width is assigned {rhs!r}; it must be "
            "max(routed, shared) or the width actually allocated"
        )

    # And the width must be the larger of the two, not either one alone.
    assert re.search(
        r"inter_row_width\s*=\s*std::max\(intermediate_size,\s*"
        r"shared_intermediate_size\)", src), (
        "the row width is not max(routed, shared)"
    )


# A buffer cache must key on every dimension it sizes by. ensure_int4_moe_buffers
# sizes by rows_per_token, hidden_size, row width, top_k and num_shared_experts,
# so keying on batch size alone lets a second shape reuse the first's buffers.

@pytest.mark.parametrize("tree", [_VLLM, _SGL], ids=["vllm", "sglang"])
def test_int4_buffer_cache_keys_on_every_sizing_dimension(tree):
    path = tree / "moe_batch/moe_int4.sycl"
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = code(path.read_text())
    i = src.find("static void ensure_int4_moe_buffers(")
    assert i >= 0, "ensure_int4_moe_buffers not found -- re-derive this test"
    body = src[i:i + 3000]

    # The quantities the allocations are sized by.
    sized_by = set()
    for m in re.finditer(r"torch::empty\(\s*\{([^}]*)\}", body):
        for tok in re.findall(r"\b(rows_per_token|hidden_size|inter_row_width|"
                              r"row_w|top_k|num_shared_experts)\b", m.group(1)):
            sized_by.add("row_w" if tok == "inter_row_width" else tok)
    assert sized_by, "no buffer allocations found -- re-derive this test"

    # The CONDITION only -- from `if (` to the `{` or `return;` that closes it.
    # Slicing to the first torch::empty swept in the grow-to-max assignments
    # that follow the guard, which mention every dimension and made the check
    # pass with the guard reverted to a single key.
    # The guard must name each dimension DIRECTLY. Following indirection --
    # a `shape_changed` flag, a `shape_now[]` array -- turned out to be the
    # trap: a reverted guard left the array in place as dead code and the
    # inlining picked it up, so the check passed with the defect restored.
    # Requiring the names inline is cruder and cannot be fooled that way; a
    # tree that prefers the array form can name the array's elements in a
    # comment on the guard, which is the honest cost.
    gm = re.search(r"if \(([^{]*?s_int4_cached.*?)\)\s*(?:return;|\{)", body)
    assert gm, "cannot locate the cache guard's condition"
    guard = gm.group(1)

    # The row width goes by two names: the request (inter_row_width) and the
    # grown value (row_w). Either in the guard keys on that dimension.
    aliases = {"row_w": ("row_w", "inter_row_width")}
    missing = sorted(q for q in sized_by
                     if not any(re.search(rf"\b{a}\b", guard)
                                for a in aliases.get(q, (q,))))
    assert not missing, (
        f"the cache guard ignores {missing}, but the buffers are sized by "
        "them; a second shape in this thread reuses buffers sized for the "
        "first and strides past their end"
    )


# The tiny fp16-shared kernels walk hidden_size (and shared_inter_size, in
# finalize) at stride 64 with a fixed block_load<fp16,64> and no tail, so their
# entries need the matching modulus or the load overreads the row.

@pytest.mark.parametrize("tree", [_VLLM, _SGL], ids=["vllm", "sglang"])
def test_tiny_fp16_shared_entries_gate_their_stride(tree):
    path = tree / "moe_batch/moe_int4.sycl"
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = code(path.read_text())
    seen = 0
    for fn, axes in (("moe_tiny_fp16_shared_up", ("hidden_size",)),
                     ("moe_tiny_fp16_shared_finalize",
                      ("hidden_size", "shared_inter_size"))):
        i = src.find(f"torch::Tensor {fn}(")
        if i < 0:
            continue
        seen += 1
        # Bound the body at the NEXT entry, not a fixed 4000 chars: the window
        # ran into moe_tiny_fp16_shared_finalize, whose gates satisfied the
        # assertion even with _up's deleted. A neighbour's guard is not this
        # entry's guard.
        nxt = src.find("torch::Tensor moe_", i + 20)
        body = src[i:nxt if nxt > 0 else i + 4000]
        for ax in axes:
            mods = {int(m) for m in re.findall(
                rf"TORCH_CHECK\({ax} % (\d+) == 0", body)}
            assert mods and max(mods) >= 64, (
                f"{fn} gates {ax} at {sorted(mods) or 'nothing'}; its kernel "
                "strides that axis by 64 with a fixed-width load and no tail"
            )
    assert seen == 2, (
        f"found {seen} of the 2 tiny fp16-shared entries; the anchor moved and "
        "this test stopped checking them"
    )


# moe.sycl's shared kernels take ONE intermediate_size for both the shared-weight
# stride and the intermediates-buffer stride, assuming S == I. That is false
# across the Qwen shared-expert family, so the entries must refuse S != I.
# At S > I both weight arms land inside the gate half, so the up projection
# silently computes a second gate; at S < I they read past the tensor.
#
# These gates refuse S != I rather than supporting it; supporting it needs the
# moe_int4.sycl treatment, with two stride splits here where that file has one.

@pytest.mark.parametrize("tree", [_VLLM, _SGL], ids=["vllm", "sglang"])
def test_moe_shared_entries_check_the_shared_intermediate_size(tree):
    path = tree / "moe_batch/moe.sycl"
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = code(path.read_text())
    # Bound entry bodies by static helpers too: slicing from one
    # `torch::Tensor moe_*` to the next runs through any `static void` between
    # them, and an entry then inherits a neighbour's shared launch.
    bounds = sorted(m.start() for m in re.finditer(
        r"(?:torch::Tensor|static\s+void|void)\s+\w+\(", src))
    entries = [(m.start(), m.group(1)) for m in
               re.finditer(r"torch::Tensor (moe_\w+)\(", src)]
    assert entries, "no moe entry points found -- re-derive this test"

    checked, ungated = 0, []
    for i, (pos, name) in enumerate(entries):
        nxt = [b for b in bounds if b > pos]
        end = nxt[0] if nxt else len(src)
        body = src[pos:end]
        # moe_down_finalize_* consumes the shared DOWN weight and strides it by
        # the routed size exactly as the _shared_ kernels do; a needle naming
        # only `_shared_` left every finalize-launching entry unexamined.
        if not re.search(r"moe_(?:(?:up|down)_shared|down_finalize)_\w+_kernel\(",
                         body):
            continue
        if "shared_gate_up_weight" not in body and \
                "shared_down_weight" not in body:
            continue          # takes no shared weight tensor
        checked += 1
        # Check that the refusal COVERS each tensor the entry consumes, not
        # that some numel() appears: gate_up and down are independent, and an
        # entry passing nullptr for an arm it does feed is not gated on it.
        call = re.search(r"moe_check_shared_inter_matches\(([^;]*?)\);", body,
                         re.S)
        if not call:
            ungated.append(f"{name} (no refusal at all)")
            continue
        args = " ".join(call.group(1).split())
        for tensor in ("shared_gate_up_weight", "shared_down_weight"):
            if re.search(rf"\b{tensor}\b", body) and f"&{tensor}" not in args:
                ungated.append(f"{name} (does not gate {tensor})")
    assert checked, "no shared-launching entries found -- re-derive this test"
    assert not ungated, (
        f"{tree.name}: these entries launch a shared kernel without checking "
        f"the shared intermediate size against the routed one: {ungated}. "
        "The kernel strides the shared weights by the routed size, so S != I "
        "silently computes a second gate projection (S > I) or reads past the "
        "tensor (S < I)"
    )


@pytest.mark.parametrize("tree", [_VLLM, _SGL], ids=["vllm", "sglang"])
def test_moe_shared_refusal_helper_compares_against_the_routed_size(tree):
    """The call sites are pinned above; this pins that the callee refuses.

    Centralising the check moved the substance out of the entries, so a helper
    that derived S and never compared it would leave every call site passing.
    """
    path = tree / "moe_batch/moe.sycl"
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = code(path.read_text())
    m = re.search(r"static void moe_check_shared_inter_matches\(", src)
    assert m, "the shared-intermediate refusal helper is gone"
    # Brace-match the body. Bounding at the next `static` ran ~8 functions past
    # the helper and counted their returns as its own -- the same too-wide
    # window that has produced a false reading in this suite before.
    open_brace = src.index("{", m.end())
    depth, j = 0, open_brace
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    assert depth == 0, "unbalanced braces in the refusal helper"
    body = src[m.start():j + 1]

    # Both arms, each against the routed size -- and 2*I for gate_up, since a
    # fused gate+up tensor is twice as wide. Comparing the raw numel to I would
    # refuse every correct model instead.
    assert re.search(r"two_s\s*==\s*2 \* intermediate_size", body), (
        "the gate_up arm no longer compares the derived shared size against "
        "twice the routed intermediate size"
    )
    assert re.search(r"\bs\s*==\s*intermediate_size", body), (
        "the down arm no longer compares the derived shared size against the "
        "routed intermediate size"
    )
    assert body.count("TORCH_CHECK") >= 2, (
        f"only {body.count('TORCH_CHECK')} TORCH_CHECK in the refusal helper; "
        "both arms must be able to refuse"
    )
    # A helper that returns before checking refuses nothing. Exactly one
    # early-out is legitimate -- the num_shared_experts<=0 case, where there is
    # no shared tensor to measure -- and it must be that one, guarded on that
    # variable, not an unconditional return that skips both TORCH_CHECKs.
    rets = re.findall(r"\breturn\s*;", body)
    assert len(rets) == 1, (
        f"{len(rets)} bare returns in the refusal helper; an early return past "
        "the TORCH_CHECKs makes the refusal vacuous"
    )
    assert re.search(r"if \(num_shared_experts <= 0\) return\s*;", body), (
        "the single early-out is no longer the num_shared_experts<=0 case; a "
        "return on any other condition skips the refusal for real shapes"
    )


def test_gdn_conv_entry_refuses_a_tap_count_it_cannot_serve():
    """The layout gate is not a tap-count gate; the kernel needs both.

    `conv_state.size(-1) <= 8` correctly distinguishes the native pool layout
    from the transposed legacy copy. It says nothing about W, and the kernel
    hardcodes it on both axes: 3 state taps (`block_load<fp16,192>` with
    `select<64,3>`, or three `k * dim` loads) and 4 weights
    (`block_load<fp16,256>` with `select<64,4>`). Without a tap check, W < 4
    reads past the conv row and W > 4 mis-strides the taps -- writing wrong
    values back into the live pool, with no throw for the caller's wrapper to
    catch. vllm refuses `conv_kernel_size != 4` upstream in Python; this tree
    has no Python gate, so the refusal has to live at the entry.
    """
    # BOTH trees: checking one leaves the other's wrapper free to export conv
    # entries with no TORCH_CHECK at all.
    checked, ungated = 0, []
    for tree in (_VLLM, _SGL):
        path = tree / "xpu/esimd_kernel_lgrf.sycl"
        if not path.exists():
            continue
        src = code(path.read_text())
        checked += _collect_ungated_conv_entries(src, tree.parents[1].name,
                                                 ungated)
    assert checked, "no conv entries found -- re-derive this test"
    assert not ungated, (
        "these conv entries launch a kernel hardwired to 3 taps / 4 weights "
        f"without refusing anything else: {ungated}"
    )


def _collect_ungated_conv_entries(src, tree_name, ungated):
    """Append each entry whose refusal is missing, vacuous or not a refusal."""
    checked = 0

    # PER ENTRY, brace-matched. Reading the whole file let one entry's gate
    # satisfy the assertion for every entry -- and it did: esimd_gdn_conv_fused
    # shipped with no gate at all while its _seq sibling's gate kept this test
    # green. The too-wide window again, in the test written to close the hole.
    # No `^` anchor: code() collapses newlines, so a line-start anchor matches
    # nothing here.
    for m in re.finditer(r"at::Tensor (esimd_gdn_conv_fused\w*)\(", src):
        name = f"{tree_name}::{m.group(1)}"
        ob = src.index("{", m.end())
        depth, j = 0, ob
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = src[m.start():j + 1]
        if "conv_state" not in body:
            continue
        checked += 1

        def refuses(pattern, body=body):
            """The comparison must be the SUBJECT of a live TORCH_CHECK.

            Matching the comparison's text anywhere in the body accepts
            `const bool taps_ok = (... == 3); (void)taps_ok;` -- the refusal
            deleted, the text preserved. Requiring a bare `TORCH_CHECK(`
            accepts `TORCH_CHECK(true || (... == 3), ...)` -- vacuously true.
            Both defeated the previous version of this test at 476/9 green.
            """
            for c in re.finditer(r"TORCH_CHECK\(", body):
                d, e = 0, c.end() - 1
                while e < len(body):
                    if body[e] == "(":
                        d += 1
                    elif body[e] == ")":
                        d -= 1
                        if d == 0:
                            break
                    e += 1
                cond = body[c.end():e].split(",")[0]
                if re.search(pattern, cond) and "||" not in cond:
                    return True
            return False

        # An entry that detects the layout must read the tap count from the
        # axis holding it in EACH layout: native is (cache, conv_dim, W-1) so
        # size(-1), transposed is (cache, W-1, conv_dim) so size(-2). One axis
        # for both admits the other layout's conv_dim as a tap count. An entry
        # with no detection is transposed-only, so -2 unconditionally.
        if "conv_native" in body:
            if not re.search(r"conv_native\s*\?\s*conv_state\.size\(-1\)\s*"
                             r":\s*conv_state\.size\(-2\)", body):
                ungated.append(f"{name} (tap axis not selected per layout)")
            elif not refuses(r"conv_taps == 3"):
                ungated.append(f"{name} (no tap-count refusal)")
        elif not refuses(r"conv_state\.size\(-2\) == 3"):
            ungated.append(f"{name} (no tap-count refusal)")

        if not refuses(r"conv_weight\.size\(-1\) == 4"):
            ungated.append(f"{name} (no conv_weight width refusal)")

        # Equality, not a bound: `<= 3` re-admits every short row the kernel
        # then overreads, the defect class this check exists to close.
        if re.search(r"(?:conv_taps|conv_state\.size\(-2\))\s*(?:<=|>=|<|>)\s*3",
                     body):
            ungated.append(f"{name} (tap check weakened to a magnitude test)")
    return checked


def test_state_pool_sentinel_is_negative_everywhere():
    """One pool, one sentinel. Slot 0 is real and writable.

    gdn_conv_fused_seq_spec.h used `> 0` / `<= 0` while every other kernel
    touching this pool used `>= 0`, so a legitimate slot-0 row had its conv
    history dropped, its state write-back discarded, and -- worst case -- zeros
    written to output and z_out with no throw for the caller to catch. The
    allocator states the convention outright (sglang_for_multi_arc.patch:
    "Slot 0 is a real, writable slot"), and the spec path's OWN host guard
    rejects `< 0` and accepts 0, so host and kernel disagreed inside one
    function.
    """
    files = []
    for tree in (_VLLM, _SGL):
        for rel in ("xpu/esimd_kernels/gdn_conv_fused.h",
                    "xpu/esimd_kernels/gdn_conv_fused_seq.h",
                    "xpu/esimd_kernels/gdn_conv_fused_seq_spec.h",
                    "eagle/eagle.sycl"):
            p = tree / rel
            if p.exists():
                files.append(p)
    assert files, "no state-pool kernels found -- re-derive this test"

    offenders = []
    for p in files:
        src = code(p.read_text())
        for m in re.finditer(
                r"\b(\w*(?:state_idx|_indices|conv_idx|ssm_idx))\s*"
                r"(>|<=)\s*0\b", src):
            # `> 0` treats slot 0 as null; `<= 0` refuses it. Both are the
            # sentinel-as-zero convention this pool does not use.
            offenders.append(f"{p.name}: {m.group(1)} {m.group(2)} 0")
    assert not offenders, (
        "these tests treat slot 0 as a null sentinel; the pool's sentinel is "
        f"-1 and slot 0 is a real writable slot: {offenders}"
    )


def _live_conditions(scope):
    """Conditions in `scope` that actually govern something.

    An `if (...)`/`while (...)` condition, or a `bool x = (...)` whose flag is
    read again. A `const bool ok = (i < rows); (void)ok;` mentions the right
    tokens and governs nothing -- that decoy defeated two earlier versions of
    the test below.
    """
    out = []
    for cm in re.finditer(r"\b(?:if|while)\s*\(", scope):
        d, e = 0, cm.end() - 1
        while e < len(scope):
            if scope[e] == "(":
                d += 1
            elif scope[e] == ")":
                d -= 1
                if d == 0:
                    break
            e += 1
        out.append(scope[cm.end():e])
    for fm in re.finditer(r"(?:const\s+)?bool\s+(\w+)\s*=\s*([^;]+);", scope):
        name, expr = fm.group(1), fm.group(2)
        for u in re.finditer(rf"\b{name}\b", scope):
            if u.start() <= fm.end():
                continue
            if scope[max(0, u.start() - 7):u.start()].rstrip().endswith("(void)"):
                continue
            out.append(expr)
            break
    return out


_FN_START = re.compile(
    r"SYCL_ESIMD_KERNEL|ESIMD_INLINE|inline void|static void|\bstruct\s+\w+")


def _enclosing_function(src, pos):
    """Source of the function containing pos.

    A bound only protects the pointer formed in its own function:
    gdn_conv_fused.h forms a conv_state pointer from `conv_idx` in TWO kernels,
    so a file-wide search lets one kernel's condition satisfy both.
    """
    starts = [m.start() for m in _FN_START.finditer(src) if m.start() <= pos]
    ends = [m.start() for m in _FN_START.finditer(src) if m.start() > pos]
    return src[max(starts) if starts else 0: min(ends) if ends else len(src)]


def test_state_indices_are_bounded_above_not_only_below():
    """A slot id from device memory needs both ends, at the pool it addresses.

    Every guard in these kernels was lower-bounded and none was upper-bounded:
    21 sites, five kernel families, both trees. The lower bound is live by
    design (-1 marks a graph-replay padding row); the upper bound was left to a
    Python check that vllm_for_multi_arc.patch:10589 runs only when
    `state_indices.device.type == "cpu"`, i.e. never on a real XPU decode. It is
    skipped there to avoid a per-layer device sync -- which is why the bound
    belongs in the kernel, where it costs one compare and no sync. sglang's
    kv_scatter.h:45 already did it this way.

    ANCHORED ON THE POINTER FORMATION. Keying the sweep on index NAMES meant a
    rename deleted a site from it entirely -- not reported unbounded, just no
    longer counted, with the floor absorbing the loss. `pool + (int64_t)idx *
    stride` cannot be renamed away without changing the arithmetic, and it
    names the pool, the index and the stride together, so the bound can be
    checked against the right pool and in the right function.
    """
    POOL_ROWS = {"conv_state": ("conv_rows", "cs_rows"),
                 "ssm_state": ("ssm_rows",),
                 "inter_conv": ("ic_rows",),
                 "inter_ssm": ("issm_rows", "inter_rows"),
                 "inter": ("inter_rows", "issm_rows")}
    FORMATION = re.compile(
        r"(\w*(?:conv_state|ssm_state|inter_conv|inter_ssm|inter)\w*)\s*\+\s*"
        r"\(int64_t\)\s*([\w\[\]]+)\s*\*\s*(\w*stride0)")

    # A pointer formation may HOIST the product into a named base one loop
    # level up: `int64_t inter_base = idx * stride;` then `inter + inter_base`.
    # Neither half matches FORMATION -- the first has no pool, the second no
    # stride -- so the write is invisible unless the hoisted form is matched
    # too.
    HOISTED = re.compile(
        r"(?:int64_t|int)\s+(\w+)\s*=\s*[^;]*?\(int64_t\)\s*([\w\[\]]+)"
        r"\s*\*\s*(\w*stride0)")

    sites, bad = 0, []
    for tree in (_VLLM, _SGL):
        for rel in ("xpu/esimd_kernels/gdn_conv_fused.h",
                    "xpu/esimd_kernels/gdn_conv_fused_seq.h",
                    "xpu/esimd_kernels/gdn_conv_fused_seq_spec.h",
                    "eagle/eagle.sycl"):
            path = tree / rel
            if not path.exists():
                continue
            src = code(path.read_text())
            found_here = [(m.start(), m.group(1), m.group(2))
                          for m in FORMATION.finditer(src)]
            for hm in HOISTED.finditer(src):
                base, idx, stride = hm.group(1), hm.group(2), hm.group(3)
                # Only count it if the base is actually added to a pool.
                pool_use = re.search(
                    r"(\w*(?:conv_state|ssm_state|inter_conv|inter_ssm|inter)\w*)"
                    rf"\s*\+\s*{re.escape(base)}\b", src)
                if pool_use:
                    found_here.append((hm.start(), pool_use.group(1), idx))
            for pos, pool, idx in found_here:
                sites += 1
                where = (f"{tree.parents[1].name}/{path.name}:"
                         f"{src[:pos].count(chr(10)) + 1}")
                key = next((k for k in POOL_ROWS
                            if pool.startswith(k) or pool == k), None)
                allowed = set(POOL_ROWS.get(key, ()))
                bare = re.escape(idx)
                found = set()
                for cond in _live_conditions(_enclosing_function(src, pos)):
                    for cm in re.finditer(
                            rf"{bare}\s*(?:<|>=)\s*(\w+_rows)\b", cond):
                        found.add(cm.group(1))
                if not found:
                    bad.append(f"{where}: {idx} strides {pool} with no row "
                               "bound in a live condition of its own function")
                elif not found & allowed:
                    bad.append(f"{where}: {idx} strides {pool} but is bounded "
                               f"by {sorted(found)}, not {sorted(allowed)}")

    assert sites >= 28, (
        f"only {sites} pool pointer formations found -- the sweep is wrong, "
        "not the source"
    )
    assert not bad, (
        "these state indices address a pool without being bounded against that "
        f"pool's row count, in the function that forms the pointer: {bad}"
    )


def test_every_eagle_kernel_guards_the_sentinel_not_just_one():
    """Both kernels in the file, not the one the fix was written for.

    eagle.kernels.{fp16,bf16}.h holds two kernels that read the SAME two index
    tensors from the SAME host call. The sentinel fix reached gdnRecur only;
    causalConv1dUpdate kept computing offsetCs from an unchecked index and
    STORING through it. Indices arrive through a uint32_t*, so -1 does not
    compare less than zero -- the sign test has to be explicit, and a bare
    `< 1` is a lower bound on the subscript, not on the value.
    """
    ungated = []
    for dt in ("fp16", "bf16"):
        path = _SGL / f"eagle/eagle.kernels.{dt}.h"
        if not path.exists():
            continue
        src = code(path.read_text())
        for m in re.finditer(r"ESIMD_INLINE void (\w+)\(", src):
            name = m.group(1)
            ob = src.index("{", m.end())
            depth, j = 0, ob
            while j < len(src):
                if src[j] == "{":
                    depth += 1
                elif src[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            body = src[m.start():j + 1]
            # Only kernels that actually index a state pool with these tensors.
            if not re.search(r"(?:convStateIdx|ssmStateIdx|acceptedTokens)\[",
                             body):
                continue
            # The sign test must be explicit: an unsigned compare against 0 or 1
            # is what let the sentinel through in the first place.
            if not re.search(r"\(int32_t\)\s*\w+", body):
                ungated.append(f"{dt}::{name} (no signed sentinel test)")
    assert not ungated, (
        "these eagle kernels read a -1-sentinel index through a uint32_t* and "
        f"never test it signed: {ungated}"
    )


def test_softmax_reciprocal_is_guarded_in_every_phase():
    """Phase3 guarded it and explained why; Phase2 did the same divide bare.

    An empty batch contributes no exp() terms, the softmax sum stays 0, and
    1/0 makes every output lane NaN -- which is then STORED to out. Phase3
    carried that guard with a comment; Phase2 did not. And Phase3 only runs
    when maxKvSeqLen > 1024 (`flag = maxKvSeqLen > 1024 ? 0 : 1`, then
    `if (flag == 0) submit(Phase3)`), so at or below 1024 the unguarded one was
    the only reciprocal on the path: a guard on the wrong side of a dispatcher.
    """
    unguarded = []
    for tree in (_VLLM, _SGL):
        for name in ("page.attn.h", "page.attn.fp8.h", "page.attn.gqa2.h"):
            path = tree / "eagle" / name
            if not path.exists():
                continue
            src = path.read_text()
            for m in re.finditer(r"1\.0f\s*/\s*(softmaxMul\w*)", src):
                ln = src[:m.start()].count("\n") + 1
                line = src.split("\n")[ln - 1]
                # Either the scalar ternary or the simd merge, both of which
                # substitute 0 for an empty batch rather than dividing.
                if "> 0.0f" not in line and "merge(" not in line:
                    unguarded.append(
                        f"{tree.parents[1].name}/{name}:{ln}")
    assert not unguarded, (
        "these softmax reciprocals divide by a sum that is 0 for an empty "
        f"batch, storing NaN to out: {unguarded}"
    )


def test_moe_scatter_bounds_the_expert_id():
    """expert_id indexes a [num_experts] array with nothing cross-checking it.

    It arrives from a caller tensor at the exported op, with num_experts passed
    as a separate scalar. Cast to uint32_t for the byte offset it WRAPS, so a
    negative lands ~4 GiB out -- the atomic_add in Init writes there, and in
    Copy the same id feeds `dp`, a write index into scattered_hidden and
    scattered_weights.

    The guard is deliberately NOT the single-term `if (x >= y) return;` shape:
    test_perf_invariants.py counts that exact form against the global-id reads
    with a margin of exactly zero, so widening an existing guard turns the suite
    red. A disjunction with `continue` adds no read and removes no guard -- and
    `continue` rather than `return` is also the correct semantics, since a
    return abandons the token's remaining top-k slots.
    """
    checked = 0
    for tree in (_VLLM, _SGL):
        path = tree / "xpu/esimd_kernels/moe_ops.h"
        if not path.exists():
            continue
        src = code(path.read_text())
        for kernel in ("MoE_Scatter_Init_Kernel", "MoE_Scatter_Copy_Kernel"):
            checked += 1
            i = src.find(f"struct {kernel}")
            assert i >= 0, f"{tree.parents[1].name}: {kernel} is gone"
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
            assert re.search(
                r"expert_id < 0 \|\| expert_id >= num_experts", body), (
                f"{tree.parents[1].name}/{kernel}: expert_id is not bounded "
                "against num_experts; a negative wraps to ~4 GiB out"
            )
            assert "continue;" in body, (
                f"{tree.parents[1].name}/{kernel}: the skip must be `continue`, "
                "not `return` -- a return abandons the remaining top-k slots"
            )
            # Whoever WRITES a sentinel owes every reader a test for it. The
            # scatter's -1 marker went into topk_ids, and MoE_Gather_Kernel --
            # its only consumer -- had no reason to test for it, because the
            # CPU producer it was written against never emitted one. Unchecked,
            # `(size_t)(-1)` is SIZE_MAX, so the read lands at an enormous
            # offset rather than just before the buffer.
            if kernel == "MoE_Scatter_Copy_Kernel":
                g = src.find("struct MoE_Gather_Kernel")
                assert g >= 0, f"{tree.parents[1].name}: the gather kernel is gone"
                gob = src.index("{", g)
                gd, gj = 0, gob
                while gj < len(src):
                    if src[gj] == "{":
                        gd += 1
                    elif src[gj] == "}":
                        gd -= 1
                        if gd == 0:
                            break
                    gj += 1
                gbody = src[g:gj + 1]
                assert re.search(r"ids\[k\]\s*<\s*0", gbody), (
                    f"{tree.parents[1].name}: the scatter writes -1 into "
                    "topk_ids but the gather never tests for it"
                )

    # Both files gone would empty the loop and pass having checked nothing --
    # caught by test_meta_suite_integrity on this very test.
    assert checked >= 2, (
        f"examined {checked} scatter kernels; moe_ops.h is missing from both "
        "trees, so this test proved nothing"
    )
