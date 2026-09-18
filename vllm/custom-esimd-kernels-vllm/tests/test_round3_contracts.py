"""Static contracts for kernel guards.

Each assertion pins the guard's effect rather than its wording, so that a
rephrasing passes and a removal or a vacuous rewrite (`if (true)`, a
self-comparison) fails.
"""

import ast
import re
from pathlib import Path

import pytest

from srctext import assert_single_write, assert_live, code, schema_of, tokens

_ROOT = Path(__file__).resolve().parents[3]
_VLLM = Path(__file__).resolve().parents[1] / "csrc"
_SGL = _ROOT / "sglang/custom-esimd-kernels/csrc"

_GDN = [_VLLM / "xpu/esimd_kernels/gdn_conv_fused.h",
        _SGL / "xpu/esimd_kernels/gdn_conv_fused.h"]
_PAGE_ATTN = sorted(
    list((_VLLM / "eagle").glob("page.attn*.h"))
    + list((_SGL / "eagle").glob("page.attn*.h"))
)
_EAGLE = [_VLLM / "eagle/eagle.sycl", _SGL / "eagle/eagle.sycl"]
_KQUANT = [_SGL / "xpu/esimd_kernels/q5_k_GEMV.h",
           _SGL / "xpu/esimd_kernels/q6_k_GEMV.h"]


def _ids(p):
    return f"{p.parents[2].name}/{p.name}"


@pytest.mark.parametrize("path", _KQUANT, ids=lambda p: p.name)
def test_kquant_refuses_the_narrow_shuffle_tile(path):
    """qh is pre-shuffled against a fixed 512 tile; VL/8 only matches there."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    tag = "Q5_K_VL" if "q5" in path.name else "Q6_K_VL"
    assert re.search(rf"TORCH_CHECK\(\s*K % {tag} == 0,", c), (
        "a K that is not a whole number of shuffle tiles must be refused, "
        "not routed to a narrower tile that de-shuffles the wrong elements"
    )
    assert f"{tag} / 2" not in c and f"{tag}/2" not in tokens(path.read_text()), (
        "the narrow instantiation cannot consume the host's shuffle layout"
    )


@pytest.mark.parametrize("path", _GDN, ids=_ids)
def test_shift_kernel_retires_dead_threads(path):
    """The shift kernel is the only shift path, so its guard is load-bearing."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    i = c.find("conv_state_shift_kernel")
    assert i >= 0, "conv_state_shift_kernel not found"
    body = c[i : i + 2000]
    assert_live(body, "if (tid >= useful_threads) return;",
                "dead threads store past the conv_state slot into the next entry")


@pytest.mark.parametrize("path", _GDN, ids=_ids)
def test_conv_state_shift_has_no_inline_path(path):
    """The shift must not run inside the compute kernel, at any work-group count.

    Every one of the HV work-groups reads this sequence's conv_state in Phase 1,
    and an inline shift stores to it from hv == 0. A SYCL barrier orders only
    the work-group that executes it, so nothing stops that store from landing
    before another group's read of the same seq_idx.

    The removed form gated itself on `total_wgs <= WG_SIZE`, reasoning that a
    grid that fits one scheduling wave runs concurrently. Concurrent is not
    ordered: the race is between two work-groups that are both resident, which
    is exactly when they overlap. The condition selected for the hazard rather
    than against it. vllm's twin never had the parameter.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    assert "inline_conv_shift" not in c and "inline_shift" not in c, (
        "an inline conv_state shift path is back; hv == 0 stores to conv_state "
        "while the other HV work-groups are still reading it, and no "
        "work-group barrier can order that"
    )
    # The separate kernel must still be submitted, or nothing shifts at all and
    # the convolution window never advances.
    assert "conv_state_shift_kernel(" in c, (
        "the ordered shift kernel is gone; conv_state would never advance"
    )


_GDN_SEQ = [_VLLM / "xpu/esimd_kernels/gdn_conv_fused_seq.h",
            _SGL / "xpu/esimd_kernels/gdn_conv_fused_seq.h"]


# Both twins, not only the _seq pair: the non-seq file forms the same
# cstate_base from the same conv_idx and needs the same guard.
@pytest.mark.parametrize("path", _GDN_SEQ + _GDN, ids=_ids)
def test_every_conv_state_base_is_gated_not_returned(path):
    """conv_idx * stride reads before the allocation when conv_idx is -1.

    The guard must gate the conv1d loads, NOT return from the whole kernel.
    The compute kernels own the zero-fill for a padding row -- the `else` that
    writes 0 to output_ptr is why the caller need not call .zero_() -- so a
    blanket return skips the zero-fill and the z extraction and hands back a
    reused buffer holding the previous decode step's activations.

    The shift kernels are exempt and keep their bare return: they write only
    cache, so there is no output to abandon.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    bases = c.count("fp16* cstate_base = conv_state_ptr + (int64_t)conv_idx")
    assert bases, f"{path.name}: no conv_state base found -- parse is wrong"

    # Judge each function by whether it writes an output: a whole-file count
    # cannot separate the shift kernels from the compute kernels.
    parts = re.split(r"(?=ESIMD_INLINE void |inline void )", c)
    checked = 0
    for body in parts:
        if "cstate_base = conv_state_ptr" not in body:
            continue
        checked += 1
        writes_output = "output_ptr +" in body
        if writes_output:
            assert "if (conv_idx < 0) return;" not in body, (
                f"{path.name}: a kernel that writes output_ptr returns on a "
                "padding slot, abandoning the zero-fill and the z extraction. "
                "Gate the conv loads instead, as gdn_conv_fused.h:186 and "
                "gdn_conv_fused_seq_spec.h:199 do."
            )
            assert "conv_idx >= 0" in body, (
                f"{path.name}: a kernel that writes output_ptr forms "
                "cstate_base with no `conv_idx >= 0` gate; a padding slot "
                "reads a full stride before the allocation"
            )
            # The gate must guard the loads, not merely be declared: `if (true)`
            # keeps the declaration and re-arms the OOB read.
            sites = [m.start() for m in
                     re.finditer(r"block_load<fp16, (?:64|192)>\(cstate_base", body)]
            assert sites, f"{path.name}: no cstate_base load found"
            for pos in sites:
                # The nearest *enclosing* `if (...) {`, i.e. the last one whose
                # brace is still open here. A window of recent conditions is not
                # enough: the hi-chunk load sits after the lo-chunk gate closed.
                head = body[:pos]
                depth, chain = 0, []
                for m in reversed(list(re.finditer(
                        r"\}|if \(([^()]*(?:\([^()]*\)[^()]*)*)\) \{|\{", head))):
                    tok = m.group(0)
                    if tok == "}":
                        depth += 1
                    elif depth:
                        depth -= 1
                    elif m.group(1) is not None:
                        chain.append(m.group(1))
                assert chain, (
                    f"{path.name}: a cstate_base load has no enclosing if"
                )
                # Any link in the still-open chain may carry the slot test: a
                # nested branch (e.g. conv_native) inside an outer slot gate is
                # fine; what must not happen is no link testing it at all.
                # Accept either spelling -- the flag the _seq files use, or the
                # inline `conv_idx >= 0` of the non-seq twin.
                gates = [x for x in chain
                         if re.search(r"\bhave_conv_slot\b", x)
                         or re.search(r"\bconv_idx\s*>=\s*0\b", x)]
                assert gates, (
                    f"{path.name}: no enclosing gate of a cstate_base load "
                    f"tests the slot; open conditions were {chain}"
                )
                # ...and the gate must still exclude a padding row. A substring
                # test admits `have_conv_slot || true` and the self-comparison,
                # which keep the identifier while re-arming every load.
                for g in gates:
                    assert "||" not in g, (
                        f"{path.name}: the slot gate is disjoined and cannot "
                        f"exclude a padding row: {g!r}"
                    )
                    assert not re.search(r"\b(?:true|1)\b", g), (
                        f"{path.name}: the slot gate is widened by a constant: "
                        f"{g!r}"
                    )
                    # Self-comparison is vacuous. Flag spelling only: the inline
                    # form names conv_idx twice, once per bound.
                    assert "have_conv_slot" not in g \
                        or g.count("have_conv_slot") == 1, (
                        f"{path.name}: the slot gate compares the flag with "
                        f"itself: {g!r}"
                    )
            # The gate must not be pinned open.
            assert not re.search(r"have_conv_slot\s*=\s*(?:true|1)\s*;", body), (
                f"{path.name}: have_conv_slot is pinned true, re-arming the "
                "out-of-bounds read while keeping the gate's text"
            )
            # Accept a trailing conjunct so an added upper bound still passes,
            # while requiring the lower bound to derive from conv_idx. Flag
            # spelling only: the inline form is covered by the gate walk above.
            if "have_conv_slot" in body:
                assert re.search(
                    r"have_conv_slot\s*=\s*\(?conv_idx >= 0\b", body), (
                    f"{path.name}: have_conv_slot must be derived from conv_idx"
                )
            assert not re.search(r"if \(\s*true\s*\) \{", body), (
                f"{path.name}: a gate has been replaced by if (true)"
            )
        else:
            # Cache-only kernel: a bare return is correct and expected.
            # Substring, not exact: the guard is two-sided now
            # (`conv_idx < 0 || conv_idx >= conv_rows`) and an exact match
            # required it to end at the lower bound.
            assert re.search(r"if \(conv_idx < 0\b", body), (
                f"{path.name}: a cache-only kernel forms cstate_base without "
                "guarding conv_idx at all"
            )
    assert checked >= 2, (
        f"{path.name}: only {checked} functions matched -- the split is wrong, "
        "not the source"
    )


@pytest.mark.parametrize("path", _GDN, ids=_ids)
def test_conv_state_base_is_gated_on_a_real_slot(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    # The needle stops before the `)` so an added upper bound still matches:
    # a two-sided guard is what this pool needs, and pinning the closing paren
    # made the safer form fail.
    assert_live(c, "if (is_valid && conv_idx >= 0",
                "conv_idx is -1 for a padding slot; cstate_base then points "
                "before the allocation")


@pytest.mark.parametrize("path", _PAGE_ATTN, ids=_ids)
def test_partial_output_write_is_bounded_by_the_allocation(path):
    """pTempOut is strided by the host hint but indexed by the device seq_len."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    sites = len(re.findall(r"int kvSeqOutGroup = ", c))
    if sites == 0:
        pytest.skip("no phase-1 group index in this file")
    clamps = len(re.findall(
        r"if \(kvSeqOutGroup > allocatedGroups\) kvSeqOutGroup = allocatedGroups;", c
    ))
    assert clamps == sites, (
        f"{sites} phase-1 group indices but {clamps} clamps; every site must "
        "bound the write index, not only the phase-3 read"
    )
    assert "allocatedGroups = (int)((longestBatch + 1023) >> 10)" in c, (
        "the bound must come from the allocation, not a sentinel"
    )


@pytest.mark.parametrize("path", _EAGLE, ids=_ids)
def test_host_refuses_a_seq_len_past_the_hint(path):
    """Clamping alone would reduce over a prefix and return a confident answer."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    assert ".item<int64_t>()" not in c, (
        "a device-to-host sync breaks graph capture, and under capture "
        "max_seq_len is pinned to the page-table width so it cannot fire"
    )
    assert re.search(
        r"TORCH_CHECK\(\(int64_t\)block_table\.size\(1\) \* \(int64_t\)pageSize"
        r" >= max_seq_len,", c
    ), "the page table must be bounded host-side against max_seq_len"
    assert re.search(
        r"TORCH_CHECK\(seq_lens\.scalar_type\(\) == at::kInt,", c
    ), "seq_lens is read as uint32; an int64 tensor halves every other batch"


def test_q8_ba_writes_its_declared_output():
    """xn is declared Tensor(b!); passing nullptr returns it unwritten."""
    path = _SGL / "xpu/esimd_kernel.sycl"
    if not path.exists():
        pytest.skip("sglang tree not present")
    c = code(path.read_text())
    i = c.find("esimd_resadd_norm_gemv_q8_ba")
    assert i >= 0
    body = c[i : i + 2500]
    assert "(void)p_xn;" not in body, "the xn output is being discarded"
    assert "reinterpret_cast<fp16*>(p_xn)" in body, "xn must reach the kernel"


_LADDERS = [
    _VLLM / "xpu/esimd_kernels/scaled_resadd_norm_gemv_fp8.h",
    _VLLM / "xpu/esimd_kernels/accum_norm_add_norm.h",
    _VLLM / "xpu/esimd_kernels/norm_gemv_norm_fp16.h",
    _SGL / "xpu/esimd_kernels/norm_gemv_norm_fp16.h",
]


@pytest.mark.parametrize("path", _LADDERS, ids=lambda p: f"{p.parents[3].name}/{p.name}")
def test_vl64_arm_rejects_an_indivisible_k(path):
    """The terminal arm is VL=64 and every kernel walks K in whole chunks."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    assert re.search(r"TORCH_CHECK\(\s*(?:K|hidden_size) % 64 == 0,", c), (
        "the K <= 4096 disjunct admits any shape into the VL=64 arm, which "
        "drops the residue with no tail path"
    )


_NORM_ADD_NORM = [
    _VLLM / "xpu/torch_extension.cc",
    _SGL / "xpu/torch_extension.cc",
]


@pytest.mark.parametrize("path", _NORM_ADD_NORM, ids=lambda p: p.parents[2].name)
def test_norm_add_norm_declares_what_it_writes(path):
    """h1 and out are written in place; an unmarked slot licenses DCE."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    # Raw text, not code(): code() blanks string-literal bodies and a torch
    # schema IS a string literal, so the old c.find() returned -1 for both ops
    # and the `continue` skipped every assertion in this test.
    raw = path.read_text()
    seen = 0
    for op in ("esimd_norm_add_norm", "esimd_accum_norm_add_norm"):
        # Join adjacent literals before the presence test: splitting the op
        # name as "esimd_foo" "(Tensor ..." is legal C++ that compiles
        # identically, and an unjoined check skipped the op entirely -- every
        # alias marker could then be dropped with the suite green.
        if op + '(' not in re.sub(r'"\s*"', "", raw):
            continue
        schema = schema_of(raw, op)
        seen += 1
        assert "Tensor(a!) h1" in schema, f"h1 is written in place: {schema}"
        assert "Tensor(b!) out" in schema, f"out is written: {schema}"
    assert seen, f"{path}: neither op found -- the parse is wrong, not the source"


_SHARED_EXPERT = [
    _VLLM / "moe_batch/moe_int4.sycl",
    _SGL / "moe_batch/moe_int4.sycl",
]


@pytest.mark.parametrize("path", _SHARED_EXPERT, ids=lambda p: p.parents[2].name)
def test_shared_expert_accumulation_is_bounded(path):
    """The cross-expert accumulate is an unsynchronised read-modify-write."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    assert re.search(r"TORCH_CHECK\(num_shared_experts == 1,", c), (
        "a second expert can run before the sid == 0 seed and read the "
        "uninitialised allocation"
    )


@pytest.mark.parametrize("path", _GDN, ids=_ids)
def test_both_conv_state_reads_are_gated(path):
    """The lo and hi chunks share one cstate_base derived from conv_idx."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    gated = len(re.findall(r"is_valid && conv_idx >= 0", c))
    assert gated >= 2, (
        f"only {gated} of the two conv_state reads test conv_idx; both derive "
        "from the same base, which is negative for a padding slot"
    )
    # BOTH ends, per gate. The lo and hi chunks load from different offsets
    # (chunk_start vs chunk_start_hi), so the two gates are NOT redundant --
    # dropping either exposes its own loads. Count the upper bound too: the
    # lower-bound substring survives deletion of `&& conv_idx < conv_rows`.
    bounded = len(re.findall(r"conv_idx < conv_rows", c))
    assert bounded >= gated, (
        f"{gated} conv_idx gates but only {bounded} bound it above; a slot id "
        "past the pool indexes another allocation, and the two gates cover "
        "different chunk offsets so neither covers for the other"
    )


def test_kv_scatter_bounds_its_destination():
    """dst comes from device memory and indexes the KV cache directly."""
    hdr = _SGL / "xpu/esimd_kernels/kv_scatter.h"
    binding = _SGL / "xpu/esimd_kernel.sycl"
    if not hdr.exists():
        pytest.skip("sglang tree not present")
    k = code(hdr.read_text())
    assert "if (dst < 0 || dst >= cache_rows) return;" in k, (
        "a stale slot id scatters past the cache into unrelated memory"
    )
    b = code(binding.read_text())
    i = b.find("void esimd_kv_scatter")
    body = b[i : i + 2000]
    assert re.search(r"TORCH_CHECK\(indices\.scalar_type\(\) == at::kLong,", body), (
        "indices is reinterpreted as int64"
    )
    assert re.search(r"TORCH_CHECK\(row_dim % 32 == 0,", body), (
        "the VL=32 arm copies whole chunks"
    )


_NMAJOR_GEMM = [
    _VLLM / "moe_batch/moe_int4.sycl",
    _SGL / "moe_batch/moe_int4.sycl",
]


@pytest.mark.parametrize("path", _NMAJOR_GEMM, ids=lambda p: p.parents[2].name)
def test_nmajor_gemm_pins_the_group_size(path):
    """The kernel hardcodes BS=128; a smaller group overruns the weight row."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    i = c.find("torch::Tensor moe_gemm_int4_nmajor")
    assert i >= 0, "moe_gemm_int4_nmajor not found"
    body = c[i : i + 2500]
    assert re.search(r"TORCH_CHECK\(group_size == 128,", body), (
        "group_size is taken from the caller but the kernel hardcodes 128"
    )
    assert "scale.size(2) == K_groups" in body, "the scale table must match K"


def test_int4_resadd_stores_whole_chunks_only():
    """A full-width store at offset<K writes past both output buffers."""
    path = _VLLM / "xpu/esimd_kernels/resadd_norm_gemv_int4.h"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    assert "const int k_end = (K / VL) * VL;" in c, (
        "the loop must stop at the last whole chunk"
    )
    assert "offset < K;" not in c, "a full-width store overruns at offset < K"
    assert re.search(r"TORCH_CHECK\(K % 128 == 0,", c), (
        "the narrowest arm is VL=128 and there is no tail path"
    )


def test_batched_rms_norm_dispatches_vl():
    """A hardcoded VL=512 leaves the tail of every row stale."""
    path = _VLLM / "xpu/esimd_kernels/fused_add_rms_norm_batched.h"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    assert "constexpr int VL = 512;" not in c, "VL must follow K, not be pinned"
    assert "template<int VL>" in c, "the kernel must be VL-templated"
    assert re.search(r"TORCH_CHECK\(K % 64 == 0,", c)


def test_gemma4_buffer_cache_keys_every_dimension():
    """top_k, hidden and intermediate all size the cached allocations."""
    path = _SGL / "moe_batch/moe.sycl"
    if not path.exists():
        pytest.skip("sglang tree not present")
    c = code(path.read_text())
    assert "sg_cached_ntokens >= n_tokens && sg_cached_topk == top_k" in c, (
        "keying on n_tokens alone reuses an undersized buffer on a shrink"
    )
    assert "sg_cached_hidden == hidden_size" in c
    assert "sg_cached_inter == intermediate_size" in c


def test_rope_positions_dtype_is_checked():
    """positions is reinterpreted as uint32; int64 reads pairwise."""
    seen = 0
    for path in (_VLLM / "xpu/esimd_kernel.sycl", _SGL / "xpu/esimd_kernel.sycl"):
        if not path.exists():
            continue
        c = code(path.read_text())
        if "reinterpret_cast<uint32_t*>(positions.data_ptr())" not in c:
            continue
        seen += 1
        assert re.search(r"TORCH_CHECK\(positions\.scalar_type\(\) == at::kInt,", c), (
            f"{path.name}: an int64 positions tensor rotates every other token "
            "by the wrong angle"
        )
    # Without this, renaming the cast in BOTH trees empties the loop and the
    # test passes having checked nothing. Verified: it does.
    assert seen, ("the positions cast was not found in either tree; the anchor "
                  "moved and this test stopped checking anything")


_FP8_GEMV = [_VLLM / "xpu/esimd_kernels/fp8_GEMV_v2.h",
             _SGL / "xpu/esimd_kernels/fp8_GEMV_v2.h"]


@pytest.mark.parametrize("path", _FP8_GEMV, ids=_ids)
def test_fp8_gemv_floor_matches_its_ladder(path):
    """Both trees: a catch-all that launches a fixed width discards the
    selection and reads past the row."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    # Compare the floor against the arms that exist, rather than pinning the
    # floor's text. The literal form asserted the buggy state as the contract:
    # a floor of 128 with no 64/32 arms aborted K=576/1056/320/192 (gemma-3-4b),
    # and this test passed throughout.
    m = re.search(r"while \(vl > kpt \|\| kpt % vl != 0\) \{ if \(vl > (\d+)\)", c)
    assert m, "the walk-down loop is no longer recognisable"
    floor = int(m.group(1))
    for macro in sorted(set(re.findall(r"\{ (LAUNCH\w*)\(\d+, \d+\)", c))):
        widths = {int(w) for w, _ in re.findall(
            rf"\{{ {macro}\((\d+), (\d+)\)", c)}
        assert min(widths) == floor, (
            f"{macro}: the walk stops at vl={floor} but the narrowest arm is "
            f"{min(widths)}. Above it, splits needing a narrower arm abort at "
            f"the catch-all (this is how K=576/1056/320/192 broke); below it, "
            f"an emitted split has no arm at all."
        )
    assert re.search(r"TORCH_CHECK\(kpt % vl == 0,", c), (
        "an unsupported split must be refused"
    )
    assert not re.search(r"else \{ LAUNCH\w*\((?:128|32), 1\) \}", c), (
        "a catch-all arm substitutes a fixed width for the selected one"
    )
    # Per ladder, not per file. A `>= 3` count over the whole file is held up by
    # the other dispatchers, so one ladder's refusing arm could be deleted --
    # leaving an unhandled (vl,ks) to launch NOTHING and hand back whatever the
    # caller's torch.empty held -- with the count still satisfied.
    for host in re.findall(r"inline void (GEMV_fp8_\w*host)\(", c):
        i = c.index(host + "(")
        nxt = min((j for j in (c.find("inline void ", i + 1), len(c)) if j > 0))
        body = c[i:nxt]
        if "LAUNCH" not in body:
            continue
        assert re.search(r"else \{ TORCH_CHECK\(false,", body), (
            f"{host} has dispatch arms but no refusing terminal arm: an "
            "unhandled (vl, ks) launches nothing and the output keeps its "
            "uninitialised contents"
        )


def _select_vl_ks(n, k, floor, vl0=512):
    """LITERAL transcription of select_vl_ks. Both copies share this body; they
    differ ONLY in the walk-down floor, so it is a parameter.

      fp8_GEMV_v2.h:96   floor 128, followed by TORCH_CHECK(kpt % vl == 0)
      fp8_GEMM_pert.h:76 floor  32, NO TORCH_CHECK

    The previous version of this helper was written from memory rather than
    copied. It dropped the `vl = 128` assignment inside the ks branches and
    invented an `n <= 256` / `n <= 512 && k >= 1024` ladder that does not exist,
    so it reported 12 reachable pairs where the real GEMV selector reaches 6. Two
    commits reasoned from it: one added three arms that are unreachable, and one
    deleted four arms from the GEMM file -- whose floor is 32, not 128 -- that
    were reached by 1,630,208 (N,K) pairs. THE FLOOR IS A PER-FILE FACT.
    """
    vl, ks = vl0, 1
    if k < 512:
        vl, ks = 128, 1
    elif k == 512:
        vl, ks = 256, 1
    if n <= 128 and k >= 2048:
        vl, ks = 128, 8
    elif n <= 512 and k >= 2048:
        vl, ks = 128, 4
    kpt = k // ks
    while vl > kpt or kpt % vl != 0:
        if vl > floor:
            vl //= 2
        elif ks > 1:
            ks //= 2
            kpt = k // ks
        else:
            break
    return vl, ks, kpt


def _assert_transcription_matches(path, floor, fn="select_vl_ks"):
    """Guard the guard: pin the source lines the helper mirrors.

    A transcription that drifts from its source is worse than no test, because
    every coverage claim routed through it inherits the error silently. This
    guard caught a real divergence on its first run: there are THREE distinct
    select_vl_ks copies in this repo, not one.

      vllm fp8_GEMV_v2.h:96     vl = 512, floor  32, + TORCH_CHECK
      vllm fp8_GEMM_pert.h:76   vl = 512, floor  32, no check
      sglang fp8_GEMM_pert.h:77 vl = SGLANG_GEMV_VL_CAP (default 256!),
                                floor 32, plus a ks_mode == 1 && K == 5376 arm

    So the shared clauses are pinned for every copy, and the initial vl and the
    floor are pinned per copy. Nothing here may be generalised across files.
    """
    c = code(path.read_text())
    i = c.index("inline void " + fn + "(")
    body = c[i:c.index("}", c.index("while (vl > kpt", i))]
    shared = ("if (K < 512) { vl = 128; ks = 1; }",
              "else if (K == 512) { vl = 256; ks = 1; }",
              "if (N <= 128 && K >= 2048) { vl = 128; ks = 8; }",
              "else if (N <= 512 && K >= 2048) { vl = 128; ks = 4; }",
              f"if (vl > {floor}) {{")
    for needle in shared:
        assert needle in body, (
            f"{path.name}: the helper's transcription no longer matches the "
            f"source; missing {needle!r}"
        )
    # The initial vl differs per copy, and sglang's is an env-var cap whose
    # default only matters when K >= 512 and the ks branches do not fire.
    assert ("vl = 512; ks = 1;" in body) or ("vl = default_vl; ks = 1;" in body), (
        f"{path.name}: the initial (vl, ks) is neither 512 nor the env cap"
    )
    if "vl = default_vl" in body:
        # code() blanks string-literal bodies, so the env var NAME is not
        # visible here -- match the getenv call shape instead.
        assert 'std::getenv("")' in body, (
            f"{path.name}: default_vl is no longer read from the environment"
        )
        # The cap is now clamped to the ladder's arm widths; the default is
        # still 256, reached either by an absent env var or by the reject arm.
        assert "if (!value) return 256;" in body, (
            f"{path.name}: the env cap's default changed; the enumeration below "
            "assumes 256"
        )


def _pert_bmg_redirect(k):
    """Transcription of the redirect at the head of GEMV_fp8_pert_host."""
    return (k % 128 != 0
            or (k < 512 and k % 256 != 0)
            or (1024 <= k < 2048 and k % 256 != 0))


_LADDERS = [(t, m, nxt) for t in (_VLLM, _SGL)
            for m, nxt in (("LAUNCH", "GEMV_fp8_pern_fused_host"),
                           ("LAUNCH_FUSED", "select_vl_ks"),
                           ("LAUNCH_PERT", "GEMV_fp8_pert_fused_host"),
                           ("LAUNCH_PERT_FUSED", None))]


@pytest.mark.parametrize("tree,macro,nxt", _LADDERS,
                         ids=[f"{'vllm' if t is _VLLM else 'sglang'}/{m}"
                              for t, m, _ in _LADDERS])
def test_every_ladder_covers_every_pair_its_selector_emits(tree, macro, nxt):
    """All EIGHT ladders, not just vllm's pert one.

    Hardcoding one tree's host lets an arm be deleted from the other's ladder
    with the suite green, sending every (N,K) that needed it to a refusing arm.
    """
    path = tree / "xpu/esimd_kernels/fp8_GEMV_v2.h"
    if not path.exists():
        pytest.skip(f"{path} not present")
    _assert_transcription_matches(path, 32)
    c = code(path.read_text())
    arms = {(int(v), int(sp))
            for v, sp in re.findall(rf"{macro}\((\d+), *(\d+)\)", c)}
    assert arms, f"no {macro} arms found -- the parse is wrong, not the ladder"

    uncovered = set()
    for n in range(1, 4097):
        for k in range(64, 8193, 64):
            if _pert_bmg_redirect(k):
                continue
            vl, ks, kpt = _select_vl_ks(n, k, 32)
            if kpt % vl:
                continue
            if (vl, ks) not in arms:
                uncovered.add((vl, ks, n, k))
    assert not uncovered, (
        f"{path.parents[3].name}/{macro}: the selector emits pairs this ladder "
        f"cannot launch: {sorted(uncovered)[:4]}"
    )


def test_vllm_pert_ladder_covers_every_pair_its_selector_emits():
    """The selector's image must be a subset of the ladder's arms.

    Three arms were missing at once -- (512,4), (512,8), (256,8) -- each
    reachable at an ordinary decode shape with kpt % vl == 0. Sampling N found
    only one of the three: (512,4) needs 128 < N <= 256, a band that a sampled
    N list steps straight over. Enumerate, do not sample.
    """
    path = _VLLM / "xpu/esimd_kernels/fp8_GEMV_v2.h"
    if not path.exists():
        pytest.skip(f"{path} not present")
    # Pin the transcription: without it, moving the selector's N threshold
    # leaves this test verifying coverage against a selector that no longer
    # exists, and it goes on passing.
    _assert_transcription_matches(path, 32)
    c = code(path.read_text())

    body = c[c.index("GEMV_fp8_pert_host"):]
    body = body[:body.index("GEMV_fp8_pert_fused_host")]
    arms = {(int(v), int(s))
            for v, s in re.findall(r"LAUNCH_PERT\((\d+), *(\d+)\)", body)}
    assert arms, "no LAUNCH_PERT arms found -- the parse is wrong, not the ladder"

    emitted, uncovered = set(), set()
    for n in range(1, 4097):
        for k in range(64, 8193, 64):
            if _pert_bmg_redirect(k):
                continue
            vl, ks, kpt = _select_vl_ks(n, k, 32)
            if kpt % vl:
                continue          # refused loudly by the TORCH_CHECK
            emitted.add((vl, ks))
            if (vl, ks) not in arms:
                uncovered.add((vl, ks, n, k))
    assert not uncovered, (
        "the selector emits pairs the ladder cannot launch; the catch-all would "
        f"substitute a fixed width and read past the row: {sorted(uncovered)[:4]}"
    )
    assert emitted <= arms
    # The simulated floor must be the SOURCE's floor. This line passed 128 while
    # _assert_transcription_matches above pinned the source at 32 -- the two
    # contradicted each other, so the test validated a selector the code does
    # not have and missed three arms missing from LAUNCH_PERT.
    # Derived from the same walk, not hardcoded: a literal set here is what let
    # the floor-128 image be asserted as the contract after the source moved to
    # 32. What matters is that the ladder covers whatever the selector emits,
    # which `uncovered` already checks; this pins that the image is non-trivial.
    assert emitted and emitted <= arms, (
        f"the GEMV selector's image changed: {sorted(emitted)}"
    )


def test_pert_redirect_catches_what_the_floor_orphans():
    """K = 64*odd must reach bmg, not the TORCH_CHECK.

    Raising the walk-down floor to 128 orphaned this ladder's 64/32 arms. 1392
    (N,K) pairs that computed correct answers before the floor change now
    depend on this redirect; narrowing it to K % 64 turns them into throws.
    """
    path = _VLLM / "xpu/esimd_kernels/fp8_GEMV_v2.h"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    head = c[c.index("GEMV_fp8_pert_host"):]
    head = head[:head.index("select_vl_ks(N, K, vl, ks)")]
    assert re.search(r"if \(K % 128 != 0 \|\| \(K < 512 && K % 256 != 0\)\)", head), (
        "the pert redirect must test K % 128; at K % 64 every K = 64*odd "
        "(576, 704, 832, 960, ...) falls through to a hard throw"
    )
    for k in (576, 704, 832, 960, 2112, 3008):
        assert _pert_bmg_redirect(k), f"K={k} must route to the masked-tail kernel"


def test_sglang_fp8_gemv_refuses_what_it_cannot_dispatch():
    """The ladders have no arm below VL=128, and a catch-all that launches
    VL=128 anyway reads past the row for whatever the walk-down chose."""
    path = _SGL / "xpu/esimd_kernels/fp8_GEMV_v2.h"
    if not path.exists():
        pytest.skip("sglang tree not present")
    c = code(path.read_text())
    assert re.search(r"TORCH_CHECK\(kpt % vl == 0,", c), (
        "an unsupported split must be refused, not silently re-dispatched"
    )
    assert not re.search(r"else \{ LAUNCH\w*\(128, 1\) \}", c), (
        "a catch-all arm substitutes VL=128 for the selected width"
    )
    assert len(re.findall(r"else \{ TORCH_CHECK\(false,", c)) >= 3, (
        "every dispatch ladder needs a refusing terminal arm"
    )
    # The floor must match where the ladder arms stop; that comparison lives in
    # test_fp8_gemv_floor_matches_its_ladder rather than a text pin here.


def test_verify_family_bounds_its_device_indices():
    """inter_indices carries triton's -1 sentinel; predicts is indexed by it."""
    path = _SGL / "eagle/eagle.sycl"
    if not path.exists():
        pytest.skip("sglang tree not present")
    c = code(path.read_text())
    # The guard must scope only the cache write: returning abandons out and
    # zout. GdnFusedConvWindowFn writes only the cache, so returning is fine
    # there; every other site must keep producing out.
    # Match up to the lower bound: the guard is two-sided now
    # (`inter_indices[n] < 0 || inter_indices[n] >= ic_rows`), so an exact
    # string ending at `;` no longer matches the safer form.
    assert len(re.findall(r"if \(inter_indices\[n\] < 0\b", c)) == 1, (
        "returning leaves the row's out unwritten in a multi-output kernel"
    )
    assert not re.search(r"if \(inter_indices\[batch_id\] < 0\b", c)
    # The flag must be derived from the index, not pinned true. Match up to the
    # lower bound rather than to the `;`: an exact-string count required the
    # statement to TERMINATE at `>= 0`, so adding the upper bound the pool
    # needs -- strictly safer -- would have failed this test.
    assert len(re.findall(
        r"const bool has_inter = inter_indices\[batch_id\] >= 0\b", c)) >= 2, (
        "has_inter must test the device index"
    )
    assert len(re.findall(
        r"const bool has_inter = inter_indices\[n\] >= 0\b", c)) >= 1
    assert "has_inter = true" not in c, "the flag has been made vacuous"
    assert c.count("if (has_inter && step < cache_steps)") >= 2
    assert re.search(r"if \(last_accepted_retrive_idx < 0 \|\|\s*last_accepted_retrive_idx >= predicts_numel\) \{", c), (
        "retrive_index is device data and indexes predicts directly"
    )
    assert "int inter_stride0 = (int)intermediate.stride(0);" in c, (
        "the row stride must come from the tensor, not be synthesised"
    )


def test_verify_family_checks_index_dtypes():
    """cache_indices and friends are cast to const int*."""
    path = _SGL / "eagle/eagle.sycl"
    if not path.exists():
        pytest.skip("sglang tree not present")
    c = code(path.read_text())
    assert c.count("cache_indices.scalar_type() == at::kInt") >= 2, (
        "an int64 index tensor reads pairwise"
    )
    assert "query_start_loc.scalar_type() == at::kInt" in c
    assert re.search(r"TORCH_CHECK\(depth > 0,", c), (
        "depth forms a row stride that goes negative at zero"
    )


def test_splitk_matches_its_sibling_guards():
    """splitk_decode_attention sits beside a far better-guarded op."""
    path = _SGL / "eagle/eagle.sycl"
    if not path.exists():
        pytest.skip("sglang tree not present")
    c = code(path.read_text())
    i = c.find("void splitk_decode_attention")
    body = c[i : i + 4000]
    assert re.search(r"page_size > 0 && \(page_size & \(page_size - 1\)\) == 0,", body), (
        "the kernel masks with page_size - 1"
    )
    assert re.search(r"TORCH_CHECK\(num_q_heads % num_kv_heads == 0,", body)
    assert re.search(r"TORCH_CHECK\(num_splits > 0,", body)
    assert re.search(
        r"TORCH_CHECK\(\(int64_t\)page_table_stride \* \(int64_t\)page_size"
        r" >= max_seq_len,", body
    )


def test_kv_indices_scatter_is_bounded():
    """kv_indptr is device-side and offsets the destination."""
    path = _SGL / "xpu/esimd_kernel.sycl"
    if not path.exists():
        pytest.skip("sglang tree not present")
    c = code(path.read_text())
    assert re.search(r"dst < 0 \|\| dst >= kv_indices_numel\)", c), (
        "a stale kv_indptr scatters past the end of kv_indices"
    )
    assert re.search(r"req < 0 \|\| req >= req_rows\)", c)


_SCHEMA_PARITY = [
    ("esimd_qkv_split_norm_rope", "xpu/torch_extension.cc",
     ["Tensor(a!) q_out", "Tensor(b!) gate_out", "Tensor(c!) k_out",
      "Tensor(d!) v_out"]),
    ("esimd_gdn_conv_fused", "xpu/torch_extension_lgrf.cc",
     ["Tensor(a!) conv_state", "Tensor(b!) ssm_state", "Tensor(c!) output",
      "Tensor(d!) z_out"]),
    ("esimd_moe_gemm_fp8_blockscale", "xpu/torch_extension_moe.cc",
     ["Tensor(a!) output"]),
]


@pytest.mark.parametrize("op,rel,marks", _SCHEMA_PARITY, ids=lambda v: v if isinstance(v, str) else "")
def test_mutable_slots_are_declared_in_both_trees(op, rel, marks):
    """An unmarked output licenses reordering or dead-store elimination.

    Read raw text via schema_of(), not `code()`: a torch schema IS a string
    literal, and `code()` blanks literal bodies, so every lookup misses and the
    test passes vacuously. Never skip on a miss.
    """
    seen = 0
    for root in (_VLLM, _SGL):
        path = root / rel
        if not path.exists():
            continue
        raw = path.read_text()
        # Join adjacent literals before the presence test: splitting the op
        # name as "esimd_foo" "(Tensor ..." is legal C++ that compiles
        # identically, and an unjoined check skipped the op entirely -- every
        # alias marker could then be dropped with the suite green.
        if op + '(' not in re.sub(r'"\s*"', "", raw):
            continue
        schema = schema_of(raw, op)
        seen += 1
        for mark in marks:
            assert mark in schema, (
                f"{root.parents[0].name}/{rel}: {op} writes this tensor but "
                f"does not declare {mark}: {schema}"
            )
    assert seen, (
        f"{op} found in neither tree -- this test is not running, and a miss "
        "here must fail rather than skip."
    )


def test_an_aliased_return_is_declared_aliased():
    """A schema returning a `(a!)` argument must say so.

    Nothing in either suite checked a return type, which is how sglang's rope
    kept `-> Tensor` while returning q_out, declared `Tensor(a!)`. vllm gets it
    right by discarding the return in a void lambda and declaring `-> ()`,
    because auto-functionalize v1 rejects `-> Tensor(a!)` outright.
    """
    seen = 0
    for root in (_VLLM, _SGL):
        path = root / "xpu/torch_extension.cc"
        if not path.exists():
            continue
        raw = path.read_text()
        if '"esimd_qkv_split_norm_rope(' not in raw:
            continue
        seen += 1
        schema = schema_of(raw, "esimd_qkv_split_norm_rope")
        assert "Tensor(a!) q_out" in schema, schema
        tail = schema[schema.rindex("->"):]
        assert "()" in tail or "Tensor(a!)" in tail, (
            f"{root.parents[0].name}: rope returns q_out, which is Tensor(a!), "
            f"but declares an unaliased return: {tail}"
        )
    assert seen, (
        "the rope schema was not found in either tree; the anchor moved and this test stopped checking returns")


_NO_REGISTER_CACHE = [
    _VLLM / "xpu/esimd_kernels/norm_gemv_norm_fp16.h",
    _SGL / "xpu/esimd_kernels/norm_gemv_norm_fp16.h",
    _VLLM / "xpu/esimd_kernels/accum_norm_add_norm.h",
    _SGL / "xpu/esimd_kernels/norm_add_norm.h",
]


@pytest.mark.parametrize("path", _NO_REGISTER_CACHE, ids=lambda p: f"{p.parents[3].name}/{p.name}")
def test_no_simd_register_cache_array(path):
    """norm_add_norm.h records a Level Zero resource exhaustion caused by a
    simd<float, VL>[MAX_CHUNKS] array: 16 KB per thread against an 8 KB
    budget. These kernels stream instead.

    Superseded by test_every_simd_local_array_fits_the_grf_budget, which sweeps
    every header and computes the footprint. This one keeps its four-file list
    on purpose: widening it is a FALSE POSITIVE, because resadd_norm_gemv_int4.h
    holds a legitimate 8 KB res_chunks[4] that matches the same regex. A
    hardcoded list beside a sweep is how the 32 KB cache went unnoticed, so the
    sweep is the load-bearing test and this is a narrow backstop.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    t = tokens(path.read_text())
    assert not re.search(r"simd<float,VL>\w+\[MAX_CHUNKS\]", t), (
        "a per-thread register cache of VL-wide chunks exceeds the GRF budget"
    )


# The 8 KB per-thread GRF budget without -doubleGRF. norm_add_norm.h:23-29
# records 16 KB as having caused a production UR_RESULT_ERROR_OUT_OF_RESOURCES.
_GRF_BUDGET_BYTES = 8 * 1024

# Sweep both trees rather than maintaining a file list: a site can sit behind a
# run_large_k_impl<MC> indirection instead of a host ladder, or in a new file.
_ALL_KERNEL_HEADERS = sorted(
    list((_VLLM / "xpu/esimd_kernels").glob("*.h"))
    + list((_SGL / "xpu/esimd_kernels").glob("*.h"))
)


@pytest.mark.parametrize("path", _ALL_KERNEL_HEADERS,
                         ids=lambda p: f"{p.parents[2].name}/{p.name}")
def test_every_simd_local_array_fits_the_grf_budget(path):
    """Any simd<T, N> arr[M] is N*sizeof(T)*M bytes of per-thread GRF.

    Matched on the TYPE, not a variable name: the previous sweep grepped for
    `res_chunks` and missed sglang's copy under a different name, then a
    filename-anchored list missed six sites in four files. Where the extent is a
    template parameter the instantiations are read from the same file, so an arm
    added later is counted too.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    t = tokens(src)

    width = {"float": 4, "uint32_t": 4, "int32_t": 4, "fp16": 2,
             "uint16_t": 2, "int16_t": 2, "uint8_t": 1, "int8_t": 1}
    over, unresolved = [], []
    for m in re.finditer(r"simd<(\w+),(\w+)>\w+\[(\w+)\]", t):
        ty, vl_tok, mc_tok = m.groups()
        if ty not in width:
            continue

        def scoped(tok, at):
            """Resolve an extent using the declaration NEAREST the use.

            A file-wide scrape reads the wrong scope. resadd_norm_gemv_int4.h has
            TWO MAX_CHUNKS: a `template<int MAX_CHUNKS>` at :139 instantiated <8>
            and <16>, and a local `constexpr int MAX_CHUNKS = 4;` at :373 inside a
            different function. Scraping the file gave 4, computed 512*4*4 = 8192,
            and `> 8192` is false -- so the 32 KB cache this test exists to catch
            sat exactly on the boundary and passed. Reverting the whole GRF fix
            left all 337 tests green.

            So: look only at text BEFORE the use, and take whichever of the two
            declaration forms is closer. A template parameter expands to its
            instantiation set, found by `fn<digits>(` on the enclosing function.
            """
            if tok.isdigit():
                return [int(tok)]
            head = t[:at]
            const_at = head.rfind(f"constexprint{tok}=")
            if const_at < 0:
                const_at = head.rfind(f"constint{tok}=")
            tmpl_at = head.rfind(f"template<int{tok}>")
            if const_at > tmpl_at:
                m2 = re.search(rf"(?:constexpr|const)int{tok}=(\d+);", head[const_at:])
                return [int(m2.group(1))] if m2 else []
            if tmpl_at >= 0:
                # The function this template parameter belongs to, then every
                # `fn<digits>(` call anywhere in the file.
                fn = re.search(r"void(\w+)\(int", t[tmpl_at:tmpl_at + 200])
                if fn:
                    inst = re.findall(rf"{fn.group(1)}<(\d+)>\(", t)
                    if inst:
                        return sorted({int(v) for v in inst})
                return []
            return []

        vls, mcs = scoped(vl_tok, m.start()), scoped(mc_tok, m.start())
        if not vls or not mcs:
            # The extent is not statically resolvable (template parameter, enum,
            # #define, constexpr auto...). Silently skipping is how a 32 KB cache
            # re-entered behind `enum { CACHE_N = 16 };`. When the array is
            # VL-wide, reject the SHAPE instead of trying to size it: every one
            # of the six removed caches was exactly `simd<float, VL> x[MC]`.
            # Narrow to the two-pass CACHE shape, not every VL-wide array. A
            # GEMM accumulator (`acc[TILE_M]`, `vacc[M]`) is also VL-wide and
            # unresolvable, but is sized by its tile ladder and sound, so
            # rejecting it is a false positive. A chunk cache is distinguishable:
            # indexed by the chunk counter in a loop over K, i.e. `name[c] =`
            # with a sibling `n_chunks` walk.
            name_m = re.search(
                rf"simd<{ty},{vl_tok}>(\w+)\[{mc_tok}\]", t)
            arr = name_m.group(1) if name_m else ""
            if arr and re.search(rf"{arr}\[c\]\s*=", t) and "n_chunks" in t:
                unresolved.append(f"simd<{ty},{vl_tok}>{arr}[{mc_tok}]")
            continue
        for vl in vls:
            for mc in mcs:
                nbytes = vl * width[ty] * mc
                if nbytes > _GRF_BUDGET_BYTES:
                    over.append(f"simd<{ty},{vl}>[{mc}] = {nbytes // 1024} KB")
    assert not over, (
        f"{path.name}: per-thread GRF over the {_GRF_BUDGET_BYTES // 1024} KB "
        f"budget: {sorted(set(over))}. Stream and re-read from L3, as "
        "norm_add_norm.h does after the device-loss it documents."
    )
    assert not unresolved, (
        f"{path.name}: a VL-wide per-thread cache whose extent cannot be sized "
        f"from source text: {sorted(set(unresolved))}. At VL=512 even 4 chunks "
        "is 8 KB -- the whole budget. Stream instead, or make the extent a "
        "literal so the budget can be checked."
    )


_GATED_NORM = [
    _VLLM / "xpu/esimd_kernel.sycl",
    _SGL / "xpu/esimd_kernel.sycl",
]


@pytest.mark.parametrize("path", _GATED_NORM, ids=lambda p: p.parents[2].name)
def test_rms_norm_gated_pins_its_hardwired_width(path):
    """The kernel's loads are hardwired to 128; V only offsets and divides."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    i = c.find("esimd_rms_norm_gated")
    if i < 0:
        pytest.skip("op not in this tree")
    body = c[i : i + 2000]
    assert re.search(r"TORCH_CHECK\(V == 128,", body), (
        "V=64 makes rows overlap; V=256 leaves half of every row unwritten"
    )


@pytest.mark.parametrize("path", [_VLLM / "xpu/esimd_kernels/norm_gemv_int4.h",
                                  _SGL / "xpu/esimd_kernels/norm_gemv_int4.h"],
                         ids=lambda p: p.parents[3].name)
def test_int4_head_split_divides_hv(path):
    """heads_per_thread is HV / K_SPLIT: a split that does not divide HV
    leaves the remaining heads out of the contraction entirely."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    assert "if (HV % 8 == 0) ks = 8;" in c, (
        "the split must divide HV, not merely be no larger than it"
    )
    assert "if (HV >= 8) ks = 8;" not in c


_INT4_GEMM = [_VLLM / "xpu/esimd_kernels/int4_GEMM.h",
              _SGL / "xpu/esimd_kernels/int4_GEMM.h"]


@pytest.mark.parametrize("path", _INT4_GEMM, ids=_ids)
def test_k_thread_fallback_keeps_whole_scale_groups(path):
    """Forcing k_threads 3 -> 2 splits a scale group when K % 256 != 0.

    sglang carried `if (k_threads == 3) k_threads = 2;` unconditionally while
    vllm computed `(K % 256 == 0) ? 2 : 1`. At K = 384, 1152, 1920, ... (K % 384
    == 0 and K % 256 != 0) the unconditional form gives k_per_thread = K/2, an
    odd multiple of 64, so the top thread's last K_LOAD=128 block runs 64
    elements past the row while thread 0's last block double-counts the same 64.
    The op's own TORCH_CHECK(K % 128 == 0) passes, and the 2D surface width
    clamps the read, so it ships as silently wrong numbers.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    assert re.search(
        r"k_threads = \(\(int\)K % \(2 \* INT4_GEMM_GROUP_SIZE\) == 0\) \? 2 : 1;", c), (
        "the 3->2 fallback must keep whole scale groups per thread"
    )
    assert "if (k_threads == 3) k_threads = 2;" not in c, (
        "the unconditional 3->2 fallback splits a scale group"
    )


def test_int4_gemm_k_split_never_overruns_the_row():
    """Compute the split the host picks and check it covers K exactly."""
    # Pin the transcription: without it the sweep below proves the PYTHON copy
    # safe and says nothing about the kernel, so moving the real budget or the
    # real walk leaves this green.
    for path in _INT4_GEMM:
        if not path.exists():
            continue
        src = code(path.read_text())
        i = src.find("int k_threads = std::max(1, std::min(4,")
        assert i >= 0, f"{path.name}: the k_threads budget is no longer recognisable"
        head = src[i:i + 420]
        # The three needles sit inside a 420-char window; a `k_threads = 4;`
        # just past it pins K_THREADS=4 with every needle intact, starting a
        # thread inside a scale group. Count the writes through the
        # dispatch, not just the text of the walk.
        through = src[i:src.find("#define DISPATCH", i)]
        # The same counter the other pins use, so a compound write counts:
        # `k_threads += (4 - k_threads);` forces K_THREADS=4 unconditionally
        # while an inline plain-write regex sees nothing.
        n_w = len(re.findall(r"(?<![+\-*/%&|^!<>=])k_threads\s*=(?![=])", through))
        n_w += len(re.findall(
            r"(?:^|[;{}])\s*k_threads\s*(?:\+|-|\*|/)=",
            re.sub(r"for\s*\([^;]*;[^;]*;[^)]*\)", " ", through), re.M))
        assert n_w == 2, (
            f"{path.name}: k_threads is plainly assigned {n_w} times before "
            "the dispatch (expected the initial value and the ==3 fixup; the "
            "walk uses k_threads--); an extra write overrides the split the "
            "pins below describe"
        )
        for needle in ("std::min(4, bmg_hw_threads(q) / std::max(n_wgs, 1))",
                       "while (k_threads > 1 && ((int)K % (k_threads * "
                       "INT4_GEMM_GROUP_SIZE) != 0)) k_threads--;",
                       "k_threads = ((int)K % (2 * INT4_GEMM_GROUP_SIZE) == 0) "
                       "? 2 : 1;"):
            assert needle in head, (
                f"{path.name}: the host's split no longer matches the mirror "
                f"below; missing {needle!r}"
            )
    assert any(p.exists() for p in _INT4_GEMM), "neither int4_GEMM.h is present"

    gs = 128
    # The host asks the device for its thread count now, so the sweep runs over
    # every target a Battlemage part can report rather than one literal: the
    # overrun property below must hold on B60 and B70 alike.
    TARGETS = (1280, 2048)
    def k_threads(n, k, target=2048):
        n_wgs = (n + 15) // 16
        kt = max(1, min(4, target // max(n_wgs, 1)))
        while kt > 1 and k % (kt * gs) != 0:
            kt -= 1
        if kt == 3:
            kt = 2 if k % (2 * gs) == 0 else 1
        return kt

    bad = []
    for target in TARGETS:
      for n in (16, 64, 256, 512, 1024, 2048, 4096):
        for k in range(gs, 16385, gs):
            kt = k_threads(n, k, target)
            kpt = k // kt
            if kt * kpt != k:
                bad.append((n, k, kt, "K not covered"))
                continue
            # Each thread walks its slice in whole K_LOAD=128 blocks.
            last = (kt - 1) * kpt + ((kpt - 1) // 128) * 128
            if last + 128 > k:
                bad.append((n, k, kt, f"{last + 128 - k} elements past the row"))
            if kpt % gs:
                bad.append((n, k, kt, "thread starts inside a scale group"))
    assert not bad, f"{len(bad)} overrunning splits, e.g. {bad[:4]}"


_FP8_PERT = [_VLLM / "xpu/esimd_kernels/fp8_GEMM_pert.h",
             _SGL / "xpu/esimd_kernels/fp8_GEMM_pert.h"]


@pytest.mark.parametrize("path", _FP8_PERT, ids=_ids)
def test_batched_pert_catch_all_refuses(path):
    """A substituted K_SPLIT races ks lanes onto one output address.

    batched_gemv_fp8_pert_host sizes the grid N*ks with ks lanes per group, but
    the kernel takes kp = K / K_SPLIT from the template argument. When the
    catch-all launched VL=32/K_SPLIT=1 for a selection of, say, (256,8), lane
    lid walked from lid*K instead of lid*(K/8) -- 16128 bytes past the row at
    N=1 K=2304 -- and all 8 lanes took the `if constexpr (K_SPLIT == 1)` branch,
    which skips slm_init and the barrier, writing the same output[m*N + n].
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    i = c.index("batched_gemv_fp8_pert_host")
    body = c[i:c.index("ws_gemm_fp8_pert_host", i)]
    assert re.search(r"else \{ TORCH_CHECK\(false,", body), (
        "the batched ladder's catch-all must refuse, not substitute a width"
    )
    assert not re.search(r"else \{ LAUNCH_BATCHED\(\d+, \d+\) \}", body), (
        "a fixed-width catch-all discards the selected split"
    )


@pytest.mark.parametrize("path", _FP8_PERT, ids=_ids)
def test_batched_pert_ladder_covers_its_selector(path):
    """Enumerate the selector's image against the batched ladder's arms.

    Uses floor=32, because fp8_GEMM_pert.h has its OWN select_vl_ks (at :76/:77)
    whose walk-down floors at 32 and which carries no TORCH_CHECK. An earlier
    revision of this test passed floor=128 here -- the GEMV file's value -- and
    so reported the 64/32 arms unreachable. They were deleted on that basis, and
    1,630,208 (N,K) pairs that had computed correct answers began to throw.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    _assert_transcription_matches(path, 32)
    c = code(path.read_text())
    # sglang caps the initial vl at 256 via SGLANG_GEMV_VL_CAP; enumerate both
    # that default and vllm's 512 so the arm set must cover either tree's image.
    initial_vls = (256, 512) if "vl = default_vl" in c else (512,)
    i = c.index("batched_gemv_fp8_pert_host")
    body = c[i:c.index("ws_gemm_fp8_pert_host", i)]
    arms = {(int(v), int(sp))
            for v, sp in re.findall(r"LAUNCH_BATCHED\((\d+), *(\d+)\)", body)}
    assert arms, "no LAUNCH_BATCHED arms found -- the parse is wrong"

    uncovered, emitted = {}, set()
    for vl0 in initial_vls:
        for n in range(1, 4097):
            for k in range(32, 16385, 32):
                vl, ks, kpt = _select_vl_ks(n, k, 32, vl0)
                if kpt % vl:
                    continue      # refused by the throwing catch-all
                emitted.add((vl, ks))
                if (vl, ks) not in arms:
                    uncovered.setdefault((vl, ks), (n, k))
    assert not uncovered, (
        f"{path.name}: the batched ladder cannot launch "
        f"{sorted(uncovered.items())}; the catch-all would substitute VL=32/"
        "K_SPLIT=1 against a grid sized for ks lanes and race them onto one "
        "output address"
    )
    # The narrow pairs are the ones that were wrongly deleted; pin them so the
    # same reasoning cannot recur.
    for pair in ((32, 1), (32, 2), (64, 1)):
        assert pair in emitted, (
            f"{path.name}: {pair} is no longer reachable -- if the selector "
            "really changed, re-derive this list from it rather than trimming "
            "the ladder"
        )


def test_verify_family_gates_cache_indices_without_abandoning_out():
    """cache_indices multiplies a state stride with no bound in the kernel.

    A dtype check on these indices says nothing about their value. Three of the
    four sites write `out` (CausalConv1dVerifyFn, GdnTargetVerifyFn,
    GdnFusedVerifyFn), so upstream's bare `return` on its pad sentinel does not
    transfer -- that kernel writes only state. Gate the history/seed load and
    keep producing the row.

    Specifically NOT a clamp to row 0: that makes a padding slot read another
    sequence's live state.
    """
    path = _SGL / "eagle/eagle.sycl"
    if not path.exists():
        pytest.skip("sglang tree not present")
    c = code(path.read_text())

    # Every read of cache_indices must produce a slot test -- either a
    # have_slot flag, or an immediate `if (slot < 0) return;` in a kernel that
    # writes no output.
    #
    # The first version of this accepted `slot = cache_indices[{idx}]` as
    # evidence of a guard. That string IS the pre-fix source (main:1547 and
    # main:1662), so naming the variable `slot` satisfied it and a full revert of
    # the commit passed. It also used a fixed +-200/+120 character window, which
    # is not a scope.
    # ...and it still did, at +-160/260. A fixed character count is still not a
    # scope: it cannot see a guard one statement further out, and it breaks on
    # any edit that lengthens the statement -- adding the upper bound this pool
    # needs pushed form B's anchored match off the end. Walk statements instead,
    # bounded by the enclosing braces, so the window tracks the code.
    def statement_group(pos, before=2, after=3):
        """The few statements around pos, clipped at the enclosing braces."""
        lo = pos
        for _ in range(before):
            prev = max(c.rfind(";", 0, lo), c.rfind("{", 0, lo),
                       c.rfind("}", 0, lo))
            if prev < 0:
                break
            lo = prev
        hi = pos
        for _ in range(after):
            nxt = c.find(";", hi + 1)
            brace = c.find("}", hi + 1)
            if nxt < 0 or (0 <= brace < nxt):
                hi = brace if brace >= 0 else len(c)
                break
            hi = nxt
        return c[max(0, lo):min(len(c), hi + 1)]

    sites = list(re.finditer(r"cache_indices\[(\w+)\]", c))
    assert sites, "no cache_indices read found -- the parse is wrong"
    for m in sites:
        idx = m.group(1)
        scope = statement_group(m.start())
        tail = scope
        gated = (
            # form A: a flag derived from this very index
            re.search(rf"have_slot = cache_indices\[{idx}\] >= 0\b", scope)
            or re.search(r"have_slot = slot >= 0\b", scope)
            # form B: an immediate refusal, valid only for a cache-only kernel
            # Stop at the lower bound: the guard carries an upper bound now,
            # so pinning through `) return;` rejected the safer form.
            or re.search(r"cache_indices\[\w+\];\s*if \(slot < 0\b",
                         scope)
        )
        assert gated, (
            f"cache_indices[{idx}] multiplies a state stride with no slot test "
            f"within its own statement group: ...{tail[:120]}"
        )

    # The flag must be USED, not merely derived. A 120-character lookback for
    # the identifier passes on a full de-gating, because the `const bool
    # have_slot = ...;` declaration sits inside that window -- presence near the
    # site is not use at the site. Require the pointer to be built by a ternary
    # on the flag, or the statement group to refuse outright.
    for m in re.finditer(r"(conv_state|ssm_state) \+ \(int64_t\)"
                         r"(?:cache_indices\[\w+\]|slot)", c):
        head = c[max(0, m.start() - 200):m.start()]
        gated = (
            # `cs = have_slot ? conv_state + ... : nullptr;`
            re.search(r"=\s*have_slot\s*\?\s*$", head)
            # `... = have_slot ? <ptr expr>` spread over the preceding tokens
            or re.search(r"have_slot\s*\?\s*(?:\w+\s*)?$", head)
            # a cache-only kernel that refuses before forming the pointer.
            # Regex to the lower bound, not an exact string: the guard is
            # two-sided now (`slot < 0 || slot >= cs_rows`).
            or re.search(r"if \(slot < 0\b", head)
        )
        assert gated, (
            f"a {m.group(1)} pointer is formed without the slot test applying "
            f"to it: ...{head[-110:]}"
        )

    # And the flag must be consumed at the load, not only at the pointer: a
    # `cs[i]` read with the ternary removed from the loop re-arms the OOB.
    if "const T* cs = have_slot" in c:
        assert re.search(r"win\[i\] = have_slot \? \(float\)cs\[i\] : 0\.0f;", c), (
            "the history load must be gated too; a nullptr cs with an ungated "
            "read is a null dereference, and an ungated pointer is the original "
            "out-of-bounds read"
        )
    # Every site that derives the flag must reference it at least twice (once to
    # derive, once or more to use).
    for name in ("have_slot",):
        derivations = len(re.findall(rf"(?:const bool )?{name} = ", c))
        uses = c.count(name) - derivations
        assert uses >= derivations, (
            f"{name} is derived {derivations} times but used only {uses}; a "
            "derived-and-unused flag is a gate in name only"
        )

    # The clamp form must not come back.
    assert not re.search(r"\(\s*have_slot \? cache_indices\[\w+\] : 0\s*\)", c), (
        "clamping a padding slot to row 0 reads another sequence's live state"
    )
    assert "cache_indices[n] < 0) return;" not in c, (
        "CausalConv1dVerifyFn writes out for every row; returning abandons it"
    )
    # ...and the conv output store must stay unconditional.
    assert "oc[(int64_t)t * os_t] = (T)acc;" in c, (
        "the per-token conv output store is the row's only definition"
    )


def test_norm_gemv_norm_fp16_ties_every_tensor_to_proj_w():
    """N and K come from proj_w alone; the rest must be checked against them.

    vllm had zero guards here. The kernel reads residual, scale_with_root and
    pre_ff_w across [0,K) and WRITES moe_input across the same range, so
    proj_w=[128,4096] with residual.numel()=2048 overread 4096 bytes past three
    inputs and overwrote 4096 bytes past moe_input -- a heap write. The sglang
    twin hard-pins 2816/128; the dim-relative form catches the same mismatches
    without baking in one model's shape.
    """
    seen = 0
    for root in (_VLLM, _SGL):
        path = root / "xpu/esimd_kernel.sycl"
        if not path.exists():
            continue
        c = code(path.read_text())
        i = c.find("esimd_norm_gemv_norm_fp16(")
        if i < 0:
            continue
        seen += 1
        body = c[i:i + 3000]
        assert "TORCH_CHECK" in body, (
            f"{root.parents[0].name}: esimd_norm_gemv_norm_fp16 has no guard at "
            "all; the kernel writes moe_input across proj_w's K"
        )
        # Dim-relative in either tree's spelling (vllm names the dim K, sglang
        # names it `hidden`), or the older pinned form. Matching one tree's
        # variable name made this fail when the other was made dim-relative.
        relative = re.search(r"residual\.numel\(\) == (?:K|hidden)\b", body)
        pinned = re.search(r"residual\.numel\(\) == \d+", body)
        assert relative or pinned, (
            f"{root.parents[0].name}: nothing ties residual's extent to the "
            "range the kernel walks"
        )
        assert re.search(
            r"moe_input\.(?:numel\(\) == (?:K|hidden)\b|sizes\(\) == residual\.sizes\(\))",
            body), (
            f"{root.parents[0].name}: moe_input is WRITTEN across [0,K) and its "
            "extent is unchecked"
        )
        assert re.search(
            r"router_logits\.numel\(\) == (?:N|n_experts\b|\d+)", body), (
            f"{root.parents[0].name}: router_logits is written at index n < N"
        )
    assert seen, ("the anchor was not found in either tree; this test stopped\n                   checking anything")


# Ops that genuinely write nothing: a void return with no mutable slot is correct
# for these, so they are not Category A.
_PURE_VOID_OPS = {"dispose", "register_buffer", "meta_size"}


@pytest.mark.parametrize("rel", [
    "xpu/torch_extension.cc", "xpu/torch_extension_ar.cc",
    "xpu/torch_extension_moe.cc", "xpu/torch_extension_lgrf.cc",
    "xpu/torch_extension_gemm.cc", "xpu/torch_extension_q4_0.cc",
    "xpu/torch_extension_topk_v2.cc", "xpu/torch_extension_deepseek.cc",
])
def test_no_void_op_declares_zero_mutable_slots(rel):
    """`-> ()` with no Tensor(a!) reads as PURE and licenses DCE.

    Scanning every void op beats a fixed list, which cannot cover ops added
    later. Note `Tensor!` is a valid mutable marker too (17 uses in
    sglang's eagle registrations); a regex accepting only `Tensor(a!)` invents
    about 25 false positives.
    """
    offenders = []
    for root in (_VLLM, _SGL):
        path = root / rel
        if not path.exists():
            continue
        raw = path.read_text()
        # Join adjacent literals so a schema split across lines reads as one.
        joined = re.sub(r'"\s*"', "", raw)
        for m in re.finditer(r'"(\w+)\(([^"]*?)\)\s*->\s*\(\)"', joined):
            op, args = m.group(1), m.group(2)
            if op in _PURE_VOID_OPS or "Tensor" not in args:
                continue
            if "!" in args:          # Tensor(a!) or the Tensor! shorthand
                continue
            offenders.append(f"{root.parents[0].name}/{rel}:{op}")
    assert not offenders, (
        "void ops with no mutable slot read as pure to torch.compile: "
        f"{offenders}"
    )


# Each entry names every tensor an op writes, so dropping one annotation fails.
# Checking for a '!' anywhere in the schema does not: a sibling slot's marker
# satisfies it.
_MUTATED_SLOTS = [
    ("xpu/torch_extension.cc", "esimd_scaled_resadd_norm_gemv_fp8_pert",
     ["residual", "qkv_out"]),
    ("xpu/torch_extension.cc", "esimd_norm_gemv_norm_fp16",
     ["router_logits", "moe_input"]),
    ("xpu/torch_extension_ar.cc", "init_custom_ar", ["meta", "rank_data"]),
    ("xpu/torch_extension.cc", "esimd_norm_add_norm_gemv_gelu_fp8",
     ["residual_output", "activation_output"]),
    ("xpu/torch_extension.cc", "esimd_rmsnorm_gemv_fp8", ["output"]),
    ("xpu/torch_extension.cc", "esimd_dual_rmsnorm_residual_scalar", ["output"]),
    # All three siblings are structurally identical and need the same marker.
    ("xpu/torch_extension_moe.cc", "esimd_moe_gemm_fp8", ["output"]),
    ("xpu/torch_extension_moe.cc", "esimd_moe_gemm_fp8_pert", ["output"]),
    ("xpu/torch_extension_moe.cc", "esimd_moe_gemm_fp8_blockscale", ["output"]),
]


@pytest.mark.parametrize("rel,op,written", _MUTATED_SLOTS,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_every_written_tensor_carries_its_own_marker(rel, op, written):
    """hidden_states is deliberately absent: the kernel takes it const.

    The host casts it non-const, which is gratuitous -- scaled_resadd's
    kernel declares `const fp16* hidden_ptr` and the file has zero stores
    through it. Marking it would claim a mutation that does not happen.
    """
    seen = 0
    for root in (_VLLM, _SGL):
        path = root / rel
        if not path.exists():
            continue
        raw = path.read_text()
        # Join adjacent literals before the presence test: splitting the op
        # name as "esimd_foo" "(Tensor ..." is legal C++ that compiles
        # identically, and an unjoined check skipped the op entirely -- every
        # alias marker could then be dropped with the suite green.
        if op + '(' not in re.sub(r'"\s*"', "", raw):
            continue
        schema = schema_of(raw, op)
        seen += 1
        for name in written:
            assert re.search(rf"Tensor(?:\([a-z]!\)|!)\s+{name}(?=[,)\s])", schema), (
                f"{root.parents[0].name}/{rel}: {op} writes {name} but does not "
                f"declare it mutable: {schema}"
            )
    assert seen, f"{op} found in neither tree -- this test is not running"


_DIVISIBILITY_GUARDS = [
    ("xpu/esimd_kernels/fp16_GEMV.h",
     r"TORCH_CHECK\(K % 32 == 0 && K % ks == 0 && \(K / ks\) % vl == 0,",
     "the walk-down breaks at vl=32 regardless of divisibility and the "
     "catch-all ignores the selection; 56715 (N,K) pairs overread and 1410 "
     "silently dropped the tail"),
    ("xpu/esimd_kernels/int4_GEMV.h",
     r"TORCH_CHECK\(K % ks == 0 && kp % INT4_GROUP_SIZE == 0,",
     "kp % GROUP alone passes at N=128 K=2050 while 8*256 visits 2048 of "
     "2050 elements; 358 pairs pass the old guard and drop 1-7"),
    ("xpu/esimd_kernels/fused_add_rms_norm_batched.h",
     r"TORCH_CHECK\(K % 64 == 0,",
     "K=100 sums 64 of 100 elements while dividing by 100 and leaves "
     "columns 64..99 of every row stale in both hidden and residual"),
]


@pytest.mark.parametrize("rel,pattern,why", _DIVISIBILITY_GUARDS,
                         ids=lambda v: v.split("/")[-1] if isinstance(v, str) else "")
def test_divisibility_guard_present_in_both_trees(rel, pattern, why):
    """These kernels walk K in whole VL chunks with no tail path."""
    seen = 0
    for root in (_VLLM, _SGL):
        path = root / rel
        if not path.exists():
            continue
        c = code(path.read_text())
        seen += 1
        assert re.search(pattern, c), f"{root.parents[0].name}/{rel}: {why}"
    assert seen, f"{rel} found in neither tree -- this test is not running"


def test_rmsnorm_residual_scalar_checks_its_weight():
    """weight is block_loaded across all of k, not just the dtype check."""
    path = _SGL / "xpu/esimd_kernel.sycl"
    if not path.exists():
        pytest.skip("sglang tree not present")
    c = code(path.read_text())
    i = c.find("esimd_rmsnorm_residual_scalar(")
    assert i >= 0
    body = c[i:i + 2000]
    assert "weight.numel() == k" in body, (
        "weight of 256 with k=5376 reads 10240 bytes past a 512-byte allocation"
    )
    assert "weight.is_contiguous()" in body, (
        "a strided weight is read as if contiguous"
    )


def test_int4_resadd_bounds_only_the_path_that_needs_it():
    """The VL=512 bound belongs to the ks<=1 arm, not to every path.

    Placed above the ks computation it rejected the K_SPLIT kernels too, which
    have their own VL arms and were always correct: 49,152 (N,K) pairs over 96
    distinct K -- 11008, 14336, 16384, 22016 among them -- began to throw. And
    K % 128 is too weak for the arm that does need a bound, because
    run_large_k_impl's n_chunks = K / 512 has no tail loop: at K=2816 with N>512
    it contracted 2560 of 2816 elements and left normed_out stale past 2560.
    """
    path = _SGL / "xpu/esimd_kernels/resadd_norm_gemv_int4.h"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    i = c.index("resadd_norm_gemv_int4_pert_host")
    body = c[i:]

    ks_at = body.index("int ks = 1;")
    arm_at = body.index("if (ks <= 1 && K >= 512) {")
    k512 = body.index("K % 512 == 0")
    assert ks_at < arm_at, "the ks ladder must precede its arm"
    assert k512 > arm_at, (
        "the VL=512 bound sits above the ks computation, so it rejects the "
        "K_SPLIT paths as well -- 96 distinct K that were correct then throw"
    )
    # K % 128 applies to every path and must stay above the arm.
    k128 = body.index("K % 128 == 0")
    assert k128 < ks_at, "K % 128 applies to all three kernels"
    # The bound must precede the residual pre-pass submit.
    prepass = body.index("ResAddResidualOnly_int4_kernel")
    assert k512 < prepass, (
        "a throw after the pre-pass leaves the caller's residual already added"
    )
    # No companion K <= 8192: the kernel streams, so nothing bounds K.
    assert "K <= 8192" not in body, (
        "a bound with no live object behind it; the kernel streams"
    )


@pytest.mark.parametrize("tree", ["vllm", "sglang"], ids=lambda v: v)
def test_streamed_resadd_refuses_an_aliasing_normed_out(tree):
    """The streamed path has no aliasing immunity, so it must refuse aliases.

    Pass 2 re-loads residual from L3 while only work-group 0 writes normed_out,
    with no ordering between them, so `normed_out == residual` is a
    cross-work-group race. A register cache would make it harmless; streaming
    has none.
    """
    root = _VLLM if tree == "vllm" else _SGL
    path = root / "xpu/esimd_kernel.sycl"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    sigs = c.count("at::Tensor normed_out,")
    assert sigs, f"{tree}: no normed_out binding found -- the parse is wrong"
    guards = len(re.findall(
        r"TORCH_CHECK\(normed_out\.data_ptr\(\) != residual\.data_ptr\(\),", c))
    assert guards == sigs, (
        f"{tree}: {sigs} ops take normed_out but only {guards} refuse an "
        "aliasing buffer"
    )


def test_fp4_scale_clamp_saturates_to_max_finite():
    """143 - 112 = 31, and exponent field 31 is the fp16 Inf encoding.

    Every raw scale >= 143 decoded to Inf, and Inf times an E2M1 zero
    (m == 0 -> 0x0000) is NaN, poisoning a whole output tile rather than
    clipping it. Field 30 (raw 142) is max-finite, 32768.0. Reverting this left
    the suite green until now.
    """
    path = _VLLM / "deepseek_v41/fp4_dequant.h"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    m = re.search(r"min\(shifted, \(uint16_t\)(\d+)\)", c)
    assert m, "the UE8M0 scale clamp is gone"
    clamp = int(m.group(1))
    bias = 112
    assert clamp - bias <= 30, (
        f"clamp {clamp} maps to fp16 exponent field {clamp - bias}; field 31 is "
        "Inf/NaN, so a saturated scale becomes Inf and NaNs the tile"
    )
    assert clamp - bias == 30, (
        f"clamp {clamp} saturates below max-finite (field {clamp - bias} < 30), "
        "which silently shrinks large scales"
    )


def test_onednn_primitive_cache_bounds_its_device_index():
    """mappings is a fixed 16-element thread_local array indexed by device id.

    Unbounded, a 17th device corrupts thread-local storage rather than faulting.
    Both sibling pools in the same subsystem already call check_device_index.
    """
    path = _SGL / "xpu/onednn_w8a16/onednn_ext.h"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    i = c.find("get_cache(const int device_id)")
    assert i >= 0, "get_cache not found"
    body = c[i:i + 400]
    assert "check_device_index(device_id)" in body, (
        "device_id indexes a fixed-size thread_local array with no bound"
    )
    assert body.index("check_device_index") < body.index("mappings[device_id]"), (
        "the bound must precede the indexing"
    )


def test_q5q6_admission_gate_matches_the_kernel_it_admits_into():
    """The caller-side gate and the kernel's TORCH_CHECK must agree.

    This is the arm-set-vs-selector class a third time, inverted: the VL=256
    instantiation was deleted from both kernels and both hosts were given
    TORCH_CHECK(K % 512 == 0), but the Python admission gate kept testing
    K % 256. Exactly half of all super-block-aligned K -- 5376 (Gemma-3/4),
    2816, 1280, 768, ... -- was admitted, repacked with a 256-element tile, and
    handed to a kernel that refuses it. Three source comments asserted the
    deleted arm still existed, which is why the gate looked correct.
    """
    patch = _ROOT / "sglang/patches/sglang_for_multi_arc.patch"
    if not patch.exists():
        pytest.skip("sglang patch not present")
    ptext = patch.read_text()

    # What the kernels demand.
    demanded = {}
    for name, rel in (("q5_k", "xpu/esimd_kernels/q5_k_GEMV.h"),
                      ("q6_k", "xpu/esimd_kernels/q6_k_GEMV.h")):
        kpath = _SGL / rel
        if not kpath.exists():
            pytest.skip(f"{kpath} not present")
        kc = code(kpath.read_text())
        tag = name.upper() + "_VL"
        m = re.search(rf"constexpr int {tag} = (\d+);", kc)
        assert m, f"{name}: {tag} not found"
        vl = int(m.group(1))
        assert re.search(rf"TORCH_CHECK\(\s*K % {tag} == 0,", kc), (
            f"{name}: the host no longer pins K % {tag}"
        )
        # Any narrower instantiation would make the gate's /2 legitimate.
        narrow = re.findall(rf"{name.upper()}_gemv(?:_M)?_kernel<[^>]*{tag} */ *2",
                            kc)
        demanded[name] = (vl, bool(narrow))

    for name, (vl, narrow) in demanded.items():
        assert not narrow, (
            f"{name}: a {vl // 2} instantiation exists again -- if the narrow arm "
            "is back, the gate may halve the modulus, but say so here"
        )

    # What the gate admits. No site may halve the tile while no narrow arm exists.
    halved = re.findall(r"_Q5Q6_VL // 2", ptext)
    assert not halved, (
        f"{len(halved)} admission sites test half the kernel tile while the "
        "kernels are instantiated only at the full width: half of all "
        "super-block-aligned K is admitted straight into a TORCH_CHECK throw"
    )
    m = re.search(r"_Q5Q6_VL = (\d+)", ptext)
    assert m, "_Q5Q6_VL not found in the patch"
    assert int(m.group(1)) == demanded["q5_k"][0], (
        f"the gate's tile {m.group(1)} disagrees with the kernel's "
        f"{demanded['q5_k'][0]}"
    )
    # And no comment may still promise the deleted arm.
    assert "is served by the VL=256 instantiation" not in ptext, (
        "a comment still asserts the deleted narrow arm serves 5376"
    )


@pytest.mark.parametrize("path", _GDN_SEQ, ids=_ids)
def test_padding_row_conv_result_is_zero_in_both_trees(path):
    """The bias add and the SiLU must sit inside the slot gate.

    With the gate closed before the MAC, a padding row has s0..s2 and x_f32 all
    zero, so conv_result = conv_bias and then SiLU(conv_bias), which is NOT zero
    for a nonzero bias -- while the other tree, with the MAC inside the gate,
    leaves conv_result at 0.0f. A silent cross-tree numeric divergence on the
    same input. The cstate_base walk does not cover this: it checks that LOADS
    are enclosed, and the bias add is not a cstate_base load.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    gate = "if (have_conv_slot) {"
    if gate not in c:
        pytest.skip(f"{path.name}: no lo-chunk slot gate (large_h-only file)")
    start = c.index(gate)

    # Walk to the matching close brace of the gate.
    depth, end = 0, None
    for m in re.finditer(r"[{}]", c[start:]):
        depth += 1 if m.group(0) == "{" else -1
        if depth == 0:
            end = start + m.start()
            break
    assert end is not None, f"{path.name}: unbalanced braces after the slot gate"
    gated = c[start:end]

    for needle, why in (
        ("conv_bias_ptr", "the bias add: outside the gate a padding row gets "
                          "SiLU(conv_bias) instead of 0"),
        ("exp(-conv_result)", "the SiLU: it must not run on a bias-only sum"),
    ):
        assert needle in gated, (
            f"{path.name}: {needle} is outside the have_conv_slot gate -- {why}"
        )
    # And conv_result must be zero-initialised so the fall-through is defined.
    assert re.search(r"simd<float, 64> conv_result\(0\.0f\)", c), (
        f"{path.name}: conv_result must be zero-initialised for the padding row"
    )


_OMNI = _ROOT / "omni/omni_xpu_kernel/omni_xpu_kernel/csrc"


def test_omni_int4_gemm_siblings_agree_on_their_group_guards():
    """omni is outside compile_check.sh's two-tree sweep and has no other test.

    Reverting the two TORCH_CHECKs added to onednn_int4_gemm_add_to_output left
    all 337 tests green: that subsystem was guarded by nothing whatsoever. It
    stays outside the 41-TU sweep because its build flags differ (-doubleGRF,
    other device targets), so this contract test is the only cover it has.

    Both entry points do the identical num_groups / group_size arithmetic and
    must therefore carry the identical checks -- the preconverted one always did,
    add_to_output did not, and group_size becomes the oneDNN scale group stride,
    so a non-exact K/num_groups mis-strides every group past the first.
    """
    path = _OMNI / "onednn_int4_gemm.cpp"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())

    fns = [m.start() for m in re.finditer(
        r"onednn_int4_gemm_(?:preconverted|add_to_output)\(", c)]
    assert len(fns) >= 2, (
        f"expected both int4 gemm entry points, found {len(fns)}"
    )
    checked = 0
    for at in fns:
        body = c[at:at + 2600]
        if "num_groups = scales_f16.size(0)" not in body:
            continue          # a declaration, not the definition
        checked += 1
        assert "TORCH_CHECK(scales_f16.size(1) == N" in body, (
            "an int4 gemm entry reads scales_f16 as [G, N] without checking N; "
            "an undersized scale tensor is over-described to oneDNN"
        )
        assert re.search(r"TORCH_CHECK\(group_size \* num_groups == K", body), (
            "an int4 gemm entry derives group_size = K / num_groups without "
            "checking the division is exact; K=3360 with 64 groups gives 52 "
            "(52*64 = 3328) and every group past the first reads the wrong row"
        )
    # Both `continue`s can fire together -- renaming the num_groups expression
    # empties the loop and this test passes having checked no entry point at
    # all. len(fns) >= 2 above does not cover it: that counts declarations.
    assert checked >= 2, (
        f"only {checked} int4 gemm definition(s) were examined; the anchor "
        "moved and this test stopped checking the entry points"
    )


def test_omni_norm_entries_guard_hidden_size():
    """layer_norm and fused_add_rms_norm omit rms_norm's hidden_size guard.

    The only guards that ever existed were assert(), and -DNDEBUG is set at six
    setup.py sites, so they compile to nothing. hidden_size=48 writes 32 of 48
    elements and computes mean/variance over 32, so even the written values are
    wrong; fused_add_rms_norm mutates caller buffers in place, leaving residual
    half-updated and the written columns sqrt(48/32) = 1.22x too large.
    """
    path = _OMNI / "norm.cpp"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    # Per ENTRY POINT, not file-wide. A file-wide count passes on rms_norm's
    # guard alone, which is how layer_norm and fused_add_rms_norm stayed
    # unguarded while this test was green.
    for fn in ("torch::Tensor rms_norm(", "torch::Tensor layer_norm(",
               "void fused_add_rms_norm("):
        i = c.find(fn)
        if i < 0:
            continue
        body = c[i:i + 3000]
        assert "int64_t hidden_size = " in body, (
            f"norm.cpp: {fn} no longer derives hidden_size -- re-derive this test"
        )
        assert re.search(r"TORCH_CHECK\([^;]*hidden_size % 32 == 0", body) or \
               re.search(r"TORCH_CHECK\([^;]*supported_hidden_size", body), (
            f"norm.cpp: {fn} walks hidden_size in whole 32/64-wide blocks with "
            "no tail and does not check divisibility; the only guards that ever "
            "existed were assert(), which -DNDEBUG compiles to nothing"
        )
    # The surviving assert(hidden_size % BS == 0) calls live in the KERNEL
    # bodies, which correctly carry no TORCH_CHECK -- the contract belongs at the
    # host entries checked above. They are dead under -DNDEBUG (set at six
    # setup.py sites), so they are documentation, not cover; the point of the
    # per-entry loop above is that the host checks are what actually run.
    dead = len(re.findall(r"assert\(hidden_size % BS == 0\)", c))
    hosts = len(re.findall(r"TORCH_CHECK\([^;]*hidden_size % 32 == 0", c)) + \
        len(re.findall(r"TORCH_CHECK\([^;]*supported_hidden_size", c))
    assert hosts >= 3, (
        f"{dead} kernel-level assert(hidden_size % BS) are dead under -DNDEBUG "
        f"and only {hosts} host TORCH_CHECKs carry the contract; all three "
        "entry points (rms_norm, layer_norm, fused_add_rms_norm) need one"
    )


_FP8_BMG = [_VLLM / "xpu/esimd_kernels/fp8_GEMV_bmg.h",
            _SGL / "xpu/esimd_kernels/fp8_GEMV_bmg.h"]

_BMG_HW_THREADS = 2048


def _select_bmg(n, k):
    """LITERAL transcription of select_bmg in fp8_GEMV_bmg.h."""
    tks = 8 if n * 8 <= _BMG_HW_THREADS else (
        4 if n * 4 <= _BMG_HW_THREADS else (
            2 if n * 2 <= _BMG_HW_THREADS else 1))
    ks, sp = 1, tks
    while sp >= 1:
        if k % sp == 0:
            ks = sp
            break
        sp //= 2
    kp = k // ks
    for c in (256, 128, 64, 32):
        if kp >= c:
            tail = kp - (kp // c) * c
            if tail == 0:
                return c, 0, ks
            if tail in (8, 16, 32, 64, 128):
                return c, tail, ks
    return 32, -1, ks


@pytest.mark.parametrize("path", _FP8_BMG, ids=_ids)
def test_bmg_ladder_covers_every_arm_its_selector_emits(path):
    """The masked catch-all is exact but runs at width 32 whatever was chosen.

    The ladder omitted 40 of the (vl_big, vl_tail, ks) triples select_bmg emits,
    covering 2,311,424 shapes at up to 8x narrow -- including (256,8,4) at
    K=1056 for N in [257,512], this file's own headline shape, where the
    redirect comment advertises a 1.78x speedup that was being lost.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())

    # Pin the transcription: an unpinned model silently invalidates the sweep.
    assert "int candidates[] = {256, 128, 64, 32};" in c, (
        "select_bmg's VL_BIG candidate list changed; re-derive _select_bmg"
    )
    assert "for (int t : {8, 16, 32, 64, 128})" in c, (
        "select_bmg's tail power-of-2 set changed; re-derive _select_bmg"
    )
    assert "if (N * 8 <= (uint32_t)hw_threads) target_ks = 8;" in c, (
        "select_bmg's ks ladder changed; re-derive _select_bmg"
    )
    # The target is a parameter now, because B60 and B70 have different Xe core
    # counts. _select_bmg models the default, so that default must stay 2048
    # (B70) and the parameter must actually default to it -- otherwise the
    # sweep below describes a selector no device gets.
    occ = path.parent / "bmg_occupancy.h"
    occ_c = code(occ.read_text()) if occ.exists() else c
    assert re.search(r"BMG_HW_THREADS = 2048", occ_c), (
        "BMG_HW_THREADS changed; _select_bmg hardcodes 2048"
    )
    assert "int hw_threads = BMG_HW_THREADS)" in c, (
        "select_bmg's thread target no longer defaults to BMG_HW_THREADS; the "
        "sweep below models that default"
    )

    notail = {(int(v), int(k))
              for v, k in re.findall(r"LAUNCH_NOTAIL\((\d+), *(\d+)\)", c)}
    tail = {(int(v), int(t), int(k)) for v, t, k in
            re.findall(r"LAUNCH_TAIL\((\d+), *(\d+), *(\d+)\)", c)}
    assert notail and tail, "no LAUNCH arms found -- the parse is wrong"

    uncovered = {}
    for n in range(1, 4097):
        for k in range(8, 16385, 8):
            vb, vt, ks = _select_bmg(n, k)
            if vt == -1:
                continue        # the masked kernel is the selected path here
            key = (vb, ks) if vt == 0 else (vb, vt, ks)
            if key not in (notail if vt == 0 else tail):
                uncovered.setdefault(key, (n, k))
    assert not uncovered, (
        f"{path.name}: {len(uncovered)} arms the selector emits have no "
        f"instantiation, so they fall to the width-32 masked catch-all: "
        f"{sorted(uncovered.items())[:4]}"
    )
    # (256,8,4) is K=1056's arm for N in [257,512].
    assert (256, 8, 4) in tail, (
        "K=1056 at N in [257,512] selects (256,8,4); without that arm this "
        "file's headline shape runs 8x narrow"
    )


def test_kv_scatter_gate_admits_only_what_the_kernel_accepts():
    """A 2-byte itemsize proxy admits bf16 into this fp16-only kernel.

    The kernel hard-requires at::ScalarType::Half on k, v and both caches, so a
    `store_dtype.itemsize == 2` test is not sufficient: it is equally true of
    bfloat16, which then aborts on its first KV write.
    """
    patch = _ROOT / "sglang/patches/sglang_for_multi_arc.patch"
    kern = _SGL / "xpu/esimd_kernel.sycl"
    if not patch.exists() or not kern.exists():
        pytest.skip("sglang tree not present")
    ptext = patch.read_text()
    kc = code(kern.read_text())

    # What the kernel demands of the four tensors.
    i = kc.index("esimd_kv_scatter(")
    body = kc[i:i + 2500]
    halfs = len(re.findall(r"scalar_type\(\) == at::ScalarType::Half", body))
    assert halfs >= 4, (
        f"kv_scatter checks Half on only {halfs} of k/v/k_cache/v_cache; "
        "re-derive this test from the kernel"
    )

    # The gate must test the dtype exactly, not proxy it by width.
    assert "store_dtype.itemsize == 2" not in ptext, (
        "an itemsize proxy admits bfloat16 into a kernel that requires Half; "
        "the forward pass aborts on the first KV write"
    )
    assert "store_dtype == torch.float16" in ptext, (
        "the kv_scatter gate must test store_dtype == torch.float16"
    )


def test_omni_onednn_caches_key_on_the_device():
    """A oneDNN primitive cache owning an engine must key on the device.

    CachedPrimitive holds a dnnl::engine and dnnl::stream, and the lookup returns
    before consulting the engine resolved for the caller -- so a key without the
    device index makes rank 0 populate the entry and every other rank wrap its own
    USM pointers in GPU 0's engine and submit on GPU 0's stream. 7 of 8 ranks wrong
    at TP=8. The int8 and fp8 siblings in this directory already key on it.
    """
    path = _OMNI / "onednn_int4_gemm.cpp"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    m = re.search(r"using CacheKey = std::tuple<([^>]*)>;", c)
    assert m, "CacheKey is no longer a std::tuple -- re-derive this test"
    n_fields = len(m.group(1).split(","))
    assert n_fields >= 6, (
        f"CacheKey has {n_fields} fields; the device index is missing, so a "
        "primitive built on one GPU's engine is reused on every other GPU"
    )
    for m2 in re.finditer(r"CacheKey key\(([^;]*)\);", c):
        assert "device.index()" in m2.group(1), (
            f"a CacheKey is constructed without the device index: {m2.group(1)}"
        )


def test_omni_sdp_vscale_cache_keys_on_device_and_locks_its_read():
    """The fast path read four mutable statics with no lock and no device test."""
    path = _OMNI / "sdp.cpp"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    i = c.index("cached_v_scale_broadcast")
    body = c[i:i + 3000]
    assert "cached_device == v.device()" in body, (
        "the V-scale fast path does not test the device: a GPU-5 rank hits it "
        "holding GPU-0 tensors, and cached_effective_alpha.data_ptr() is a "
        "GPU-0 USM address handed to a kernel on GPU 5's queue"
    )
    # The read must be inside a lock_guard, like the writers.
    read_at = body.index("v_scaled = v / cached_v_scale_broadcast;")
    head = body[:read_at]
    assert "lock_guard<std::mutex> guard(cache_mutex)" in head, (
        "the fast-path read holds no lock while the writers take cache_mutex; a "
        "torn torch::Tensor assignment is a use-after-free even on one device"
    )


def test_onednn_get_stream_bounds_its_device_index():
    """get_engine validates the same argument four lines above."""
    path = _SGL / "xpu/onednn_w8a16/onednn_runtime.h"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    for fn in ("get_engine(", "get_stream("):
        i = c.index(fn)
        body = c[i:i + 500]
        assert "check_device_index(device_index)" in body, (
            f"{fn} indexes a pool by an unvalidated device_index; get_stream "
            "does an unchecked operator[] WRITE into stream_pool"
        )


def test_deepseek_v41_ops_still_refuse():
    """Nothing pinned either refusal, and deleting one is the natural first move.

    Both kernels read uninitialised registers and store nothing (topk), or build
    an invalid nd_range at M<16 (fp4 gemm). Xe2 XMX has no FP4 or FP8 matrix
    arithmetic, so neither can work as written.
    """
    path = _VLLM / "xpu/torch_extension_deepseek.cc"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    for op in ("deepseek_v41_fp4_gemm", "deepseek_v41_noaux_tc_topk"):
        i = c.find("void " + op + "(")
        if i < 0:
            i = c.find(op + "(")
        assert i >= 0, f"{op} not found"
        body = c[i:i + 1200]
        # Match  exactly: a regex on TORCH_CHECK(...) alone accepts
        # TORCH_CHECK(true, ...), which is the shape a careless re-enable takes.
        assert re.search(r"TORCH_CHECK\(\s*false\s*,", body) \
            and not re.search(r"TORCH_CHECK\(\s*true\s*,", body), (
            f"{op} no longer refuses at entry; the kernel neither loads its "
            "operands nor stores a result, and Xe2 XMX has no FP4/FP8 matrix path"
        )


def test_int4_resadd_keeps_its_small_k_arm():
    """The kernel body dispatches again below the host's ks ladder.

    `ks <= 1` is not the end of the dispatch: the kernel's operator() branches
    three ways, and `K < 512` goes to run_small_k, which is VL=128 and exact for
    K in {128, 256, 384}. A `K % 512` guard covering the whole ks<=1 arm killed
    that path -- no K satisfies K % 128 == 0 && K % 512 == 0 && K < 512, so
    run_small_k became dead code and 12,288 (N,K) pairs began to throw while
    vllm's twin still accepted them.

    Generalisation for the next guard: enumerate the dispatchers BELOW it,
    including branches inside the kernel body, not just the host ladder.
    """
    path = _SGL / "xpu/esimd_kernels/resadd_norm_gemv_int4.h"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())

    # run_small_k must still be reachable: the guard has to admit K < 512.
    assert "run_small_k(n);" in c, "the small-K arm is gone from the dispatch"
    assert "if (ks <= 1 && K >= 512) {" in c, (
        "the VL=512 bound must be scoped to K >= 512; the ks<=1 arm also serves "
        "run_small_k at VL=128, which is exact for K in {128, 256, 384}"
    )

    def ks_of(n, k):
        ks = 1
        if n <= 128 and k >= 2048:
            ks = 8
        elif n <= 512 and k >= 2048:
            ks = 4
        elif n <= 512 and k >= 512:
            ks = 2
        while ks > 1 and k % (ks * 128) != 0:
            ks //= 2
        return ks

    def admitted(n, k):
        if k % 128:
            return False
        return not (ks_of(n, k) <= 1 and k >= 512 and k % 512)

    for k in (128, 256, 384):
        for n in (1, 128, 256, 512, 4096):
            assert admitted(n, k), (
                f"K={k} N={n} reaches run_small_k at VL=128 and was correct "
                "before the guard; it must not throw"
            )
    # ...and a bound with no live object behind it must not come back.
    assert "K <= 8192" not in c, (
        "K <= 8192 has no live object behind it; the kernel streams and its "
        "grid is K-independent"
    )


def test_gemv2_has_no_bound_without_a_live_object():
    """A numeric bound must name the live object that justifies it.

    The gemv2 kernel has no array, no SLM and a K-independent grid, so a
    `K <= 8192` bound has nothing behind it and refuses real hidden sizes.
    """
    seen = 0
    for root in (_VLLM, _SGL):
        path = root / "xpu/esimd_kernel.sycl"
        if not path.exists():
            continue
        # RAW, not code(): these needles are the TORCH_CHECK message strings, and
        # code() blanks string-literal bodies -- the same trap that made the
        # schema tests vacuous. File-scoped because a fixed window from the
        # function NAME lands on the binding rather than the definition in one
        # tree (the measured offset came out negative), and these messages are
        # unique to this op anyway.
        c = path.read_text()
        if "esimd_resadd_norm_gemv2_fp8_pert" not in c:
            continue
        seen += 1
        assert "resadd_norm_gemv2: K must be <= 8192" not in c, (
            f"{root.parents[0].name}: a K bound with no array behind it; the "
            "kernel streams with one reused accumulator"
        )
        # The real guard must survive.
        assert "resadd_norm_gemv2: K must be a multiple of 512" in c, (
            f"{root.parents[0].name}: the K % 512 guard is real -- the M==1 "
            "kernel hardcodes VL=512 with no tail"
        )
    assert seen, ("the anchor was not found in either tree; this test stopped\n                   checking anything")


def test_gemv2_w1_check_is_above_the_m_dispatcher():
    """Both kernels index w1 with w0's K, so the check is unconditional.

    Placed below the `if (M > 1)` return it sat inside the M==1 branch, leaving
    the M-tile path -- the one that actually does the shared-K indexing --
    unchecked: w0 [128,4096] with w1 [128,2048] overreads 262144 bytes past w1 at
    any M >= 2 and fills all of o1 from garbage.
    """
    path = _SGL / "xpu/esimd_kernel.sycl"
    if not path.exists():
        pytest.skip("sglang tree not present")
    c = code(path.read_text())
    i = c.index("esimd_resadd_norm_gemv2_fp8_pert(")
    body = c[i:i + 4000]
    w1_at = body.index("w1.size(1) == K")
    m_at = body.index("if (M > 1) {")
    assert w1_at < m_at, (
        "the w1 invariant is below the M dispatcher, so the M-tile path -- the "
        "one that indexes w1 with w0's K -- never reaches it"
    )


_FP8_PERT_FILES = [_VLLM / "xpu/esimd_kernels/fp8_GEMM_pert.h",
                   _SGL / "xpu/esimd_kernels/fp8_GEMM_pert.h"]


@pytest.mark.parametrize("path", _FP8_PERT_FILES, ids=_ids)
def test_k_threads_walk_is_not_undone_by_a_later_write(path):
    """A guard expressed as a computation is only as good as the last write.

    The walk `while (kt > 1 && K % (kt*64)) kt--;` established the kernel's
    invariant and the very next line, `if (kt == 3) kt = 2;`, broke it: at
    M=8 N=2816 K=576 (gemma-3-4b hidden) the walk left kt=3, satisfying K % 192,
    and the fixup forced 2, whose K % 128 is violated. k_per_thread became 288,
    the 64-strided loop with no tail walked 320, and [288,320) was accumulated by
    BOTH threads through SLM -- a wrong answer on every channel, 145,856 pairs.

    Halving keeps the walk inside the ladder's {4,2,1} arm set, so nothing can be
    launched that the walk did not validate.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())

    # The fixup must precede the walk, and the walk must halve.
    assert "if (k_threads == 3) k_threads = 2; while (k_threads > 1 && " \
           "(K % (k_threads * 64) != 0)) k_threads /= 2;" in c, (
        "the 3->2 fixup must come BEFORE the divisibility walk, and the walk "
        "must halve rather than decrement; otherwise the fixup undoes the "
        "invariant the walk just established"
    )
    assert "k_threads * 64) != 0)) k_threads--;" not in c, (
        "a decrementing walk can leave k_threads == 3, which the fixup then "
        "rewrites to 2 without rechecking divisibility"
    )

    def select(n, k):
        n_wgs = (n + 15) // 16
        kt = max(1, min(4, 640 // max(n_wgs, 1)))
        if kt == 3:
            kt = 2
        while kt > 1 and k % (kt * 64) != 0:
            kt //= 2
        return kt

    bad = []
    for n in range(17, 4097, 1):
        for k in range(64, 16385, 64):
            kt = select(n, k)
            if k % (kt * 64) != 0 or kt not in (1, 2, 4):
                bad.append((n, k, kt))
                if len(bad) > 3:
                    break
        if len(bad) > 3:
            break
    assert not bad, (
        f"the selector still emits a k_threads whose K % (kt*64) != 0, or one "
        f"with no ladder arm: {bad[:4]}"
    )


_MOE_ACCUM = [_VLLM / "moe_batch/moe.sycl", _SGL / "moe_batch/moe.sycl"]


@pytest.mark.parametrize("path", _MOE_ACCUM, ids=_ids)
def test_moe_accumulate_covers_the_tail_its_gate_admits(path):
    """`hidden_size / 64` truncates and the 64-wide store has no tail.

    The dispatch gate admits hidden % 32, and 128 of the 256 values it admits are
    not multiples of 64 -- 32, 96, 160, 224, 288, ... -- so half the admitted
    range left [(hidden/64)*64, hidden) of every output row unwritten. It could
    not fail loudly: the destination is a 4-deep torch::empty ring, so the tail
    read back as uninitialised memory on the first call and stale data from four
    calls ago after. The repo's own test_moe_decode_gelu_tanh.py drives (288, 64)
    through an unguarded caller.

    The invariant was in the file -- `hidden_size % 64 == 0` on 4 of 8 references
    -- just never on the callers that needed it. A guard on the entry point you
    were pointed at does not protect the kernel; the durable place is inside the
    launcher, or better, a tail arm so no caller has to care.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())

    assert "const int num_chunks = hidden_size / 64;" in c, (
        "the accumulate chunking changed -- re-derive this test"
    )
    # A tail arm must exist, be submitted, and be reached exactly when needed.
    assert "MoeAccumulateTail" in c, (
        "no tail arm: hidden % 64 != 0 leaves the last 32 elements of every row "
        "at whatever the reused torch::empty ring held"
    )
    assert re.search(
        r"const int tail_lanes = hidden_size - tail_base;", c), (
        "tail_lanes must be derived from hidden_size, not assumed"
    )
    assert re.search(r"if \(tail_lanes > 0\) \{ submit_kernel\(cgf_tail,", c), (
        "the tail lambda is defined but never submitted -- defining a predicate "
        "and not using it is the shape this suite keeps failing on"
    )
    # The tail must load/store 32 lanes, matching what the %32 gate guarantees.
    assert "block_store<fp16, 32>(final_output" in c, (
        "the tail must write 32 lanes; the gate guarantees hidden % 32 == 0"
    )
    # ...AT tail_base. Everything above holds with `const int tb = 0;` and the
    # `+ tb` dropped from the load: the tail is derived, submitted and 32 wide,
    # and it recomputes the FIRST 32 lanes while the real tail stays whatever
    # the reused ring held. Pin the offset, not just the width.
    # tb is an alias; pinning it leaves tail_base free. `tail_base =
    # num_chunks * 64 * 0` keeps every needle here verbatim and makes
    # tail_lanes = hidden_size, so the tail arm submits for every hidden size
    # and re-accumulates lanes [0,32) on top of the main arm, doubling the first
    # 32 output channels. Pin the operands the pinned text reads.
    for operand in ("tail_base", "num_chunks", "tail_lanes"):
        assert_single_write(c, operand, f"moe accumulate: {operand}")
    # Counting writes is not enough when the write itself carries the error:
    # `tail_base = num_chunks * 64 * 0` is one assignment and makes
    # tail_lanes = hidden_size, submitting the tail for every hidden size.
    # Pin the three RHS exactly.
    for name, want in (("num_chunks", "hidden_size / 64"),
                       ("tail_base", "num_chunks * 64"),
                       ("tail_lanes", "hidden_size - tail_base")):
        m = re.search(rf"const int {name}\s*=\s*([^;]+);", c)
        assert m, f"moe accumulate: {name} is no longer computed"
        got = " ".join(m.group(1).split())
        assert got == want, (
            f"moe accumulate: {name} is {got!r}, not {want!r}; any extra term "
            "changes which lanes the tail covers while every needle below "
            "stays verbatim"
        )
    assert re.search(r"const int tb = tail_base;", c), (
        "the tail's base is no longer tail_base, so it addresses the wrong "
        "32 lanes"
    )
    assert re.search(r"block_store<fp16, 32>\(final_output \+ \(size_t\)token \* "
                     r"hidden_size \+ tb,", c), (
        "the tail stores at the wrong offset; it must write [tail_base, "
        "hidden_size)"
    )
    assert re.search(r"block_load<fp16, 32>\(\s*partials \+ \(size_t\)\(token \* "
                     r"rows_per_token \+ b\) \* hidden_size \+ tb\)", c), (
        "the tail reads at the wrong offset; dropping `+ tb` makes it "
        "re-accumulate the first 32 lanes"
    )

    # And the arithmetic must cover every shape the gate admits.
    for hidden in range(32, 8193, 32):
        n_chunks = hidden // 64
        tail = hidden - n_chunks * 64
        assert n_chunks * 64 + tail == hidden, hidden
        assert tail in (0, 32), (
            f"hidden={hidden} leaves a {tail}-lane tail; a 32-wide arm cannot "
            "cover it, so the gate and the tail width disagree"
        )


def test_bmg_gemv_is_reachable_in_both_trees():
    """sglang shipped fp8_GEMV_bmg.h that nothing included.

    The file was byte-identical to vllm's except one comment, kernel_ops.h
    DECLARED esimd_gemv_fp8_pert_bmg, and nothing defined, bound or included it
    -- so half of the commit that added it was dead code, and every K the v2
    ladder can only split narrowly ran on a narrow VL instead of bmg's
    VL_BIG=256 + masked tail. A declaration with no definition is the shape
    that hid it: the header promised an op the library never exported.
    """
    checked = 0
    for tree, main in ((_VLLM, "xpu/esimd_kernel.sycl"),
                       (_SGL, "xpu/esimd_kernel.sycl")):
        hdr = tree / "xpu/esimd_kernels/fp8_GEMV_bmg.h"
        if not hdr.exists():
            continue
        checked += 1
        raw = (tree / main).read_text()
        src = code(raw)
        # A COMMENTED-OUT include still contains the string, so a raw needle
        # passed while the kernel was compiled by nothing -- exactly the state
        # this test detects. code() would blank the literal (the filename lives
        # in one), so match the directive on comment-free lines instead.
        live_includes = [l for l in raw.splitlines()
                         if l.lstrip().startswith("#include")]
        assert any("fp8_GEMV_bmg.h" in l for l in live_includes), (
            f"{tree}/{main} does not include fp8_GEMV_bmg.h, so the kernel is "
            "compiled by nothing"
        )
        assert "GEMV_fp8_pert_bmg_host(" in src, (
            f"{tree}: nothing calls GEMV_fp8_pert_bmg_host"
        )
        # The redirect must be live, not merely present.
        v2 = code((tree / "xpu/esimd_kernels/fp8_GEMV_v2.h").read_text())
        assert_live(v2, "GEMV_fp8_pert_bmg_host(p_in, p_w, p_sc, p_out",
                    f"{tree}: the bmg redirect")
    # Every assertion above lives inside the loop, past a `continue`. Deleting
    # the header in both trees emptied it and this test passed having examined
    # nothing -- confirmed by mutation, which is the exact failure the test was
    # written to detect, in the test that detects it.
    #
    # Count against the trees that are PRESENT, not a literal 2: reviewers
    # Count the trees that actually carry the file rather than hardcoding 2:
    # on a single-tree checkout the twin is legitimately absent, and failing
    # there reports something other than the contract.
    present = sum((t / "xpu/esimd_kernels/fp8_GEMV_bmg.h").exists()
                  for t in (_VLLM, _SGL))
    assert present, "fp8_GEMV_bmg.h is in neither tree"
    assert checked == present, (
        f"examined {checked} of {present} trees that have fp8_GEMV_bmg.h"
    )


_NO_8192 = [
    (_VLLM / "xpu/esimd_kernels/norm_gemv_norm_fp16.h", "K"),
    (_VLLM / "xpu/esimd_kernels/accum_norm_add_norm.h", "K"),
    (_VLLM / "xpu/esimd_kernels/scaled_resadd_norm_gemv_fp8.h", "K"),
    (_SGL / "xpu/esimd_kernels/norm_gemv_norm_fp16.h", "hidden_size"),
]


@pytest.mark.parametrize("path,var", _NO_8192, ids=lambda x: getattr(x, "name", x))
def test_ladder_sites_carry_no_bound_without_a_live_object(path, var):
    """A K bound needs a live object behind it.

    MAX_CHUNKS survives only as a template parameter that never appears in the
    kernel body, which walks `n_chunks = K / VL` with a runtime loop bound, and
    sglang's cited res_chunks by name in a file that contains no such array.

    Enumerated over K % 64 == 0 in [64, 65536]: the ladder's VL always divides
    K, so % 64 is necessary AND sufficient. The `<= 8192` bound rejected 896 of
    those, including 14336 (Llama-3-8B FFN), 22016 and 28672 (Llama-3-70B FFN);
    the redundant % 256 clause rejected a further 48 in (4096, 8192], including
    5504 (Llama-2-7B / Mistral-7B FFN at TP=2 -- this repo calls 11008 the
    unsharded value at resadd_norm_gemv_int4.h:521). Both only over-reject, and the
    callers are bare, so an unusual hidden size crashed on model load.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    assert f"{var} <= 8192" not in c, (
        f"{path.name}: the <= 8192 bound is back; nothing in this kernel is "
        "sized by it -- MAX_CHUNKS never reaches the body"
    )
    assert f"{var} % 512 == 0 || {var} % 256 == 0" not in c, (
        f"{path.name}: the redundant % 256 clause is back; the % 64 check "
        "below it is strictly weaker and the ladder covers every multiple of 64"
    )
    assert f"{var} % 64 == 0" in c, (
        f"{path.name}: the % 64 check is gone -- that one is load-bearing"
    )


_PATCHES = [_ROOT / "sglang/patches/sglang_for_multi_arc.patch",
            _ROOT / "vllm/patches/vllm_for_multi_arc.patch"]


@pytest.mark.parametrize("path", _PATCHES, ids=lambda p: p.name)
def test_patch_hunk_headers_match_their_bodies(path):
    """`git apply` without --recount rejects a patch whose counts disagree.

    sglang/docker/Dockerfile:108 applies this with `git apply
    --whitespace=nowarn` and no --recount, so a wrong count is not cosmetic --
    it fails the image build with "corrupt patch at line N". Four hunks on this
    branch had drifted (main applies cleanly), which means every sglang docker
    build from this branch was broken and nothing in the suite noticed.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    lines = path.read_text().splitlines()
    bad, i = [], 0
    while i < len(lines):
        m = re.match(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", lines[i])
        if not m:
            i += 1
            continue
        want_old = int(m.group(2) or 1)
        want_new = int(m.group(4) or 1)
        j, old, new = i + 1, 0, 0
        while j < len(lines):
            c = lines[j][:1]
            if lines[j].startswith(("@@", "diff ", "--- ", "index ")):
                break
            if c == "-":
                old += 1
            elif c == "+":
                new += 1
            elif c in (" ", ""):
                old += 1
                new += 1
            elif c == "\\":
                pass
            else:
                break
            j += 1
        if (old, new) != (want_old, want_new):
            bad.append(f"line {i + 1}: header -{want_old}/+{want_new}, "
                       f"body -{old}/+{new}")
        i = j
    assert not bad, (
        f"{path.name}: hunk headers disagree with their bodies, so "
        "`git apply` (no --recount) reports a corrupt patch and the docker "
        "build fails: " + "; ".join(bad[:4])
    )


_PY_PKGS = [
    (_ROOT / "sglang/custom-esimd-kernels/python/custom_esimd_kernels_sglang",
     _SGL / "xpu/torch_extension.cc"),
    (Path(__file__).resolve().parents[1] / "python/custom_esimd_kernels_vllm",
     _VLLM / "xpu/torch_extension.cc"),
]


@pytest.mark.parametrize("pkg,binding", _PY_PKGS, ids=["sglang", "vllm"])
def test_every_exported_name_has_a_wrapper(pkg, binding):
    """A name in _EXPORTS with no wrapper is an AttributeError on import.

    The mirror of this also bit: sglang bound esimd_gemv_fp8_pert_bmg in C++ and
    never wrapped it, so the op existed only as torch.ops.* and the package API
    silently lacked it while vllm's had it. A half-wired op reads as wired.
    """
    if not pkg.exists():
        pytest.skip(f"{pkg} not present")
    init = (pkg / "__init__.py").read_text()
    m = re.search(r"_EXPORTS = \[(.*?)\n\]", init, re.S)
    if not m:
        pytest.skip("no _EXPORTS list in this package")
    names = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert names, "_EXPORTS is empty -- re-derive this test"
    defined = {n.name for n in ast.walk(ast.parse((pkg / "ops.py").read_text()))
               if isinstance(n, ast.FunctionDef)}
    missing = sorted(names - defined)
    assert not missing, (
        f"{pkg.name}: exported but never defined in ops.py, so importing the "
        f"package raises AttributeError: {missing}"
    )

    # And the OTHER direction, which is the one that actually bit: sglang bound
    # esimd_gemv_fp8_pert_bmg in C++ and never exported it, so the op existed
    # only as torch.ops.* while vllm's package API had it. Asserting only
    # _EXPORTS subset ops.py pins the reverse of that defect -- deleting both
    # the export and the wrapper, leaving the binding, passed.
    if not binding.exists():
        return
    bound = set(re.findall(r'm\.impl\("(\w+)"', binding.read_text()))
    assert bound, "no m.impl bindings found -- re-derive this test"
    # NOT "every bound op must be exported": the integration patches call many
    # ops as torch.ops.custom_esimd_kernels_sglang.* directly, which is a
    # working convention. The defect shape is a HALF-wired op -- a wrapper in
    # ops.py that the package API does not offer.
    wrapped_and_bound = defined & bound
    unexported = sorted(wrapped_and_bound - names)
    assert not unexported, (
        f"{pkg.name}: these ops are bound in {binding.name} AND wrapped in "
        f"ops.py but missing from _EXPORTS, so the package API silently lacks "
        f"an op it has a wrapper for: {unexported}"
    )
    # Honest limit: deleting BOTH the wrapper and the export still passes here,
    # because that is indistinguishable from the torch.ops-only convention nine
    # other sglang ops legitimately use. Distinguishing them needs a policy this
    # repo does not have.


@pytest.mark.parametrize("path", [_VLLM / "xpu/esimd_kernels/fp8_GEMM_pert.h",
                                  _SGL / "xpu/esimd_kernels/fp8_GEMM_pert.h"],
                         ids=_ids)
def test_ws_dispatch_never_hands_the_kernel_a_k_below_its_vector_length(path):
    """The WS tail computes `K - VL`; a hardcoded VL makes that negative.

    GEMM_fp8_pert_dispatch had three arms calling ws_gemm_fp8_pert_host<128,*>
    with no K test, while the sibling ladder in the same file selects VL=64 for
    K % 128 != 0 and documents "supports any K>=64". At K < 128 the tail offset
    went negative and both block_loads read below their buffers -- silently,
    because the overlap zeroing still produces the exact answer.

    Two things must hold: every ws call site picks VL from K, and the host
    itself refuses K < VL. The host check is the durable one -- four call sites
    feed it, and the next arm added would miss a check placed at the dispatcher.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())

    i = c.find("inline void ws_gemm_fp8_pert_host")
    assert i >= 0, "ws_gemm_fp8_pert_host not found -- re-derive this test"
    assert re.search(r"TORCH_CHECK\(K >= \(uint32_t\)VL", c[i:i + 1200]), (
        "ws_gemm_fp8_pert_host does not refuse K < VL; the tail offset K - VL "
        "goes negative and reads below both buffers"
    )

    # And no caller may hardcode the wide VL without testing K.
    seen = 0
    for m in re.finditer(r"ws_gemm_fp8_pert_host<(\d+), *\d+>", c):
        seen += 1
        if int(m.group(1)) != 128:
            continue
        # Scope the lookback to THIS arm. A fixed 220-char window reaches back
        # into the neighbouring `else if` and is excused by ITS K % 128 test --
        # verified: re-hardcoding VL=128 on the M>64 arm passed under the wide
        # window. Cut at the nearest brace or `else`, whichever is closer.
        head = c[max(0, m.start() - 220):m.start()]
        for sep in ("{", "}", "else if", "else"):
            at = head.rfind(sep)
            if at >= 0:
                head = head[at + len(sep):]
        assert "K % 128 == 0" in head, (
            f"a ws call site hardcodes VL=128 without testing K % 128 in its "
            f"own arm: ...{c[max(0, m.start() - 90):m.start()]}"
        )
    assert seen, "no ws_gemm_fp8_pert_host call sites found"

    # Every K must reach an arm. The kernel is VL-generic, so a narrower arm
    # computes K < 64 exactly and stopping the ladder at VL=64 refuses shapes
    # for no reason. Simulate the ladder.
    widths = sorted({int(w) for w, _ in re.findall(
        r"ws_gemm_fp8_pert_host<(\d+), *(\d+)>", c)}, reverse=True)
    assert min(widths) <= 16, (
        f"the ws ladder stops at VL={min(widths)}; every K in [{min(widths)}, "
        "63) that a narrower arm computes exactly is refused instead"
    )
    unserved = [k for k in range(min(widths), 512)
                if not any(k % 128 == 0 and w == 128 or w != 128 and k >= w
                           for w in widths)]
    assert not unserved, (
        f"these K reach no arm: {unserved[:6]}"
    )


def test_env_tuned_vector_width_is_clamped_to_the_ladder():
    """An unvalidated atoi reaches `kpt % vl` as a host-side divide by zero.

    SGLANG_GEMV_VL_CAP was read with std::atoi and used directly as the initial
    vl. atoi returns 0 both for "0" and for any unparseable string, and 0 hits
    the walk's `kpt % vl` -- SIGFPE on the host, before any TORCH_CHECK can
    report anything. Every other out-of-range value survives the walk intact
    and lands on the ladder's terminal refusing arm, which is loud and correct.
    Only 0 faults, and only a clamp prevents it.
    """
    path = _SGL / "xpu/esimd_kernels/fp8_GEMM_pert.h"
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    i = c.find('std::getenv("")')
    assert i >= 0, "the env cap is no longer read here -- re-derive this test"
    body = c[max(0, i - 200):i + 600]
    for w in (32, 64, 128, 256, 512):
        assert f"case {w}:" in body, (
            f"the cap no longer accepts {w}; it must admit exactly the widths "
            "the walk can terminate on and the ladder has arms for"
        )
    assert "default:" in body and "return 256;" in body, (
        "an out-of-range cap is not rejected; atoi(0) then reaches kpt % vl "
        "as a divide by zero"
    )
    # An ordinary `case 0: return v;` satisfies every needle above and restores
    # the SIGFPE. The accepted widths share one `return v;`, so exactly one may
    # exist -- a second is an escape hatch for a value the switch is meant to
    # reject.
    n_ret_v = len(re.findall(r"return v;", body))
    assert n_ret_v == 1, (
        f"the cap has {n_ret_v} `return v;` arms; only the accepted "
        "{32,64,128,256,512} group may return the env value unchanged"
    )
    # Counting `return v;` is not enough: `case 0: return 0;` restores the
    # SIGFPE while returning a literal. Pin the accepted case labels exactly --
    # any label outside the five widths is an escape hatch for a value the
    # switch exists to reject.
    # The clamp is on how default_vl is PRODUCED; nothing pinned that it is
    # what the walk CONSUMES. One line after `vl = default_vl;` --
    # `if (getenv("SGLANG_GEMV_VL_RAW")) vl = atoi(z);` -- restores the SIGFPE
    # with every needle below intact. Between `vl = default_vl` and the walk,
    # vl may only be set to a literal (the K<512 / K==512 / ks arms).
    fn = c[c.find("inline void select_vl_ks("):]
    fn = fn[:fn.find("while (vl > kpt")]
    at = fn.find("vl = default_vl")
    assert at >= 0, "select_vl_ks no longer seeds vl from the env cap"
    for m in re.finditer(r"\bvl\s*=\s*([^;]+);", fn[at + 5:]):
        rhs = m.group(1).strip()
        assert rhs.isdigit(), (
            f"vl is assigned {rhs!r} between the clamp and the walk; only a "
            "literal width may override the clamped cap, or an unvalidated "
            "value reaches `kpt % vl` as a host divide-by-zero"
        )

    labels = {int(x) for x in re.findall(r"case (\d+):", body)}
    assert labels == {32, 64, 128, 256, 512}, (
        f"the cap accepts {sorted(labels)}; only the widths the walk can "
        "terminate on and the ladder has arms for may be admitted"
    )


@pytest.mark.parametrize("path", _FP8_GEMV, ids=_ids)
def test_the_correctness_redirect_is_not_behind_a_debug_knob(path):
    """DISABLE_BMG_GEMV must not be able to turn off the only correct path.

    For K % 32 != 0 the v2 ladder cannot split K at all -- every arm leaves
    kpt % vl != 0 and the walk ends at the TORCH_CHECK -- so the bmg redirect
    is the only path that computes those shapes. The env var is a PERFORMANCE
    selector, so it must not gate that clause: every admitted K % 32 != 0 would
    abort mid-forward at a bare call site.

    The performance clause (K in 1024..2048, K % 256 != 0) may stay behind it.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    c = code(path.read_text())
    i = c.find("K % 128 != 0 || (K < 512 && K % 256 != 0)")
    assert i >= 0, "the correctness redirect is gone -- re-derive this test"
    # Walk outward from the clause and collect every enclosing condition.
    head, depth, chain = c[:i], 0, []
    for t in reversed(list(re.finditer(
            r"\}|if\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*\{|\{", head))):
        tok = t.group(0)
        if tok == "}":
            depth += 1
        elif depth:
            depth -= 1
        elif t.group(1) is not None:
            chain.append(t.group(1))
    offenders = [cond for cond in chain if "_disable_bmg" in cond]
    assert not offenders, (
        "the K % 32 != 0 correctness redirect is inside a block guarded by "
        f"{offenders}; a debug knob must not switch off the only correct path"
    )
