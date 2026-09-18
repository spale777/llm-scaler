"""Contract guards for the custom IPC all-reduce.

Every failure mode here is silent: a no-op `out.copy_(inp)` fallback leaves each
rank holding its own partial sum, and a colliding dtype code, a default-selector
queue, a one-shot sync flag or a dropped tail all produce fluent wrong output
with no error and no NaN. Each test pins one property that keeps the reduction
either correct or refused outright.
"""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_AR_CC = Path(__file__).resolve().parents[1] / "csrc/xpu/torch_extension_ar.cc"
_AR_SYCL = Path(__file__).resolve().parents[1] / "csrc/xpu/ipc_allreduce.sycl"
_PATCH = _ROOT / "vllm/patches/vllm_for_multi_arc.patch"


def _code(path: Path) -> str:
    """Source with comments stripped, so prose describing a construct cannot
    stand in for the construct."""
    src = path.read_text()
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def test_no_silent_unreduced_fallback():
    src = _code(_AR_CC)
    m = re.search(r"out\.copy_\(inp\);\s*\n\s*return;", src)
    assert m is None, (
        "all_reduce_reg returns the input unreduced; a no-op all-reduce must "
        "never be a legal outcome -- decline at init instead"
    )
    assert "TORCH_CHECK(ctx->ready" in src, "missing the readiness assertion"


def test_queue_comes_from_the_tensors_device():
    src = _code(_AR_CC)
    assert "gpu_selector_v" not in src, (
        "a fresh gpu_selector_v queue picks the default GPU, so every rank lands "
        "on device 0 and is unordered against the producing matmul"
    )
    assert "getCurrentXPUStream(inp.device().index())" in src


def test_dtype_is_validated_not_assumed():
    src = _AR_CC.read_text()
    assert "unsupported dtype" in src, "dtype must be checked"
    for dt in ("Half", "BFloat16", "Float"):
        assert dt in src, f"{dt} missing from the dtype mapping"

    # Distinct, not merely present: a collision reduces one dtype as the other.
    codes = dict(re.findall(
        r"case at::ScalarType::(\w+):\s*return\s+(-?\d+);", src))
    for dt in ("Half", "BFloat16", "Float"):
        assert dt in codes, f"{dt} no longer maps to a literal code"
    vals = [codes[dt] for dt in ("Half", "BFloat16", "Float")]
    assert len(set(vals)) == 3, (
        f"the dtype codes collide: {codes}; two dtypes sharing a code means one "
        "is reduced as the other, silently"
    )
    assert "-1" not in vals, "a supported dtype maps to the unsupported sentinel"


def test_mutation_is_declared_in_the_schema():
    src = _AR_CC.read_text()
    m = re.search(r'm\.def\("all_reduce_reg\(([^"]*)\)', src)
    assert m, "all_reduce_reg schema not found"
    assert "Tensor(b!) inp" in m.group(1), (
        "inp is mutated but not declared mutable; torch.compile may CSE or reuse "
        "it and silently miscompile"
    )


def test_world_size_is_bounded():
    src = _AR_CC.read_text()
    # A bare "kMaxPeers" survives `<= 1024 || kMaxPeers`, so pin the compare.
    assert re.search(r"handles\.size\(\)\)?\s*<=\s*kMaxPeers", src), (
        "world_size is no longer compared against kMaxPeers; remote_slots and "
        "remote_flags are fixed kMaxPeers arrays, so a larger world writes past "
        "them"
    )
    assert not re.search(r"<=\s*kMaxPeers\s*\|\||\|\|\s*.{0,40}kMaxPeers", src), (
        "the peer-limit check has been softened with a disjunction"
    )


def test_kernel_has_scoped_fences_not_volatile():
    src = _code(_AR_SYCL)
    full = _AR_SYCL.read_text()
    assert "fence_scope::system" in full, "release fence missing"
    assert "fence_scope::system_acquire" in full, "acquire fence missing"
    assert "volatile" not in src, (
        "volatile constrains the compiler, not the memory system; an Intel GPU "
        "L1 is not coherent with incoming peer writes"
    )


def test_flags_are_sequence_numbered():
    src = _AR_SYCL.read_text()
    # A bare "seq" matches the parameter name, which `if (true) break;`
    # leaves intact; pin the comparison that does the waiting.
    assert re.search(r"got\[0\]\s*>=\s*seq", src), (
        "the peer wait no longer compares the observed flag against seq; a "
        "one-shot flag cannot be reused across calls or under graph replay"
    )
    assert "while(sync_flags" not in src and "== 0) { }" not in src, (
        "a one-shot 0/1 flag cannot be reused across calls or under graph replay"
    )


def test_tail_is_not_dropped():
    src = _AR_SYCL.read_text()
    # `simd_mask<kVL> live = 1;` keeps the needle, so pin the predicate.
    assert re.search(r"simd_mask<kVL>\s+live\s*=\s*lane\s*<", src), (
        "the tail mask is no longer derived from the element count, so the "
        "final partial vector is written in full"
    )
    assert "elements / VL" not in src, (
        "integer division of the element count silently drops the tail"
    )


def test_push_and_reduce_offsets_agree():
    """Rank r writes slot r everywhere and reduces all slots of its own buffer."""
    src = _AR_SYCL.read_text()
    assert "remote_slots" in src, "peer slot bases missing"
    assert "p * slot_stride" in src, (
        "the reduction must walk per-rank slots that the push actually wrote"
    )


def test_patch_does_not_advertise_an_unbuilt_allreduce():
    if not _PATCH.exists():
        pytest.skip("patch not present")
    src = _PATCH.read_text()
    assert "Changed for Custom IPC All-Reduce" not in src, (
        "the flag advertises a custom all-reduce that is not wired up; "
        "torch.ops.vllm.all_reduce routes to oneCCL regardless"
    )


def test_peer_memory_is_opened_against_the_queues_context():
    """A pointer opened against a different context faults inside a kernel.

    Every rank must map through the context its own queue runs on, which is
    the one PyTorch's XPU runtime owns. Creating a fresh context here is the
    natural shortcut and yields peer pointers that are accepted at map time
    and fail at first use.
    """
    src = _code(_AR_CC)
    i = src.find("void register_buffer_ctx")
    assert i >= 0, "register_buffer_ctx not found; re-derive this test"
    body = src[i:src.find("std::string export_buffer_handle", i)]
    assert "getCurrentXPUStream" in body and "get_context()" in body, (
        "peer buffers must be opened against the current stream's context"
    )
    assert "sycl::context(" not in body, (
        "a freshly constructed context gives peer pointers that fault in a kernel"
    )


def test_peer_probe_checks_atomics_over_ordered_pairs():
    """The flag handshake does atomics on peer memory.

    access_supported is not enough: concurrent atomic modify is undefined when
    the device denies atomics for that pair. Peer access is one-directional,
    so a world of N needs N*(N-1) checks and a loop that skips the reverse
    direction passes on a topology that only works one way.
    """
    src = _code(_AR_CC)
    i = src.find("bool peer_access_supported")
    assert i >= 0, "the peer probe is missing"
    body = src[i:i + 1200]
    assert "atomics_supported" in body, (
        "the probe accepts access without atomics, which the flag protocol needs"
    )
    # Two nested loops over the device list, not a triangular single pass.
    assert body.count("for (") >= 2, (
        "the probe must cover ordered pairs in both directions"
    )


def test_registration_cannot_silently_succeed():
    """ctx->ready gates the collective; a no-op registration must not set it.

    The schema-compatible register_buffer overload cannot reach the context,
    so it refuses rather than returning while leaving the caller believing the
    peers were mapped.
    """
    src = _code(_AR_CC)
    i = src.find("void register_buffer(")
    assert i >= 0, "register_buffer not found"
    body = src[i:src.find("void register_buffer_ctx", i)]
    assert re.search(r"TORCH_CHECK\(\s*false\s*,", body), (
        "the context-free overload must refuse, not return silently"
    )
    ctx_i = src.find("void register_buffer_ctx")
    ctx_body = src[ctx_i:src.find("std::string export_buffer_handle", ctx_i)]
    assert "ctx->ready = true;" in ctx_body, (
        "the real registration must mark the context ready"
    )
    assert ctx_body.index("ctx->ready = true;") > ctx_body.rindex("ipc::open("), (
        "ready is set before every peer is mapped"
    )


def test_mapped_peers_are_released():
    """IPC handles are file descriptors and leak one per peer per process."""
    src = _code(_AR_CC)
    i = src.find("void dispose_ctx")
    assert i >= 0, "dispose_ctx not found; mapped peers are never released"
    body = src[i:i + 900]
    assert "ipc::close(" in body, "dispose_ctx does not close the peer mappings"
    assert "ctx->ready = false;" in body, (
        "a disposed context must not still admit collectives"
    )
