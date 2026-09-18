# LLM-Scaler Arc Pro B70 Optimization Manifest

Status record for the B70 / 8-GPU-per-switch optimization effort.

**This document previously asserted unbuilt work in the past tense.** It has been
restructured so that state is explicit. Hardware facts live in
`architecture_hardware_reference.md`; every figure there is sourced.

## 0. How to read this

Four independent axes. A component can be fully implemented and still do nothing.

| Axis | Meaning |
|---|---|
| **BUILT** | Code exists and is complete |
| **WIRED** | Something actually calls it at runtime |
| **CORRECT** | Verified by reading; no known defect |
| **MEASURED** | Benchmarked on B70 hardware |

**Nothing in this repo is MEASURED.** The development host has no Intel GPU
(`/dev/dri` absent, no Arc on the PCI bus, `sycl-ls` shows OpenCL CPU only). AOT
compilation works; execution and measurement do not. Every performance claim below is a
prediction.

## 1. Component status

| Component | BUILT | WIRED | CORRECT | Notes |
|---|:---:|:---:|:---:|---|
| `fp8_GEMM_blockscale.h` (dense) | ✅ | vLLM only | ⚠️ | sglang copy is byte-identical but has **zero callers** |
| `fp8_moe_gemm_blockscale.h` (MoE decode) | ✅ | ✅ | ⚠️ | D7 unbounded write; likely GRF spill (K5) |
| `fp8_moe_gemm_blockscale.h` (DPAS prefill) | ✅ | vLLM only | ✅ | VNNI2 pack verified bit-perfect over all 256 combinations |
| Custom IPC all-reduce | ✅ | ❌ | ❌ | Rewritten to fail loudly; `register_buffer` refuses because the Level Zero handle exchange is unbuilt, so the op is unreachable. Never executed. See §3 |
| DeepSeek V4.1 kernels | ✅ | ⚠️ | ❌ | Tracked and built. FP4 LUT and router order corrected. The top-k op refuses: its kernel neither loads nor stores, and the group-limited noaux_tc stage is absent. `fp4_gemm.h` is a skeleton |
| MoE router rewrite (`acc[64]`) | ✅ | ✅ | ✅ | Reverted to four scalar accumulators. See K3 |
| Q4_K 2D load width 8→16 | ✅ | ✅ | ✅ | Reverted to width 8. See K1 |
| `lsc_prefetch` injection | ✅ | ✅ | ❌ | Three sites per tree on the decode GEMV weight streams. Unmeasured |
| Wide-router O(N) rewrite | ✅ | ✅ | ❌ | `range<1>(num_experts)` with `WIDE_TOK=8` token blocking. Coverage verified by exhaustive simulation; throughput unmeasured |
| PP=2 across switches | ❌ | ❌ | — | No `send`/`recv` code in any patch; every launch is `-pp=1` |
| oneCCL env tuning | ✅ | ❌ | ⚠️ | Set in one benchmark script; **no Dockerfile sets any of it** |

## 2. Defect register

Severity: **C**ritical (silent wrong results or build break) / **H**igh / **M**edium /
**L**ow. Every entry was verified by reading the named file.

### Kernel correctness

| ID | Sev | File:line | Defect |
|---|:---:|---|---|
| K1 | C | `sglang/.../moe_grouped/moe_q4k_ggemv.h:133` | 2D load width 8→16 changed the returned **row stride** to 16 B while the consumer still slices at 8 B. Every weight is paired with the **wrong scale/min**. Also: `k_base` advances by 8 while the load consumes 16 → iterations overlap; and reads 8 B past each row (hardware clamps to zeros, which enter dequant as real weights). Only half the loaded bytes are consumed, so it does not even buy bandwidth. Sibling `moe_q5k_down_ggemv.h:95` retains width 8 with an explicit row-layout comment, pinning the intended contract. **Revert to 8.** |
| K2 | C | `vllm/.../torch_extension_{gemm,lgrf,moe,q4_0,topk_v2}.cc` | `PyMODINIT_FUNC PyInit_*` deleted, replaced with nothing. `.so` has no init symbol → `ImportError`. Three are imported directly by `__init__.py:4-7`; `q4_0_quant_ops` from `vllm_for_multi_arc.patch:11558`. **Fix: `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}`** — `TORCH_EXTENSION_NAME` is confirmed defined by the vendored `esimd_build_extention.py:668,1112`. `torch_extension_q4_0.cc` also needs `#include <torch/extension.h>` added. |
| K3 | C | `vllm/.../moe_batch/moe.sycl:145,182` | `simd<float,64> acc[64]` = **16 KB GRF against an 8 KB budget** (128 GRF × 64 B; no large-GRF flag in `setup.py:166-176`). Worse: `acc[token]` with a **runtime index** cannot be register-allocated — the array goes to scratch with indirect addressing **even at n_tokens=1**, degrading the common decode path. The `n_tokens<=4` host guard (`moe.sycl:302`) means at most 4 entries are ever used, so the rewrite buys nothing. **Revert to the 4-accumulator form.** |
| K4 | H | `sglang/.../moe_batch/moe.sycl:2525` | `moe_forward_full_rtfused` calls the ≤4-token router kernel **directly**, bypassing the dispatcher that owns the guard. For `n_tokens>4`, tokens 4+ are never written — uninitialized `torch::empty` feeds softmax/top-k. Latent (`_MOE_FUSED_MAX_M` defaults to 1) but env-settable, and **became reachable when D6's shim forced this fallback to always run**. |
| K5 | H | `sglang/.../fp8_moe_gemm_blockscale.h` (DPAS prefill) | ~5.5 KB live state (`acc[8]` of `simd<float,128>` = 4096 B alone) with **no `-doubleGRF`** in `setup.py:79-82`, while sibling DPAS kernels pass it (`:214`). The comment at `:67` still says "no DPAS, standard compilation" — stale. Expect heavy spilling. |
| K6 | M | `.../fp8_moe_gemm_blockscale.h:59-66` | `build_active_experts` has **no bound on its write index**. Buffer can be as small as 1 element; kernel can write `num_experts` int32s. Safe only if `expert_idx` is a well-formed prefix sum. Device-side heap overflow with no diagnostic. |
| K7 | M | `vllm/.../moe_batch/moe.sycl:225-231` | Wide-router e4m3 edit is an **accidental inversion**: `acc += convert<float>(...)` → `acc = acc + xv*wv` downgrades a correct fp32 accumulator to fp16, and `fp16 s`→`float s`→`(fp16)s` is a no-op round-trip. Results now change **discontinuously at the n_tokens=4 dispatch boundary**. Revert. (The e5m2 narrow edit at `:182-196` moved the opposite, correct direction — keep it.) |
| K8 | M | `sglang/.../fp8_GEMM_blockscale.h:83,251,373` | `block_n` is a runtime parameter that is **accepted and silently ignored** — the kernel hardcodes `/128`. Unreachable today only because of `TORCH_CHECK(block_n==128 && block_k==128)`. N-block and K-block sizes are conflated in both dense and MoE kernels. **Blocks DeepSeek V4.1, which needs 32×32.** |
| K9 | M | `sglang/.../esimd_kernel_moe.sycl` | Missing the `weight_scale` shape `TORCH_CHECK` that the vLLM twin has (`esimd_kernel_gemm.sycl:118-123`). A transposed or `[E,1,1]` scale tensor is silently accepted and read OOB. |
| K10 | L | `.../fp8_GEMM_blockscale.h:41-49` | `fp8e4m3_to_fp16` is **254/256 exact** (verified exhaustively, including all subnormals and both zeros). The two NaN encodings `0x7F`/`0xFF` decode to **±480** instead of NaN — a corrupted weight byte yields a plausible finite number rather than propagating. Document the behavior. |

### Wider kernel surface

Covers `eagle/`, `esimd_kernels/*GEMV*.h`, `int4_*`, `fused_add_rms_norm*`, `resadd_norm*`,
`moe_ops.h`, `prefill_dpas.h`, `moe_int4.sycl`, `moe_prefill_*`, `moe_grouped/*`.

| ID | Sev | File:line | Defect |
|---|:---:|---|---|
| S1 | C | `fp8_GEMV_bmg.h:71-72`, `int4_GEMV.h:136`, `int4_GEMM.h:109,133`, `fused_add_rms_norm_batched.h:27` | **Silently dropped K-tail in four kernels.** `int tail = kp - kp_full;` is computed and **never used**. The comment at `:85-91` admits the kernel cannot handle a tail and says "host should not pick this template" — but nothing enforces it, and `fp8_GEMV_v2.h:378` **deliberately routes `K%64!=0` into it**. K=1044 silently discards the last 20 elements of every row; K=2816 leaves the last 256 elements as stale garbage in the output *and* excluded from `sum_sq`. Root cause is one design gap: `select_vl_ks` / `select_vl_ks_int4` / `select_bmg` fall through silently instead of falling back to a tail-correct kernel. **Four silent-wrong-answer bugs collapse into one missing `TORCH_CHECK` per entry point.** |
| S2 | C | `fused_add_rms_norm.h:101-113,175-197` | **Residual double-write corrupts layer carry state.** The overlapping tail window is correctly masked out of `sum_sq` but the full VL is still stored — writing `2h+r` instead of `h+r`. Compounds through every later layer. Also `k_tail = K - VL` has **no `K>=VL` guard** at six sites → negative offset → OOB write *before* the tensor base. |
| S3 | C | `resadd_norm_gemv_fused.h:92-94,119-121`; `resadd_norm_gemv_int4.h` (6 paths) | **Cross-work-group residual race** — work-group `n==0` writes `residual_ptr` in place while others still read it. Conclusive: `esimd_kernel.sycl:859-861` carries an explicit comment describing this exact hazard, and the kq/silu variants (`resadd_norm_gemv_kq.h:116`, `resadd_norm_gemv_q4k_silu.h:126`) correctly use a separate buffer. These files were never migrated. Nondeterministic, scheduling-dependent. |
| S4 | C | `moe_ops.h:471-473` | **OOB read/write in MoE scatter prefix.** Unconditional `block_load<int32_t,32>` when `num_experts % 32 != 0` — DeepSeek-V2-Lite has **60**; small MoEs have 4/6/8. Reads past the counts tensor, folds garbage into `running_sum`, stores past `num_experts` → tokens scattered to wrong rows. Live shape, not hypothetical. |
| S5 | H | `eagle.sycl:869,968,1087,1346,1704,1944,1963,1979,1995,2014,2043`; `onednn_w8a16/fp8_gemm_w8a16.h:94` | **Eleven eagle entry points always submit to device 0** — `getCurrentXPUStream()` with no device index while operating on another card's tensors. Every other file in the tree does it correctly, so this is oversight, not convention. `build_tree_kernel_efficient` and `verify_tree_greedy` are on the hot spec-decode path. Directly undermines this repo's multi-Arc purpose. |
| S6 | H | `moe_ops.h:637-646`; `moe_int4.sycl:1110` | Unbounded writes: `int ids[32]` with runtime `topk` and no check; `slm_init<16 floats>` commented "max 16 shared experts" with a runtime loop and no guard. |
| S7 | H | `resadd_norm_gemv_fused.h:78` | `simd<float,512> res_chunks[16]` with `n_chunks=K/512` → **K>8192 overruns the array**. Already **32 KB of registers** in the legal case against the 8 KB budget (see K3). |
| S8 | H | `moe_q4k/q5k/q6k_ggemv.h`, `moe_prefill_int4.sycl`, `int4_nmajor_gemm.h`, `q8_0_GEMV.h` | **32-bit byte offsets into gather/scatter** — 4 GiB ceiling. At prefill scale (hidden 8192 × >262k tokens) this wraps. Supersedes an earlier "non-issue" note that was scoped only to the narrow Q4_K case. |
| S9 | H | `int4_nmajor_gemm.h:311-322` | Comment says "Atomic add"; the code is a **plain gather+scatter**. Deterministic failure: when `(t1-t0)%MAX_M != 0` indices are clamped so multiple lanes in one scatter share an offset — duplicate-lane ordering is undefined, contributions lost. |
| S10 | M | `prefill_dpas.h:190` | `kv_surf_h = 0x3FFFFFU` is a **fabricated surface height that deliberately defeats hardware OOB clamping** on the KV cache. The `BLK_LOGICAL_CLAMP` macro above it exists because an unclamped index previously caused `UR_RESULT_ERROR_DEVICE_LOST`. |
| S11 | M | `prefill_dpas.h:44-58` | Hardcodes **104 KB SLM per work-group**; Xe2 typically exposes 64 KB (max 128 KB per `architecture_hardware_reference.md` §1, but the exposed per-WG limit is lower). If so the kernel **fails to launch** rather than running slowly. Verify with `clinfo`. |
| S12 | M | `qkv_split_norm_rope.h:62-64` | Silently `return`s for `headDim != 256` with no host `TORCH_CHECK` → **entirely uninitialized Q/K/V**, no diagnostic. |

### Distributed / IPC

| ID | Sev | File:line | Defect |
|---|:---:|---|---|
| A1 | C | `ipc_allreduce.sycl:19` vs `:41` | Write offset and read offset **disagree**. Peers all write the same `offset`; the reducer reads `p*elements + offset`. At TP=8 / 4096 elements this is a **128 KB out-of-bounds device read** past the activation tensor. |
| A2 | C | `ipc_allreduce.sycl:26-33` | `sync_flags` spun on but **never set by anyone**. Zero-initialized → infinite hang; garbage → barrier passes instantly and reads torn data. No atomic, no fence (the prior manifest claimed an `atomic_load` spinlock; there is none). |
| A3 | C | `torch_extension_ar.cc:29,34-35,41-44` | `remote_bufs` never populated; `register_buffer` is an empty body; `init_custom_ar` **discards the handles entirely** and nulls the array. `all_reduce_reg` therefore **always** returns `out = inp` — a no-op posing as an all-reduce, with no error, no log, no NaN. Root cause: the `std::vector<std::string> handles` signature was copied from CUDA. See §3. |
| A4 | H | `torch_extension_ar.cc:46-48` | `sycl::queue q(sycl::gpu_selector_v)` per call — picks the **default GPU, not the rank's device** (all 8 ranks likely select device 0); not the current XPU stream, so **no ordering against the matmul that produced `inp`**; and `out.copy_(inp)` at `:48` **races** the un-waited kernel 3. |
| A5 | H | `ipc_allreduce.sycl:16,39,41,43` | Hardcodes `float`. vLLM passes **bf16/fp16, never fp32** — so this is the common case. A bf16 tensor is reinterpreted as fp32: 2× overread, every "float" is two adjacent bf16 bit-concatenated. Compounds with A1 to 8×. |
| A6 | M | `ipc_allreduce.sycl:10` | `num_blocks = elements / VL` silently drops a tail of up to 15 elements — they retain rank-local values. Position-dependent silent corruption. |
| A7 | M | `torch_extension_ar.cc:65` | Declared `all_reduce_reg(int, Tensor inp, Tensor(a!) out)` but the kernel **mutates `inp`**, which is not marked mutable. `torch.compile`/AOTAutograd may CSE or reuse it → silent miscompile. (The manifest notes this exact `Tensor(a!)` lesson was learned for the GEMV ops.) |
| A8 | M | `setup_sycl.py:16-27` | Omits `ipc_allreduce.sycl` from the `_ar` sources (`setup.py` includes it) → **undefined symbol**, swallowed silently by the bare `except ImportError: pass` at `__init__.py:104-107`. |
| A9 | L | `torch_extension_ar.cc:22,29,32,51-54` | `world_size = handles.size()` unvalidated against fixed `void*[8]`; `dispose()` empty (FD leak once IPC is implemented); `all_reduce_unreg` ignores `reg_buffer`. |

### Build / patch

| ID | Sev | File:line | Defect |
|---|:---:|---|---|
| ~~B1~~ | — | `vllm-xpu-v0.14.0.patch:4764,5861` | **WITHDRAWN — not a defect.** The file is a `git format-patch` **mbox series of 36 commits**, not one diff. These are removal lines in commits 29/35 that revert what earlier commits added, so they *must* mirror `= self.weight_block_size` byte-for-byte. Changing them to `= None` would have introduced the breakage. |
| ~~B2~~ | — | `vllm-xpu-v0.14.0.patch:4452` | **WITHDRAWN — not a defect.** The header `@@ -616,6 +568,122 @@` is correct: the hunk body holds 116 `+` and 6 context lines, giving 122 new and 6 old. The original count compared this hunk against its mirror at `:5838`, which removes what this one adds. Second false positive of this class after B1. |
| B3 | C | `sglang_for_multi_arc.patch:~2941` | The "maintain exact line count" edit **strips a ~12-condition safety guard** (including `hidden_states.shape[0] > 8` — the kernel is decode-only) and the None-fallback, plus dtype conversions and per-expert scale derivation. Applies cleanly and fails **quietly** — bad inputs now give silently wrong numbers instead of falling back. Also calls `quant_info.weight_scale` and `dispatch_output.expert_idx`, **neither of which exists** → `AttributeError`. Semantically incomplete regardless (w13 GEMM only; no SiLU, no w2, no combine, no unscatter). **Delete the branch.** |
| B4 | H | `ops.py:1-5` | ~~`import torch.compiler.nn.functional as F`~~ — **FIXED**. Was a nonexistent module; `F` is used 4× (`F.silu` ×3, `F.softmax`). |
| B5 | H | `setup_native.sh` | ~~`set -u` aborts on oneAPI's unset `OCL_ICD_FILENAMES`~~ — **FIXED** via `set +u` around `setvars.sh`. |
| B6 | M | `setup_sycl.py`, `sglang/setup.py` | `-Xclang -funroll-loops` is **wrong** — `-funroll-loops` is a driver flag, not a cc1 option → "unknown argument". Same mistake on `-Xclang -fno-sycl-early-optimizations` (also a driver flag). Drop both `-Xclang`. Pasted into ~7 + 3 entries. Also: `-fsycl-targets=spir64_gen -Xs -device bmg` **duplicated** in the `_ar` entry. (`"-Xs", "-options -cl-intel-enable-auto-fma"` in list form is **correct** — setuptools passes argv without a shell.) |
| B7 | M | `__init__.py` | `from custom_esimd_kernels_vllm import deepseek_v41` is **unguarded** while the `_ar` import above it is try/except'd; `deepseek_v41` is absent from `setup_sycl.py` entirely. `__all__` lists two names that are **not bound** in the module namespace. |
| B8 | M | `sglang/.../__init__.py:~130` | `esimd_moe_gemm_fp8_blockscale` missing from the lazy-import allowlist → `AttributeError` on attribute access (the `torch.ops.*` path still works). |
| B9 | L | both Dockerfiles | `MAX_JOBS` was baked in rather than exposed. SYCL AOT is 2–4 GB+/TU, so a fixed high value OOMs a small CI runner and a fixed low value wastes a large host. Now `ARG MAX_JOBS=4`, overridable with `--build-arg`. |
| B10 | L | repo root | `.gitignore` covers **none** of: 14 `scratch_*.py`, `setup_native.sh`, `CLAUDE.MD`, `mise.toml`, both `build_*_18.log`, `references/` (**2.4 GB, two nested `.git` checkouts** — a `git add -A` would record broken gitlinks with no `.gitmodules`). Highest-priority ignore entry. |
| B11 | L | eval script | `platform_basic_evaluation.sh:120` — **`-m p2p` is not a valid value.** Verified in the shipped source: `supported_option_values{ sycl_mem_names[SYCL_MEM_USM] }`, `buf` commented out, `p2p` nonexistent. The benchmark prints a parse error and exits; with `set -eo pipefail` the whole script aborts before any report. P2P is controlled by `CCL_TOPO_P2P_ACCESS`, not a memory-type argument. **Revert to `-m usm`.** Also `-np 8` hardcoded while `count` is computed and ignored; `ze_peer` still only tests pair 0→1 despite the claim it was opened to 8 GPUs. |

### Documentation drift

| ID | Claim in prior manifest | Reality |
|---|---|---|
| D1 | "Fully native 5-step Push-model IPC with `atomic_load` spinlock" | No atomics anywhere; no L0 call; unreachable code |
| D2 | "Strictly bound to the TP group via `GroupCoordinator.world_size`" | Neither `GroupCoordinator` nor `get_tp_group` appears anywhere in the extension tree |
| D3 | "Injected `lsc_prefetch` into `MoeUpDecodeGeluTanh` / `MoeDownDecode`" | **Zero occurrences** of `lsc_prefetch` in either `moe_decode_gemv.h`. The hot weight streams carry no cache hints at all |
| D4 | Wide router "rewritten for O(N) bandwidth reduction" | Still `sycl::range<2>(n_tokens, num_experts)`, reloading weights per token |
| D5 | 8→16 B 2D load "doubles memory throughput by aligning to the 64 B cache line" | Reaches ¼ of the line, and **corrupts results** (K1) |
| D6 | "192 KB unified L1/SLM", "~42 GRF → 8 threads/XVE", "640 HW threads", "BMG-G21" | All wrong — see `architecture_hardware_reference.md` §1 |

## 3. Custom IPC all-reduce — design for a correct implementation

The current code is unreachable, so this is greenfield. The prior `bypass_custom_all_reduce_in_eager = False`
concern was **unfounded**: `_C_custom_ar`, `CustomAllreduce` and `should_custom_ar` appear
nowhere outside `torch_extension_ar.cc`; `XpuCommunicator.__init__` hard-sets
`ca_comm = None` (`vllm_for_multi_arc.patch:7048`); `torch.ops.vllm.all_reduce` routes to
oneCCL regardless. **Numerics today are correct.** The flag name is misleading, not dangerous.

### Immediate safety (before any implementation)

1. Replace the silent bypass at `torch_extension_ar.cc:42-43` with
   `TORCH_CHECK(false, "custom_ar: remote buffers not registered")`. **A no-op all-reduce
   must never be a legal outcome.** Graceful fallback belongs in Python at init time, via
   an availability probe that makes the GroupCoordinator pick oneCCL — never inside the
   op, where the caller cannot know it was skipped.
2. Add `TORCH_CHECK` for dtype, `numel % 16 == 0`, and contiguity so A5/A6 become hard
   errors rather than corruption.

### Handle exchange

`ze_ipc_mem_handle_t` is `char data[64]` whose first 4 bytes are a **process-local dma-buf
fd**. Intel's own recipe (`references/intel-sycl-llvm/unified-runtime/source/adapters/level_zero/usm.cpp:850-940`):

```cpp
struct ze_ipc_data_t { int pid; ze_ipc_mem_handle_t zeHandle; };
int fdRemote; memcpy(&fdRemote, &zeIpcData->zeHandle, 4);
int fdLocal = ur_duplicate_fd(zeIpcData->pid, fdRemote);   // pidfd_open + pidfd_getfd
memcpy(&zeIpcData->zeHandle, &fdLocal, 4);
zeMemOpenIpcHandle(ZeContext, ZeDevice, zeIpcData->zeHandle, 0, &Ptr);
```

The payload is `{pid, 64 bytes}` — which **does** fit the existing
`std::vector<std::string>& handles` signature. Fixing A3 does **not** require changing the
op schema; it requires actually using the handles plus a pid.

**Preferred: the SYCL-level extension**, if present in the pinned DPC++
(`sycl_ext_oneapi_inter_process_communication`, header
`sycl/ext/oneapi/experimental/ipc_memory.hpp`):

```cpp
namespace sycl::ext::oneapi::experimental::ipc::memory {
  handle get(void *Ptr, const context &Ctx);
  void  *open(const handle_data_t &H, const context &Ctx, const device &Dev);
  void   close(void *Ptr, const context &Ctx);
}
```

`handle::data()` returns a portable `std::vector<std::byte>` — pid/fd re-materialization is
internal. Removes the `-lze_loader` requirement and the raw `ze_api.h` include, and
`device.has(aspect::ext_oneapi_ipc_memory)` is exactly the init-time availability probe.
Experimental and Linux-only — verify it exists in the pinned compiler first.

**Transport:** `pidfd` primary (oneCCL's default since 2021.14), with explicit
`prctl(PR_SET_PTRACER, <consumer_pid>)` at init on the **exporting** side (TP ranks are
siblings, so per-peer, or `PR_SET_PTRACER_ANY` in-container) and a clear error on EPERM.
`SCM_RIGHTS` documented as the `ptrace_scope=2/3` fallback.

### Synchronization

`atomic_update` has **no memory-order or memory-scope parameter** — ordering must come
from an explicit scoped `fence`. `volatile int*` is provably insufficient (three
independent reasons in `architecture_hardware_reference.md` §4). Intel's own test suite
warns against the exact `while (B == 0) {}` construct in A2 because the compiler hoists
the load.

- Release: data stores → `fence<global, none, system>()` → flag store.
- Acquire: flag poll → `fence<global, invalidate, system_acquire>()` → data loads.
- **Sequence-numbered flags**, not 0/1 — each collective increments a device-memory `seq`;
  wait for `peer_flag >= my_seq`. Reusable across calls without a reset kernel, and
  graph-replay-safe. The current one-shot barrier could never be used twice even if it worked.
- Separate `alignas(128)` `start[]`/`end[]` arrays (as upstream vLLM does) to prevent both
  the second-sync-point race and false sharing of flag lines.

### Peer access

- Probe `can_access_peer(peer, atomics_supported)` — **not just `access_supported`**. The
  spec states normatively that atomics on peer memory need `memory_scope::system`, and
  concurrent atomic modify is **UB** if `atomics_supported` is false.
- `enable_peer_access` is **one-directional**: at TP=8 that is **56 ordered pairs**, not 28.
- **All devices must share one `sycl::context`** — a second, independent reason A4 is fatal.
  Use PyTorch's L0 context via `sycl::get_native<backend::ext_oneapi_level_zero>`.
- Single contiguous `zeMemAllocDevice`; not a caching-allocator slice, not expandable segments.

### Kernel structure

Replace 3 kernels + 2 host `.wait()` (two host round trips, ~20–60 µs, and uncapturable in
an XPU graph) with **two kernels, zero host waits**, both on the same in-order
`c10::xpu::getCurrentXPUStream()`. A single fully-fused kernel with a global spin barrier
**deadlocks by starvation** under SYCL's weakly-parallel forward-progress guarantee unless
launched as a persistent, occupancy-bounded grid.

### Algorithm selection

| Size | Choice | Rationale |
|---|---|---|
| ≤ 256 KB | one-shot | `7N` egress; latency-bound; single barrier dominates |
| > 256 KB | two-shot (reduce-scatter + all-gather) | `7N/4` egress — **4× less traffic**; second barrier amortizes |
| > ~2–4 MB | hand off to oneCCL | staging cost and lack of pipelining lose |

Matches upstream vLLM's empirically-tuned CUDA threshold at P=8. **Validate the crossover
with `ze_peer`; do not hardcode on faith.**

### Integration contract

Decide availability **once at init**, never per call. Populate `XpuCommunicator.ca_comm`
instead of the hard `None`; gate on `VLLM_XPU_USE_CUSTOM_ALLREDUCE`, default **off**.
`should_custom_ar` rejects dtype ∉ {fp16,bf16,fp32}, non-contiguous, `numel % 16 != 0`,
size > cap, `world_size > 8`. **Declining is the only legal fallback** — the op must never
return an unreduced tensor.

**Calibration:** upstream vLLM PR #54768 scopes itself to `world_size == 2, single node`.
Nobody upstream has shipped a validated 8-rank XPU IPC all-reduce.

## 4. Correct optimizations (keep)

Verified by reading; these are genuine and should not be reverted.

- **`fp8e4m3_to_fp16` branchless conversion** — 254/256 exact including every subnormal and
  both zeros, 4 vector ops, no branches, no LUT, no SLM. Subnormal handling is the subtle
  part and is exactly right.
- **VNNI2 hand-pack** (`fp8_moe_gemm_blockscale.h:251-263`) — bit-perfect across all 256
  index combinations. Hand-derived DPAS operand layouts are where these kernels usually break.
- **Scale-after-dot-product** — algebraically identical and **strictly more accurate** (one
  rounding instead of 128), while replacing 128 vector multiplies with 1 scalar multiply.
  The MoE kernel deliberately does the opposite (folds scale into `wf`) because it reuses one
  weight slice across `MAX_M` rows — both are correct, the inconsistency is justified.
- **Active-expert grid compaction** — solves the 99%-empty-grid problem at decode with no
  device→host sync; the sentinel-fill trick avoids a separate tensor-fill submission.
- **`ensure_moe_prefill_tile_buffers`** — per-device, per-queue, grow-only, thread-local.
  The right pattern; the decode path should adopt it (it currently does `at::full` per call,
  ~116 allocations/token at 58 layers).
- **`K_SPLIT` dispatch termination** — always terminates with a valid `ks`; brute-forced over
  K∈{128…18432} × N∈{1…7168} with no failures. (Correctly implemented against a **wrong
  target** — see K-series note below.)
- **e5m2 narrow-router fp32 accumulation** (`moe.sycl:182-196`) — an improvement; keep.
- **`q4_0_quant.sycl` include removal** — verified safe; the file references no torch types.

## 5. Occupancy target correction

`N * K_SPLIT >= 640` is live in `K_SPLIT` dispatch heuristics in at least:
`fp8_GEMV_bmg.h:188,199-203`; `fp8_GEMM_blockscale.h:155-168,309-311,420-421`;
`fp8_GEMM_pert.h:1396-1399`; `int4_GEMM.h:297` (both trees).

640 = 20 cores × 8 XVE × 4 threads — an **Arc B580 in large-GRF mode**. B70 is
**32 × 8 × 8 = 2048**. These heuristics stop splitting K roughly 3.2× too early, leaving
about two-thirds of the GPU idle on small-N shapes. `fp8_GEMV_bmg.h` is already
self-inconsistent (comment says 1280, code compares 640).

**This is the single highest-value correction in the repo** — but it is unmeasurable here,
so land it with a benchmark on real hardware rather than on the arithmetic alone.

## 6. Ranked work list

**Tier 0 — build-breaking, blocks everything**

1. K2 — restore module init in 5 vLLM `.cc` files (+ include in `q4_0.cc`)
2. A8 — add `ipc_allreduce.sycl` to `setup_sycl.py` sources
3. ~~B1, B2~~ — both withdrawn: the mbox series applies as written
4. B8 — add the blockscale op to the sglang allowlist

**Tier 1 — silent wrong results**

5. K1 — revert Q4_K 2D load to width 8
6. S1 — add a `TORCH_CHECK` per entry point (or a tail-correct fallback) in
   `select_vl_ks` / `select_vl_ks_int4` / `select_bmg`; four bugs, one root cause
7. S2 — fix the residual double-write and add the six `K>=VL` guards
8. S3 — migrate `resadd_norm_gemv_fused.h` and `_int4.h` to the separate-buffer pattern
   the kq/silu variants already use
9. S4 — mask the `block_load<int32_t,32>` for `num_experts % 32 != 0`
10. B3 — delete the dead sglang MoE patch branch
11. K3 — revert `acc[64]` router to 4 accumulators
12. K7 — revert the wide-router fp16 accumulator regression
13. S9 — make `int4_nmajor_gemm.h:311-322` match its comment, or fix the clamp
14. K4 — guard or re-route the direct router call

**Tier 2 — multi-card and safety rails**

15. S5 — pass the device index in 11 eagle entry points + `fp8_gemm_w8a16.h:94`
16. A-series — make the all-reduce bypass loud; add dtype/alignment/contiguity checks
17. S6, S7, K6 — bound the unbounded writes (`ids[32]`, `slm_init<16>`,
    `res_chunks[16]`, `build_active_experts`)
18. S8 — widen the 32-bit byte offsets to `size_t`
19. S10, S11, S12 — restore KV-cache clamping; verify the 104 KB SLM request against
    `clinfo`; add the `headDim` check
20. K9 — port the `weight_scale` shape check to sglang
21. B11 — `-m p2p` → `-m usm`; `-np "$count"`; extend `ze_peer` to all 7 peers

**Tier 3 — correctness of record**

22. Environment: `intel_iommu=on iommu=pt`; unset `FI_PROVIDER`; drop inert vars
23. Deploy the env tuning to Dockerfiles (currently in one benchmark script only)
24. B6 — drop the bogus `-Xclang` prefixes
25. B10 — `.gitignore` for `references/`, `scratch_*`, `*.log`

**Tier 4 — performance (all require hardware to validate)**

26. **Six work-group-size-1 dispatches** — `moe_ops.h:453,617,670,728` plus the two
    decode GEMV dispatches at `moe.sycl:2059-2062,2078-2081`. One work-item per
    work-group leaves ~94% of each EU's SIMD16 issue slots idle. **The highest-leverage
    perf theme in the codebase.**
27. §5 — 640 → 2048 occupancy target
28. K5 — `-doubleGRF` for the DPAS blockscale extension
29. Add the `lsc_prefetch` + cache hints that D3 claimed (the `*Grouped` variants already
    pass `cache_hint::cached`; the non-grouped path was simply missed)
30. Implement the O(N) wide-router rewrite that D4 claimed — in the **wide** kernel, where
    there is real reuse to capture and registers to spare
31. q5k/q6k `qh` loads fetch 8 bytes/row, use 2, re-fetched 4× ; scalar weight loads in
    `int4_nmajor_gemm.h:256-282` and `moe_int4.sycl:~400-425` (the q5k authors already
    fixed this exact pattern — see their note §10ax)
32. Router weight streams carry no cache hints (`moe.sycl:150,189,227,261`); missing hints
    on write-once weight streams generally
33. `submit_kernel` takes `std::function` by value — heap-allocates per launch
34. `MOE_N_TILE` is a shipped env A/B probe that **silently truncates output columns**
    when it does not divide `hidden_size` — gate or remove

## 7. Why these defects occurred

Three mechanisms account for most of the register. Each is structural, so each
has a structural countermeasure rather than a one-off fix.

### 7.1 Silent fallback as the default failure mode

The dominant pattern: a dispatcher cannot satisfy a request, so it falls through
to a path that runs but computes something else. `select_bmg` fell back to a
no-tail kernel; `fused_add_rms_norm_host` fell back to `LAUNCH_FARN(64)`;
`all_reduce_reg` fell back to `out.copy_(inp)`; `MoE_Scatter_Prefix_Kernel` read
past its array. None raised, logged or produced a NaN.

This is a *design default*, not a series of oversights. An `else` branch that
launches something is easier to write than one that proves the shape is
supported, and on the happy path it is indistinguishable from correct.

**Countermeasure.** Every dispatcher fallback must now either handle the general
case (the masked-tail kernel, the predicated prefix scan) or refuse
(`TORCH_CHECK`). The static tests assert that no catch-all `else` reaches a
launch that cannot handle the input. A no-op collective is never a legal outcome.

### 7.2 Tail and bounds handling treated as an edge case

Kernels were written for the shape in front of the author — `K % 512 == 0`,
`num_experts % 32 == 0`, `n_tokens <= 4`, offsets that fit in 32 bits — and the
remainder was left to a later pass that did not come. The K-residue bug covered
7395 of the 8161 K values in [32, 8192]; the "edge case" was the common case.

**Countermeasure.** Shape coverage in the test suite now includes the awkward
residues by default (K = 1044, 192; num_experts = 60, 6; either side of the
n_tokens = 4 boundary), and offsets that can be scaled by a token index carry an
explicit range check at the host.

### 7.3 Divergent twins (19 of 37 fix commits touch both trees)

`vllm/custom-esimd-kernels-vllm/csrc` and `sglang/custom-esimd-kernels/csrc`
carry near-copies of the same kernels. They have drifted: `fused_add_rms_norm.h`
is 323 lines in one tree and 79 in the other and fails differently in each;
`moe_decode_gemv.h` uses different variable names for the same buffers. A fix
found in one copy is not a fix in the other, and an audit that reads one copy
reports the wrong line numbers for its twin.

**Countermeasure.** Every fix in this branch was applied to both trees and
verified in both. The structural tests parametrise over both paths so a
single-tree fix fails. Converging the twins is not attempted here, but the
divergence is now measured rather than assumed.

### 7.4 A note on the audit itself

Register entries wrong on re-examination: B1 and B2 (both mbox-series hunks read
out of series context), the `setup_sycl` "missing include_dirs" claim, and S8's
magnitude (reported as comfortably bounded; it is exactly 2^32 at a reachable
prefill shape, i.e. worse). S3 conflated benign `normed_out` writes with genuine
residual races, and S11's premise rested on the 192 KB figure that the hardware
pass had already refuted.

Worse, four of the fixes introduced new defects of the same class they removed:
the resadd race fix deleted the residual stores on one path without launching
the replacement there; the KV surface bound used the logical block size where
the surface strides by physical rows; the block-scale parameterisation reached
the dense kernel but not the MoE kernel that ships; and the work-group widening
referenced struct members that were never declared, so neither tree compiled.

Two conclusions follow. First, a fix is not done when the diff looks right — it
is done when a compiler and an adversarial reader have both seen it; the
non-compiling tree passed 157 tests. Second, the register's line numbers drift
as commits land, so a stale pointer is the normal state and re-verification at
the point of edit is mandatory rather than a courtesy. Counts in §7.1 and §7.2
are descriptive of the mechanism, not audited tallies.

## 8. Coverage

The kernel surface is now covered across both trees: `moe_batch/`, `moe_grouped/`,
`eagle/`, `esimd_kernels/*GEMV*.h`, `int4_*`, `fused_add_rms_norm*`, `resadd_norm*`,
`moe_ops.h`, `prefill_dpas.h`, `moe_int4.sycl`, `moe_prefill_*`, `gdn_conv_fused*.h`,
plus all build files, patches and Python packaging.

**Not covered:** runtime behavior of any kind. Nothing here was executed.
