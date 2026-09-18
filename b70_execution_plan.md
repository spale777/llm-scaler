# B70 Execution Plan — Correctness, Scaling, and Topology Independence

Companion to `llm_scaler_b70_optimization_manifest.md` (defect register, with file:line)
and `architecture_hardware_reference.md` (sourced hardware facts).

**Goal:** correct and performant on 1, 8, 16, or any GPU count, under TP, PP, or any
combination — without the caller needing to know the topology.

**Constraint that shapes everything:** nothing here has been measured. The dev host has
no Intel GPU. Every perf item must land with a benchmark on real hardware, not on
arithmetic. Phases 0–2 are hardware-independent and can proceed now.

---

## Organizing principle: topology is a dispatch decision, not a kernel property

The current design hardcodes one topology. `remote_bufs[8]` is a fixed array; the custom
all-reduce assumes a single switch; PP does not exist. Making this work "for any number,
any layout" is **not** one all-reduce that works everywhere — it is a dispatch layer that
picks the right backend per communicator.

| Scope | Transport | Why |
|---|---|---|
| 1 GPU | no-op | No collective needed; must be a fast path, not a degenerate ring |
| 2–8, one switch | custom IPC push | Switch-local P2P, never reaches the root complex |
| 2–8, **root-complex P2P** | **custom IPC push, measured** — else oneCCL | Real D2D DMA, but through the CPU root complex. See below |
| 2–8, no P2P | oneCCL | P2P probe failed, or ACS redirect on |
| >8, one node | hierarchical: IPC intra-switch → oneCCL inter-switch | `remote_bufs[8]` cap; crossing switches has no direct D2D |
| Multi-node | oneCCL / OFI | No IPC across hosts |
| PP, any | point-to-point send/recv | No collectives cross the PP boundary — makes cross-switch collective hangs structurally impossible |

### Root-complex P2P is a distinct tier, not "no P2P"

GPUs on different root ports of the same socket — or any board without a PLX/PEX switch,
which is the common 1–4 card workstation case — can still do **device-to-device DMA routed
through the CPU root complex**. The CPU does not copy; it routes. This is materially
different from the host-staged bounce-buffer fallback (2 PCIe crossings + a host memcpy).

Three consequences:

1. **This is where the IOMMU actually matters.** Switch-local P2P never reaches the root
   port, so the kernel always permits it and IOMMU translation is not in the path (hardware
   ref §6). Root-complex P2P **does** reach the root port, so translation *is* in the path.
   `iommu=pt` (identity-mapped passthrough) is the correct setting — not `off`, which is a
   machine-wide DMA exposure, and not plain `on`, which pays translation cost. This is the
   topology where `pt` vs `on` is measurable.

2. **Permitted ≠ fast. UNVERIFIED and must be measured.** On some Intel server platforms,
   P2P across different root ports has historically been slow or serialized in ways that
   lose to host staging. Behavior varies by CPU generation, socket count, and whether the
   pair shares a root port. No sourced figure for Xe2 + a given Xeon was found — do not
   assume either direction.

3. **Therefore the probe cannot be a boolean.** `zeDeviceCanAccessPeer` reports that P2P is
   *permitted*, not that it beats oneCCL. Selection needs a **measured tier**:

   | Tier | Detection | Action |
   |---|---|---|
   | switch-local | shared upstream bridge in `lspci -tv` + peer access | custom IPC |
   | root-complex | peer access true, no shared switch | benchmark once at init; use IPC only if it beats the oneCCL baseline |
   | none | peer access false, or ACS redirect on | oneCCL |

   The init-time benchmark is a small fixed-size all-reduce against both paths. It runs
   once, costs milliseconds, and the result is fixed for the process (per §3.2 — availability
   must not flip per call). Cache it keyed on the PCI topology so it is not re-paid per launch.

   NUMA matters here too: on a dual-socket box, a GPU pair split across sockets routes over
   UPI/QPI and should be treated as a separate, worse tier — detect via
   `/sys/bus/pci/devices/*/numa_node` and prefer keeping a TP group socket-local, the same
   way §3.4 keeps it switch-local.

**Decision is made once at init, never per call.** Per-call algorithm selection by *size*
is fine; per-call *availability* flipping breaks batch invariance (vLLM #50136).

---

## Phase 0 — Unblock the build (no hardware needed)

Nothing is testable until the package imports.

| # | Item | Status |
|---|---|---|
| 0.1 | K2 — module init in 5 vLLM `.cc` + include in `q4_0.cc`; include in sglang twin | **DONE** |
| 0.2 | B4 — `ops.py` bad import | **DONE** |
| 0.3 | B5 — `set -u` vs oneAPI `vars.sh` | **DONE** |
| 0.4 | A8 — add `ipc_allreduce.sycl` to `setup_sycl.py` sources | todo |
| 0.5 | B8 — add `esimd_moe_gemm_fp8_blockscale` to the sglang lazy-import allowlist | todo |
| 0.6 | B7 — guard the `deepseek_v41` import; fix `__all__` | todo |
| 0.7 | B1/B2 — two patch-apply failures in `vllm-xpu-v0.14.0.patch` | todo |
| 0.8 | B6 — drop bogus `-Xclang` prefixes; de-duplicate AOT flags | todo |
| 0.9 | Build both trees clean; `import custom_esimd_kernels_vllm` succeeds | gate |

**Exit gate:** both packages import; both patches apply to a clean checkout.

---

## Phase 1 — Stop silent wrong answers (no hardware needed)

These produce plausible-looking wrong numbers. They survive smoke tests. Highest
priority after the build, because every later benchmark is meaningless until they are gone.

| # | Item | Failure mode |
|---|---|---|
| 1.1 | **K1** — revert Q4_K 2D load to width 8 | Every weight paired with the wrong scale/min |
| 1.2 | **S1** — `TORCH_CHECK` (or tail-correct fallback) in `select_vl_ks` / `select_vl_ks_int4` / `select_bmg` | 4 kernels silently discard the K-tail; `fp8_GEMV_v2.h:378` deliberately routes `K%64!=0` into one |
| 1.3 | **S2** — residual double-write + six missing `K>=VL` guards | Corrupts layer carry state; compounds through every later layer |
| 1.4 | **S3** — migrate `resadd_norm_gemv_fused.h` / `_int4.h` to the separate-buffer pattern | Cross-work-group race; nondeterministic |
| 1.5 | **S4** — mask `block_load<int32_t,32>` for `num_experts % 32 != 0` | DeepSeek-V2-Lite (60 experts) scatters tokens to wrong rows |
| 1.6 | **S9** — fix `int4_nmajor_gemm.h:311-322` clamp or make it match its "atomic add" comment | Duplicate-lane scatter loses contributions |
| 1.7 | **B3** — delete the dead sglang MoE patch branch | Strips a 12-condition guard; `AttributeError` on two nonexistent fields |
| 1.8 | **K3** — revert `acc[64]` router to 4 accumulators | 16 KB GRF vs 8 KB budget; runtime-indexed `simd` array forces scratch even at n_tokens=1 |
| 1.9 | **K7** — revert wide-router fp16 accumulator regression | Results change discontinuously at the n_tokens=4 dispatch boundary |
| 1.10 | **K4** — guard or re-route the direct router call | Tokens 4+ read uninitialized `torch::empty` |

**Exit gate:** a numerical parity harness (see 2.1) passes against CPU/reference for every
touched kernel at representative shapes, including tail shapes (`K%64!=0`, `num_experts%32!=0`).

---

## Phase 2 — Safety rails and portability (no hardware needed)

### 2.1 Numerical parity harness — build this first

Before any further kernel work. Shapes must include the tail cases Phase 1 exposed.
Without it, Phase 3+ changes are unverifiable and regressions are invisible.

### 2.2 Bounds and validation

| # | Item |
|---|---|
| 2.2.1 | S6, S7, K6 — bound `ids[32]`, `slm_init<16>`, `res_chunks[16]`, `build_active_experts` |
| 2.2.2 | S8 — widen 32-bit byte offsets to `size_t` (wraps at prefill scale: hidden 8192 × >262k tokens) |
| 2.2.3 | S10 — restore KV-cache OOB clamping (`kv_surf_h = 0x3FFFFF` defeats it deliberately) |
| 2.2.4 | S11 — verify the 104 KB SLM request against `clinfo`; Xe2 typically exposes 64 KB per work-group |
| 2.2.5 | S12 — `TORCH_CHECK` on `headDim != 256` instead of silently returning uninitialized Q/K/V |
| 2.2.6 | K9 — port the `weight_scale` shape check to the sglang binding |
| 2.2.7 | K10 — document the E4M3 NaN → ±480 behavior |

### 2.3 Multi-card correctness — prerequisite for all scaling work

| # | Item |
|---|---|
| 2.3.1 | **S5** — pass the device index in 11 eagle entry points + `fp8_gemm_w8a16.h:94`. They call `getCurrentXPUStream()` with no index while operating on another card's tensors. `build_tree_kernel_efficient` and `verify_tree_greedy` are on the hot spec-decode path |
| 2.3.2 | Audit every `getCurrentXPUStream()` call site tree-wide for a missing device index |
| 2.3.3 | Audit for `sycl::queue(gpu_selector_v)` — picks the default GPU, not the rank's (A4) |

**This phase is what makes "any number of GPUs" possible at all.** A kernel that always
submits to device 0 is correct on 1 GPU and wrong on every larger configuration.

### 2.4 Generalize the block-scale constraint

K8 — `block_n` is accepted and silently ignored (`/128` hardcoded); N-block and K-block
are conflated. Parameterize properly. **Blocks DeepSeek V4.1, which needs 32×32**, and is
a latent wrong-answer bug for any non-128 block size.

---

## Phase 3 — Topology-independent collective layer

The custom all-reduce today is unreachable (`ca_comm = None`; no caller anywhere), so this
is greenfield. Design detail in the manifest §3.

### 3.1 Safety first

Replace the silent bypass at `torch_extension_ar.cc:42-43` with a hard `TORCH_CHECK`.
**A no-op all-reduce must never be a legal outcome.** Fallback belongs in Python at init,
via an availability probe — never inside the op, where the caller cannot know it was skipped.

### 3.2 Capability probe (runs once, at init)

1. `device.has(aspect::ext_oneapi_ipc_memory)` — IPC available at all?
2. `can_access_peer(peer, atomics_supported)` for **all ordered pairs** — `enable_peer_access`
   is one-directional, so TP=8 is **56 pairs, not 28**. Atomics on peer memory are **UB**
   if `atomics_supported` is false.
3. **Classify the interconnect tier per pair**, not just permitted/denied:
   - shared upstream bridge in `lspci -tv` → switch-local
   - peer access true, no shared switch → root-complex (requires the init benchmark)
   - `numa_node` differs → cross-socket, worst tier
   Derive from actual PCIe topology, never from assumed device ordering.
4. **Measure once** for the root-complex tier: a small fixed-size all-reduce against both
   the IPC path and the oneCCL baseline. Use IPC only if it wins. Cache keyed on topology.
5. All devices in one `sycl::context` — required for peer USM, and a second reason A4 is fatal.

Probe result selects the backend from the table above and **is then fixed for the process**.
A mixed system (some pairs switch-local, some root-complex) either uses the worst tier for
the whole TP group or splits the group — prefer splitting, per the switch-local and
socket-local grouping rules in §3.4.

### 3.3 Correct IPC all-reduce (2–8 ranks, one switch)

Fixes A1–A9. Key points:

- **Handle exchange:** prefer `sycl_ext_oneapi_inter_process_communication` if present in
  the pinned DPC++ — `handle::data()` is already a portable payload, removes `-lze_loader`
  and the raw `ze_api.h` include. Else Intel's pidfd recipe (manifest §3). Payload is
  `{pid, 64 bytes}`, which **fits the existing op schema** — A3 does not require changing it.
- **Synchronization:** scoped fences, not `volatile`. Release: stores →
  `fence<global, none, system>()` → flag. Acquire: poll →
  `fence<global, invalidate, system_acquire>()` → loads. **Sequence-numbered flags**, not
  0/1, so the barrier is reusable and graph-replay-safe. `alignas(128)` separate
  `start[]`/`end[]` arrays.
- **Structure:** two kernels, zero host waits, same in-order stream. A single fused kernel
  with a global spin **deadlocks by starvation** under SYCL's weakly-parallel forward-progress
  guarantee unless launched as a persistent occupancy-bounded grid.
- **Algorithm by size:** ≤256 KB one-shot; >256 KB two-shot (4× less traffic); >2–4 MB hand
  to oneCCL. **Validate the crossover with `ze_peer`, do not hardcode.**
- **Dtype:** bf16/fp16 are the common case, not fp32 (A5). Handle all three or reject loudly.
- **Tail:** `numel % 16 != 0` must not silently drop elements (A6).

### 3.4 Scale beyond 8 and beyond one switch

- Replace `remote_bufs[8]` with a sized allocation; `TORCH_CHECK` the cap explicitly.
- **Hierarchical reduce for >8:** IPC intra-switch → oneCCL inter-switch → IPC broadcast back.
- Fall back to pure oneCCL whenever the probe says P2P is unavailable.

### 3.5 Pipeline parallel — transport confirmed, path unexercised

The transport exists and is verified against the shipped binary, not inferred from a
document: `nm -D libtorch_xpu.so` exports `c10d::ProcessGroupXCCL::send` and `::recv`,
`torch.distributed.is_xccl_available()` is true, and all five P2P entry points
(`send`, `recv`, `isend`, `irecv`, `batch_isend_irecv`) are present on torch 2.12+xpu.
The rendered PyTorch docs table showing ✗ for XCCL P2P is stale.
`test_pp_transport.py` pins all three facts so a torch bump cannot silently remove them.

vLLM's `DeviceCommunicatorBase` already routes `send`/`recv` to the device group, so
`XpuCommunicator` inherits a working implementation — **no override is required**, which
is why no PP code appears in the multi-arc patch. What is genuinely absent is exercise:
every launch in the repo is `-pp=1`, so the path has never run.

- PP groups are 2-rank, one per TP rank position; TP stays intra-switch.
- **No collectives cross the PP boundary** — this makes cross-switch collective hangs
  structurally impossible.
- Strict odd/even ordering or `batch_isend_irecv` to avoid the classic rank-order deadlock.
- Set `FI_PROVIDER` explicitly and identically across ranks (mismatch causes OFI init to
  hang rather than error) — `shm` intra-node, `tcp`/`verbs` inter-node. **Never `sockets`.**

**Remaining work is validation, not construction:** run `-pp=2 -tp=8` on 16 cards and
confirm the parity harness passes and no rank deadlocks. Until that runs, PP is
*untested*, which is a weaker claim than *unbuilt* but still not *working*.

### 3.6 Integration contract

Populate `XpuCommunicator.ca_comm` instead of the hard `None`
(`vllm_for_multi_arc.patch:7048`). Gate on `VLLM_XPU_USE_CUSTOM_ALLREDUCE`, default **off**
until validated. `should_custom_ar` rejects: dtype ∉ {fp16,bf16,fp32}, non-contiguous,
`numel % 16 != 0`, size > cap, world_size > cap. **Declining is the only legal fallback.**

**Calibration:** upstream vLLM PR #54768 scopes itself to `world_size == 2, single node`.
Nobody upstream has shipped a validated 8-rank XPU IPC all-reduce. Budget accordingly.

---

## Phase 4 — Environment and deployment

Currently the tuning exists in one benchmark script and **no Dockerfile sets any of it**.

| # | Item |
|---|---|
| 4.1 | `intel_iommu=on iommu=pt` (not `off` — see hardware ref §6; `off` is a machine-wide DMA exposure). Fix `installer.sh:108`, `native_bkc_setup.sh:13,153-157` |
| 4.2 | Check ACS on switch downstream ports — the **actual** P2P gate, not the IOMMU |
| 4.3 | Unset `FI_PROVIDER` globally; set per scope. Drop inert vars (`ZE_FLAT_DEVICE_HIERARCHY`, `CCL_ZE_IPC_EXCHANGE`, `TORCH_FR_BUFFER_SIZE=0`, `VLLM_XPU_INPLACE_ALLREDUCE`) |
| 4.4 | Deploy the surviving vars into the Dockerfiles, derived per topology at launch |
| 4.5 | **B11** — `-m p2p` → `-m usm` (invalid value; aborts the eval script before it runs anything); `-np "$count"`; extend `ze_peer` to all 7 peers; add a cross-switch p2p benchmark |
| 4.6 | Document or remove `SKIP_ALL_REDUCE=1` — an undocumented silent-wrong-output escape hatch |
| 4.7 | B9 — `MAX_JOBS` is a build arg, not a baked constant (SYCL AOT is 2–4 GB/TU) |
| 4.8 | B10 — `.gitignore` for `references/`, `scratch_*`, `*.log` — **DONE** |

---

## Phase 5 — Performance (requires hardware; every item needs a benchmark)

Ordered by expected value.

| # | Item | Expected |
|---|---|---|
| 5.1 | **Six work-group-size-1 dispatches** — `moe_ops.h:453,617,670,728` + `moe.sycl:2059-2062,2078-2081`. One work-item per work-group leaves ~94% of each EU's SIMD16 slots idle | Largest single win in decode |
| 5.2 | **Occupancy target 640 → 2048** across 6+ files. 640 is a B580 in large-GRF mode; B70 is 32×8×8 | ~3.2× more K-split headroom on small-N |
| 5.3 | K5 — `-doubleGRF` for the DPAS blockscale extension (~5.5 KB live state vs 8 KB budget); fix the stale "no DPAS" comment | Removes spilling |
| 5.4 | Add the `lsc_prefetch` + cache hints the manifest claimed but never had (the `*Grouped` variants already pass `cache_hint::cached`; the non-grouped path was missed) | Weight streams are the bottleneck |
| 5.5 | Implement the O(N) wide-router rewrite the manifest claimed — in the **wide** kernel, where reuse and registers exist | ÷n_tokens weight traffic |
| 5.6 | q5k/q6k `qh` loads fetch 8 B/row, use 2, re-fetched 4×; scalar weight loads in `int4_nmajor_gemm.h:256-282`, `moe_int4.sycl:~400-425` | q5k authors already solved this pattern |
| 5.7 | Cache hints on router weight streams (`moe.sycl:150,189,227,261`) and write-once streams generally | |
| 5.8 | `submit_kernel` takes `std::function` by value — heap-allocates per launch | Host-side decode overhead |
| 5.9 | Decode-path `at::full` per call (~116 allocations/token at 58 layers) — adopt `ensure_moe_prefill_tile_buffers`' pattern | |
| 5.10 | `MOE_N_TILE` silently truncates output columns when it does not divide `hidden_size` — gate or remove | Correctness risk in a perf knob |

---

## Phase 6 — DeepSeek V4.1 (gated on 16 cards)

**~551B params / ~298 GB.** B70 carries 32 GB, so 8 cards give 256 GB aggregate and need
~37 GB/card at TP=8. That exceeds the card, leaving no room for KV cache or activations,
so 8 cards remain out of reach. 16 cards (PP=2 × TP=8, ~18.6 GB/card) is the floor — the
dual-switch topology Phase 3.4/3.5 builds.

Prerequisites: Phase 2.4 (block-scale generalization to 32×32) and Phase 3.5 (PP path).

| # | Item | Effort |
|---|---|---|
| 6.1 | Fix `fp4_dequant.h` — LUT is **E2M1 × 1/8** (every ratio 0.125). Correct base `0x3800`, not `0x2C00`; decode scales to FP32, not FP16 (UE8M0 range exceeds fp16) | 0.5–1 d |
| 6.2 | Fix `topk_noaux_tc.h` — score **all** experts with sqrtsoftplus *first*, then `topk(scores + bias)`, then gather. Also implement the group-limited (noaux_tc) selection stage, which is absent. Divergence against a group-aware reference is unmeasured on hardware. FP32, eps 1e-20, E=384, `gate_temp`, `bias_vl` | 1 d |
| 6.3 | **Extend the SGLang fused-MoE guards** (`patch:9148,9427`) — they bail on `scoring_func != "softmax"`, `correction_bias is not None`, **and** `use_grouped_topk`. V4.1 trips all three, falling off the fast path **by construction**. A correct FP4 GEMM alone buys nothing | 1–2 wk |
| 6.4 | `fp4_moe_gemm.h` — fork `q4_0_GEMM.h`'s nibble-unpack DPAS loop + grouped scaffolding from `fp8_moe_gemm_blockscale.h`. **Evaluate the existing mxfp4 CUTLASS-SYCL path first** (already benchmarked at dsv4 shapes) | 1–2 wk |
| 6.5 | Delete `fp4_gemm.h` — its "native FP4 DPAS on Battlemage" premise is unachievable (no FP8/FP4 in Xe2 XMX) | — |
| 6.6 | Activation quant: per-32 FP8 + UE8M0 round-up | 2–3 d |
| 6.7 | `o_groups` block-diagonal `wo_a` | 2–3 d |
| 6.8 | Compressor + dual-theta RoPE + group-16 E4M3 FP4 (SGLang V4 pools already XPU-enabled) | 1 wk |
| 6.9 | `sparse_attn` — gather-by-index + online softmax + `attn_sink`. SGLang's `sparse_attn_func` is MInference vertical/slash, a **different algorithm**, and `fwd_sparse` is not even registered for kXPU | 2–3 wk |
| 6.10 | Indexer + two-level candidate selection | 2–3 wk |
| 6.11 | Engram n-gram | 1.5–2 wk |
| 6.12 | vLLM XPU model integration on `deepseek_v4` | 3–4 wk |

**Not required:** hyper-connections are already implemented natively on XPU
(`hc_pre_big_fuse`, `HC=4`/`SINKHORN_ITERS=20`, matching the reference config).

**Defer past v1:** DSpark MTP, vision tower.

Note 6.1/6.2 are **prerequisites, not live bug fixes** — the `deepseek_v41` tree is
untracked in git and nothing calls it.

---

## Validation matrix — the real definition of "works at any scale"

Every configuration must pass the parity harness, not just run.

| Config | Transport | What it proves |
|---|---|---|
| 1 GPU | none | No degenerate-collective overhead |
| 2 GPUs, same switch | IPC | Minimal P2P case; matches upstream's validated scope |
| **2–4, no switch (root-complex P2P)** | **measured: IPC or oneCCL** | **The workstation case. Proves the tier benchmark picks correctly rather than assuming** |
| **2+, cross-socket** | measured | UPI/QPI path; proves NUMA detection works |
| 4, 8 same switch | IPC | The design target; 56-pair probe |
| 8, P2P disabled | oneCCL | Fallback is real, not theoretical |
| **Mixed tiers in one node** | split or worst-tier | Proves group-splitting logic, not just uniform topologies |
| 16, TP=8 × PP=2 | IPC + p2p | Cross-switch path; no collectives cross the boundary |
| 16, TP=16 | hierarchical | The >8 path; `remote_bufs` cap removed |
| Multi-node | oneCCL/OFI | No IPC assumptions leak |
| Odd counts (3, 5, 6) | either | No power-of-two assumptions |

Odd counts matter: several kernels assume alignment or power-of-two shapes. The tail bugs
in Phase 1 are the same class of assumption.

---

## Sequencing

```
Phase 0 (build)  ──► Phase 1 (silent wrong answers) ──► Phase 2 (rails, multi-card)
                                                              │
                                    ┌─────────────────────────┼──────────────────┐
                                    ▼                         ▼                  ▼
                            Phase 3 (collectives)     Phase 4 (env/deploy)   Phase 5 (perf)
                                    │                                            │
                                    └──────────────► Phase 6 (DeepSeek) ◄────────┘
                                         (gated on 16 cards + 2.4 + 3.5)
```

Phases 0–2 need no hardware and should proceed immediately. Phase 5 cannot start without
GPUs. Phase 3 can be *written* without hardware but not validated. Phase 6 is gated on
both hardware and Phases 2.4 + 3.5.

**The one rule:** no perf claim enters any document without a measurement behind it. That
is the failure mode this whole effort is correcting.
