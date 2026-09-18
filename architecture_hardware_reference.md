# Intel Arc Pro B70 & PEX 89144 Architecture Reference

Every figure below is sourced. Unsourced values are marked UNVERIFIED and must not be
used as a basis for tuning. Nothing in this document has been measured on hardware —
see "Measurement status".

## 0. Measurement status

No Intel GPU is present on the development host (`/dev/dri` absent, no Arc device on
the PCI bus, `sycl-ls` reports OpenCL CPU only, `torch.xpu.is_available() == False`).
AOT compilation for `-device bmg` works without the hardware; **no performance claim in
this repo has been measured.** Treat every throughput/occupancy statement as a
prediction until run on the target cluster.

## 1. Intel Battlemage (BMG) Xe2 — corrected

| Property | Value | Source |
|---|---|---|
| Die (B70) | **BMG-G31** — not G21 | TechPowerUp / VideoCardz |
| Xe cores | 32 | Intel Arc Pro B-Series product page |
| Vector engines | 8 per Xe core, 512-bit, SIMD16/32 | Intel oneAPI GPU Optimization Guide, Xe GPU Architecture |
| XMX engines | 8 per Xe core = **256 total** | Intel Arc Pro B-Series product page |
| XMX width | 2048-bit — **UNVERIFIED**, press only, no Intel doc | — |
| **L1 cache** | **256 KB per Xe core** | Intel Xe GPU Architecture, Xe2-HPG row |
| **SLM** | **128 KB per Xe core** (separate from L1) | same |
| Cache line | 64 bytes | oneAPI Opt Guide, GPU Memory System |
| GRF | 128 regs (small) / 256 (large); register file fixed at 64 KB/XVE | oneAPI Opt Guide, small vs large register mode |
| HW threads | 8 per XVE (small GRF) / 4 (large GRF) | Intel Xe GPU Architecture |
| **Total HW threads** | **32 × 8 × 8 = 2048** (1024 in large-GRF mode) | derived from the above |
| VRAM | 32 GB GDDR6 ECC, 256-bit, 19 Gbps | Intel product page |
| Bandwidth | 608 GB/s | Intel product page |
| L2 cache | **UNVERIFIABLE** — no Intel doc or review states it | — |
| PCIe | Gen5 x16 | StorageReview B70 spec table |
| Clocks / TBP | 2280 MHz base, 2800 boost, 230 W (160–290 range) | Puget Systems / StorageReview |
| INT8 / FP32 | 367 TOPS / 22.94 TFLOPS | Intel product page |

### Corrections to prior revisions of this document

- **"192 KB unified L1/SLM" was wrong.** 192 KB is the Xe2-**LPG** (Lunar Lake iGPU)
  L1 figure. Xe2-HPG has 256 KB L1 **and** 128 KB SLM as separate resources. Any tiling
  or SLM-budget math derived from "192 KB unified" is wrong in both size and kind.
- **"~42 GRF therefore 8 threads/XVE" is a non-sequitur.** Occupancy is a binary
  consequence of the GRF *mode*, not of measured register usage. The register file is a
  fixed 64 KB: 128 regs → 8 threads, 256 regs → 4 threads. A kernel using 42 of 128
  registers gains nothing and wastes 86. The conclusion (avoid `-doubleGRF`) is right;
  the stated reason is not.
- **The occupancy target 640 is wrong for B70.** 640 = 20 cores × 8 XVE × 4 threads,
  i.e. an Arc B580 in large-GRF mode. B70 is **2048**. See the defect register.
- **Die is BMG-G31, not G21.** `-device bmg` is a family target so AOT is unaffected,
  but `intel_gpu_bmg_g21` architecture specializations would silently not match.

## 2. XMX / DPAS type support — no FP8, no FP4 on Xe2

Verified against `references/intel-sycl-llvm/sycl/include/sycl/ext/intel/esimd/xmx/`
(`dpas.hpp`, `common.hpp`) and Intel VTune's XMX instruction documentation.

**BMG XMX supports FP16, BF16, INT8, INT4, INT2. There is no FP8 and no FP4 matrix
arithmetic.**

- `dpas_argument_type` (common.hpp:24-40) has `bf8`, `hf8`, `e2m1`, `tf32` enumerators —
  but **no `e4m3`/`e5m2` at all**, and those paths are PVC-class silicon, gated to
  ExecutionSize 16.
- The ESIMD headers are **generation-agnostic**: they encode what the API accepts, not
  what the target GPU implements. `dpas.hpp:142-143` makes execution-size validation the
  caller's responsibility. **These paths compile cleanly for BMG and fail at runtime.**
- `sycl_ext_oneapi_fp8` provides **conversions only** — zero arithmetic operators.

Consequence: FP8 and FP4 are **storage formats**. They must be dequantized to FP16/BF16
before any XMX operation. This is what `fp8_moe_gemm_blockscale.h` already does, and it
is the only viable pattern on this hardware.

### DPAS constraints (dpas.hpp)

| Rule | Value | Line |
|---|---|---|
| `SystolicDepth` | must be exactly 8 | :97 |
| `RepeatCount` | 1–8 | :78-79 |
| Argument order | `dpas(C, B, A)` — **B before A** | :253-261 |
| Accumulator form | 6 type params `<T, CT, BT, AT>` | :253-261 |
| No-src0 form | 5 type params `<T, BT, AT>` | :284-289 |
| fp16×fp16 @ ExecSize 16 | T, CT ∈ {float, half} | :158-165 |
| fp16×fp16 @ ExecSize 8 | T, CT = float only | :151-157 |
| Float types | `APrecision == BPrecision` required (bf8↔hf8 is the sole exception) | :152,159,195,202,213 |
| Integer types | full cross-product of {s2,u2,s4,u4,s8,u8} | :218-233 |

Accumulator size is fully determined, not free: `OpsPerChannel = clamp(32/max(AElemBits,
BElemBits), 1, 8)`, `_K = SystolicDepth × OpsPerChannel`, `_N = ExecutionSize` (deduced
from B), result `N = RepeatCount × ExecutionSize`.

**VNNI interleave factor** = `32 / element_bits`: fp16/bf16 → 2 (VNNI2), int8/bf8/hf8 →
4, int4/e2m1 → 8, int2 → 16, tf32 → 1.

Xe2 is SIMD16-native, so ExecutionSize is 16. Nothing in the kernel sources records this
dependency; retargeting to DG2 (ExecSize 8) would silently narrow the fp16 accumulator
to float-only and change B-operand sizes.

## 3. 2D block load/store constraints

Authoritative checker: `esimd/memory.hpp:3943-3993`.

| Rule | Value | Line |
|---|---|---|
| **Max block width** | `BlockWidth × NBlocks × sizeof(T) ≤ 64 bytes` | :3932-3936 |
| Max bytes, load/prefetch | 2048 | :3951-3952 |
| Max bytes, store | 512 | :3949 |
| Width alignment | `(sizeof(T) × BlockWidth) % 4 == 0` | :3955-3956 |
| Transposed + transformed | mutually exclusive | :3953-3954 |
| Transposed | `NBlocks==1`, `sizeof(T)` ∈ {4,8} only | :3958-3960 |
| Transformed (VNNI) | `sizeof(T)` ∈ {1,2} only | :3971-3972 |
| Plain load height | ≤ 32 | :3988 |
| Plain store | `NBlocks==1`, height ≤ 8 | :3982-3983 |

**Returned layout:** dense row-major, **row stride == BlockWidth** (:4078-4100). Padding
is stripped. Changing `BlockWidth` changes the stride of the returned vector — consuming
code that slices at the old stride becomes silently wrong. This is the failure mode of
defect K1 in the register.

**Payload units** (`config_2d_mem_access`, experimental/esimd/memory.hpp:1808-1819):
`SurfaceWidth` = bytes − 1, `SurfaceHeight` = rows − 1, `SurfacePitch` = bytes − 1,
`X` = **elements**, `Y` = rows. Mixing byte and element units here is a classic silent
corruption bug.

**No compile-time base-address alignment check exists.** The only enforced alignment is
the DWORD block-width rule. Hardware requires aligned bases; misalignment is not caught.

**Cache hints** (`memory_properties.hpp:27-64`, legality in `common.hpp:561-601`):
loads accept L1 ∈ {uncached, cached, streaming} with L2 ∈ {uncached, cached}, or
L1=`read_invalidate` with L2=`cached`; both-`none` is also legal. Both-uncached is
rejected for prefetch.

## 4. Atomics and fences — for cross-GPU IPC

**`atomic_update` has no memory-order and no memory-scope parameter.** Grepping both
ESIMD header trees for `memory_order|memory_scope` returns zero matches. Ordering must
come entirely from an explicit scoped `fence`.

```cpp
template <memory_kind Kind = memory_kind::global,
          fence_flush_op FenceOp = fence_flush_op::none,
          fence_scope Scope = fence_scope::group>
void fence();                                    // memory.hpp:12006-12018
```

`fence_scope` (common.hpp:345-375): `group`=0, `local`=1, `tile`=2, `gpu`=3, `gpus`=4,
`system`=5, **`system_acquire`=6** — documented as committing downstream and peer writes
for GPUs that do not follow PCIe write ordering. `fence_flush_op`: `none`, `evict`,
`invalidate`, `clean`.

**The defaults are `global/none/group`** — a bare `fence<>()` is work-group scope and is
a silent correctness bug in IPC code. `barrier()`, `named_barrier_*` and `split_barrier`
are all work-group scope and irrelevant for cross-device synchronization.

**`volatile int*` is provably insufficient for a cross-GPU spinlock:**

1. `volatile` constrains the compiler, not the memory system. It orders only against
   other volatile accesses — not against the non-volatile data writes, and not against
   PCIe transaction ordering.
2. Intel GPU L1 is not coherent with incoming peer writes. A peer's write lands in
   device memory without invalidating the poller's L1, so a volatile load can hit a
   stale line **forever**. The existence of `fence_scope::system_acquire` is the proof.
3. Upstream vLLM reached the same conclusion — CUDA uses `st.release.sys`/`ld.acquire.sys`,
   ROCm uses `__scoped_atomic_store_n(..., __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM)`.

Correct pattern: release side = data stores → `fence<global, none, system>()` → flag
store. Acquire side = flag poll → `fence<global, invalidate, system_acquire>()` → data
loads.

**Trap:** `esimd::atomic_update<cmpxchg>` takes **src0 = new value, src1 = expected** —
reversed relative to `std::atomic` and to `lsc_atomic_update` (memory.hpp:8292-8294).
No 8-bit atomics (`sizeof(T) > 1` required, memory.hpp:5556).

## 5. Broadcom PEX 89144 and P2P

144-lane Gen5 ExpressFabric, 8× x16 downstream + 1× x16 upstream, internal switching
logic (not passive bifurcation). Switch firmware may need updating for dense GPU
hierarchies to enumerate.

**PCIe posted/non-posted:** writes are posted (fire-and-forget), reads are non-posted
(round-trip). This correctly motivates a **push (write) model** for P2P — but the
commonly stated rationale is imprecise:

- Bandwidth is limited by **outstanding-read credits**, not by the non-posted bit per
  se. Peer-read bandwidth is typically 30–60% of peer-write in practice because GPUs
  cannot keep enough reads in flight to cover round-trip latency.
- At TP=8 decode the tensors are small and **latency, not bandwidth, is the argument**.
  A read costs a full switch round trip (~1–2 µs, two hops) on the critical path; a
  write retires locally.
- A pull model also needs the same flag protocol *plus* the read latency. Push+flag is
  one round trip; pull+flag is two.

**UNVERIFIED:** no Broadcom-published read-vs-write P2P benchmark was found. Measure
with `ze_peer -o write` rather than asserting.

## 6. IOMMU — prior guidance was wrong

**`intel_iommu=off` is incorrect and is a security regression.** The Linux P2PDMA
documentation is explicit: when the path includes PCIe switches, the transaction can
route entirely within the PCIe hierarchy and never reach the root port, and **the kernel
always permits P2P in these well-defined cases**. Peer-to-peer transactions under one
switch are not IOMMU-translated at all.

The real gate is **ACS (Access Control Services)** on the switch's downstream ports. ACS
redirect forces peer TLPs upstream to the root complex for validation, defeating
switch-local routing. The PEX 89144 implements ACS.

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=on iommu=pt"
```

`iommu=pt` is identity-mapped passthrough: ~zero DMA translation overhead, IOMMU stays
active for isolation. Gives the throughput of `off` without the exposure.

**Exposure of `off`:** DMA remapping is disabled machine-wide; every PCIe device,
including anything running untrusted model code, gets unrestricted DMA to all host
physical memory.

If P2P is still slow with `iommu=pt`, check ACS first (§7B). If downstream ports show
`ReqRedir+`/`CmpltRedir+`, prefer disabling ACS redirect in switch firmware; fall back to
`pcie_acs_override=downstream,multifunction` only if the platform does not expose it —
that override weakens inter-device isolation within the switch.

Repo locations still carrying the wrong value: `vllm/tools/platform/installer.sh:108`,
`vllm/tools/native_bkc_setup.sh:13,153-157`.

## 7. Diagnostic commands

### A. Switch enumeration

```bash
lspci -tv                                    # all 8 GPUs under ONE upstream bridge
lspci -d 1000: -nn                           # Broadcom/PLX switch ports
for d in $(lspci -D -nn | grep -iE 'e20[0-9]|e21[0-9]' | cut -d' ' -f1); do
  echo "=== $d ==="; lspci -s "$d" -vv | grep -E 'LnkCap:|LnkSta:'
done
```

GOOD: `LnkSta: Speed 32GT/s, Width x16` on all 8, matching `LnkCap`.
BAD: width x8/x4 (bifurcation/firmware), speed 8/16GT/s (link trained down),
`(downgraded)`, or GPUs split across two upstream bridges (TP group would cross switches).

### B. ACS — the actual P2P gate

```bash
for d in $(lspci -D | grep -i 'PCI bridge' | cut -d' ' -f1); do
  echo "=== $d ==="; lspci -s "$d" -vv 2>/dev/null | grep -E 'ACSCap:|ACSCtl:'
done
for g in /sys/kernel/iommu_groups/*/devices/*; do
  echo "group $(basename $(dirname $(dirname $g))): $(basename $g)"
done | sort -V
```

GOOD: `ACSCtl: ... ReqRedir- CmpltRedir- ...` (all `-`); all 8 GPUs in one IOMMU group.
BAD: `ReqRedir+ CmpltRedir+` — peer TLPs forced to root complex; each GPU in its own
IOMMU group confirms ACS is isolating them.

### C. IOMMU mode

```bash
cat /proc/cmdline                            # expect intel_iommu=on iommu=pt
dmesg | grep -iE 'DMAR|IOMMU' | head -20
```

GOOD: `DMAR: IOMMU enabled` + `iommu: Default domain type: Passthrough`.

### D. Runtime enumeration

```bash
sycl-ls                                      # expect 8 [level_zero:gpu]
ZE_FLAT_DEVICE_HIERARCHY=FLAT sycl-ls        # must be identical (single-stack proof)
xpu-smi discovery && xpu-smi topology -m
```

BAD: fewer than 8 (driver/firmware enumeration failure); count differing between the two
`sycl-ls` runs; topology matrix showing host-routed links between same-switch GPUs.

### E. Prove P2P is active

```bash
for d in 1 2 3 4 5 6 7; do                   # full 0->N reachability
  ze_peer -o write -t transfer_bw -s 0 -d $d
done
ze_peer -o write -t transfer_bw --parallel_pair_targets 0:1,2:3,4:5,6:7

# A/B isolating the P2P path from the USM fallback
CCL_TOPO_P2P_ACCESS=1 CCL_LOG_LEVEL=info mpirun -np 8 /usr/bin/1ccl_benchmark \
  -a gpu -m usm -u device -e in_order -l allreduce -i 50 -w 20 -f 512 -t 67108864
CCL_TOPO_P2P_ACCESS=0 CCL_LOG_LEVEL=info mpirun -np 8 /usr/bin/1ccl_benchmark \
  -a gpu -m usm -u device -e in_order -l allreduce -i 50 -w 20 -f 512 -t 67108864
```

GOOD: tens of GB/s per pair, flat across all 7 destinations; log shows
`p2p matrix built, p2p_access_enabled=1` with 8 devices; `=1` beats `=0`.
BAD: `no p2p access between devices`; bandwidth collapsing for specific destinations
(asymmetric routing); `=1` and `=0` identical (P2P not engaging — check ACS first).

## 8. oneCCL / Level Zero environment — corrected

| Variable | Verdict | Recommended |
|---|---|---|
| `CCL_TOPO_P2P_ACCESS` | Real but **undocumented/internal**; gates `topo_manager::check_p2p_access()`. Version-fragile. | `1`, re-validate per oneCCL version |
| `CCL_ATL_TRANSPORT` | Correct | `ofi` |
| `CCL_ZE_IPC_EXCHANGE` | **Redundant** — `pidfd` is already default in 2021.14+ | omit |
| `FI_PROVIDER=sockets` | **WRONG — performance trap.** `sockets` is a *debug* provider, deprecated in favor of `tcp`, explicitly "not intended to provide performance improvements over regular TCP sockets" | **Unset globally.** `shm` intra-node, `tcp`/`verbs` inter-node |
| `ZE_FLAT_DEVICE_HIERARCHY` | **Inert.** Intel: "in a system with one stack per GPU card, FLAT and COMPOSITE are the same." B70 is single-stack. The "faster ze_peer discovery" claim has no source. | omit |
| `NEOReadDebugKeys` | Correct — master gate for NEO debug keys | `1` **benchmarking only** |
| `RenderCompressedBuffersEnabled` | Correct *for measurement accuracy*; it is not a perf optimization and may cost real bandwidth in production | `0` in eval script only |
| `TORCH_FR_BUFFER_SIZE=0` | **No-op — 0 is already the default.** Worse, pinning it destroys Flight Recorder, the best tool for diagnosing the cross-switch hangs this design risks | `0` in production; **`2000` when debugging a hang** |
| `VLLM_XPU_INPLACE_ALLREDUCE` | Redundant — the patch already defaults it to `"1"` | omit |
| `intel_iommu=off` | **WRONG** — see §6 | `intel_iommu=on iommu=pt` |

Also present: **`SKIP_ALL_REDUCE=1`** (`vllm_for_multi_arc.patch:7036`) — an undocumented
escape hatch that makes allreduce a silent no-op with numerically wrong output and no
warning. Document or remove.

**None of this tuning is deployed.** No Dockerfile sets any `CCL_*`/`FI_*`/`ZE_*`/`NEO*`
variable; the only baked ENVs are `VLLM_ALLOW_LONG_MAX_MODEL_LEN`,
`VLLM_WORKER_MULTIPROC_METHOD`, `VLLM_QUANTIZE_Q40_LIB`. Every variable above is set in
one benchmark script and nowhere else.

## 9. TP/PP topology for 2 switches × 8 GPUs

```
Switch A (PEX 89144 #1)            Switch B (PEX 89144 #2)
 ranks 0..7  = PP stage 0           ranks 8..15 = PP stage 1
 TP group A = {0..7}                TP group B = {8..15}
 PP groups: {0,8} {1,9} ... {7,15}  (one per TP rank position)
```

oneCCL selects transport **per communicator** from that communicator's device set, so
grouping is the lever — not global environment forcing.

1. **TP must never cross a switch.** Assign ranks via `ZE_AFFINITY_MASK` derived from
   actual `lspci -tv` topology, not assumed device ordering.
2. TP intra-switch: `CCL_TOPO_P2P_ACCESS=1` selects direct D2D. Verify
   `p2p_access_enabled=1` with 8 devices via `CCL_LOG_LEVEL=info`.
3. **PP cross-switch must be point-to-point only** — 8 independent 2-rank groups doing
   `send`/`recv` per microbatch. No collectives cross the switch boundary, which makes a
   cross-switch collective hang *structurally impossible*.
4. 2-rank PP groups fail the P2P check cleanly and select the OFI path unambiguously.
5. If both switches are in one chassis under one host, PP traffic is still intra-node —
   `shm` is correct, `sockets` would be absurd.

**XCCL does support point-to-point.** The rendered PyTorch docs table showing ✗ is stale;
`intel/torch-xpu-ops/src/xccl/ProcessGroupXCCL.cpp` implements `pointToPoint` (:913),
`send` (:1062), `recv` (:1105). PP is architecturally sound — but **entirely unbuilt**:
no `send`/`recv` or PP code exists in any patch, every launch is `-pp=1`, and the only
PP-aware code (`vllm-xpu-v0.14.0.patch:5328-5341`) *disables* a feature when
`pipeline_parallel_size > 1`.

**Known hang modes for PP:** rank-order deadlock (both stages `send` before `recv` —
use strict odd/even ordering or `batch_isend_irecv`); P2P-matrix misdetection (oneCCL
opens L0 IPC handles that are not mappable); IPC handle exhaustion
(`CCL_ZE_CACHE_*_IPC_HANDLES_THRESHOLD` defaults to 1000); provider mismatch across
ranks (OFI init hangs rather than erroring — a genuine argument for setting the provider
*explicitly and identically*, which is likely the real insight behind the original
`FI_PROVIDER` guidance, with the wrong value chosen).

**Contradiction to avoid:** `vllm/README.md:996` pairs `-tp=2` with two single-GPU Ray
nodes, i.e. TP spanning the network — the opposite of intra-switch TP. Do not use as a
template.

## 10. Level Zero IPC — why the CUDA API shape does not port

`ze_ipc_mem_handle_t` is `char data[64]` whose **first 4 bytes are a dma-buf file
descriptor** on Linux. The FD is **process-local**. It cannot be memcpy'd into a string,
shipped through `all_gather_object`, and reopened by a peer — the integer is meaningless
in the other process.

This is why a `std::vector<std::string> handles` signature copied from CUDA (where
`cudaIpcMemHandle_t` *is* position-independent) cannot work on XPU, and is the root cause
of defect A3 in the register.

Transports (`CCL_ZE_IPC_EXCHANGE`):
- **`sockets`** — `AF_UNIX` + `SCM_RIGHTS`. Universal, no kernel-version or capability
  requirement. **Recommended primary**; what upstream vLLM PR #54768 chose. No
  `ptrace_scope` failure mode, which matters in containers.
- **`pidfd`** — default since 2021.14. `pidfd_open` + `pidfd_getfd`. Needs Linux ≥ 5.6
  and `CAP_SYS_PTRACE` or same-uid with permissive `ptrace_scope`.
- **`drmfd`** — deprecated.

SYCL→L0 interop for the context/device handles:

```cpp
auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
```

**Hard constraint:** a pointer from `zeMemOpenIpcHandle` is usable in a kernel only if
opened against the **same `ze_context_handle_t` the queue uses**. All ranks must use the
PyTorch XPU runtime's L0 context via the interop call — never a freshly created one.

**Probe P2P before enabling:** `zeDeviceCanAccessPeer` / `device.ext_oneapi_can_access_peer`,
then `ext_oneapi_enable_peer_access`. Allocate a single contiguous `zeMemAllocDevice` —
not a caching-allocator slice, and not with expandable segments (cf. vLLM #42609, where
`expandable_segments` breaks IPC handle acquisition on CUDA for exactly this reason).

**Open hardware bugs to account for:** intel/compute-runtime#944 (`DEVICE_LOST`/OOM on
dual Arc B70), #995 (P2P copy crash with non-contiguously mapped peer blocks), vLLM
#41663 (XPU TP=2 dual B70 GP fault). Any design needs a tested, loudly-logged oneCCL
fallback.

**Calibration:** upstream vLLM PR #54768 scopes itself to `world_size == 2, single node`.
**Nobody upstream has shipped a validated 8-rank XPU IPC all-reduce.** TP=8 over a switch
is genuinely harder — 8 flags to poll, 56N vs 2N traffic, and a 28-pair reachability
matrix to verify.

## 11. Sources

Intel oneAPI GPU Optimization Guide (Xe GPU Architecture; small vs large register mode;
GPU memory system) · Intel Arc Pro B-Series product page · Intel VTune XMX instruction
docs · Linux kernel P2PDMA and IOMMU docs · fi_sockets(7), fi_shm(7), fi_provider(7) ·
oneCCL environment variables and benchmark user guide · Intel "Exposing the Device
Hierarchy" / "Flattening GPU Tile Hierarchy" · PyTorch Flight Recorder tutorial ·
intel/compute-runtime issues #944, #995 · vLLM PRs #54768, issues #41663, #42609, #50136 ·
Chips and Cheese Battlemage architecture · local: `references/intel-sycl-llvm/sycl/include/sycl/ext/intel/esimd/`,
`/opt/intel/oneapi/ccl/2022.1/`, `intel/torch-xpu-ops/src/xccl/ProcessGroupXCCL.cpp`
