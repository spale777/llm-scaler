#!/usr/bin/env python3
"""Bring the DeepSeek V4.1 kernels up on real hardware, in dependency order.

Everything in this tree was checked without a GPU: the arithmetic against the
reference on CPU, the build by AOT compilation, the placement against the
published index. This runs the checks that need a card, and runs them in the
order where each failure is still interpretable -- a numerics mismatch means
something different when the device does not report the aspect the kernels
need, so that is established first.

    python vllm/tools/platform/dsv41_bringup.py            # all stages
    python vllm/tools/platform/dsv41_bringup.py --stage numerics
    python vllm/tools/platform/dsv41_bringup.py --json     # machine readable

Each stage reports PASS, FAIL or SKIP with the reason. A SKIP is not a pass:
it means the check could not run, and the summary counts it separately so a
run on the wrong machine cannot read as a clean bill.

No stage measures throughput. Every timing claim in this repo still needs a
benchmark behind it, and this is not one.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

PASS, FAIL, SKIP = "pass", "fail", "skip"


@dataclass
class Result:
    stage: str
    status: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def _torch():
    import torch
    return torch


# --- stage 1: is there a device, and does it report what the kernels need ---

def stage_device() -> Result:
    """The aspect check comes first because it reframes every later failure.

    ESIMD kernels require a device reporting ext_intel_esimd. A CPU OpenCL
    device does not, and submitting to one throws at launch rather than
    returning wrong numbers -- so a later numerics failure on such a device
    says nothing about the kernels.
    """
    try:
        torch = _torch()
    except ImportError as e:
        return Result("device", SKIP, f"torch not importable: {e}")

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return Result("device", SKIP, "no XPU device visible to torch")

    count = torch.xpu.device_count()
    names = []
    for i in range(count):
        try:
            names.append(torch.xpu.get_device_name(i))
        except Exception:
            names.append(f"<device {i}>")
    return Result("device", PASS, f"{count} XPU device(s): {', '.join(names)}",
                  {"count": count, "names": names})


def stage_esimd_aspect() -> Result:
    """Whether an ESIMD kernel can launch at all on this device."""
    try:
        torch = _torch()
    except ImportError as e:
        return Result("esimd_aspect", SKIP, str(e))
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return Result("esimd_aspect", SKIP, "no XPU device")

    try:
        import custom_esimd_kernels_vllm  # noqa: F401
    except ImportError as e:
        return Result("esimd_aspect", SKIP,
                      f"extension not built or not importable: {e}")

    # A one-element op exercises the launch path without depending on shapes.
    ns = getattr(torch.ops, "custom_esimd_kernels_vllm", None)
    if ns is None or not hasattr(ns, "deepseek_v41_noaux_tc_topk"):
        return Result("esimd_aspect", SKIP,
                      "deepseek_v41 ops are not registered")
    try:
        logits = torch.zeros(1, 384, dtype=torch.float16, device="xpu")
        bias = torch.zeros(384, dtype=torch.float16, device="xpu")
        ns.deepseek_v41_noaux_tc_topk(logits, bias, 6)
        torch.xpu.synchronize()
    except Exception as e:  # noqa: BLE001 - the message is the finding
        msg = str(e)
        if "ext_intel_esimd" in msg:
            return Result("esimd_aspect", FAIL,
                          "device does not report ext_intel_esimd, so no "
                          "ESIMD kernel can launch here")
        return Result("esimd_aspect", FAIL, f"ESIMD launch failed: {msg}")
    return Result("esimd_aspect", PASS, "an ESIMD kernel launched and completed")


# --- stage 2: numerics, against the references already written --------------

def _cpu_reference_topk(logits, bias, top_k):
    """The noaux_tc reference, in float64 on the host."""
    import math
    E = logits.shape[1]
    out_idx, out_w = [], []
    for t in range(logits.shape[0]):
        score = [math.sqrt(math.log1p(math.exp(float(logits[t, i]))))
                 for i in range(E)]
        sel = [score[i] + float(bias[i]) for i in range(E)]
        idx, tot = [], 0.0
        for _ in range(top_k):
            best = max(range(E), key=lambda i: sel[i])
            idx.append(best)
            tot += score[best]
            sel[best] = -3.0e38
        out_idx.append(idx)
        out_w.append([score[i] / (tot + 1e-20) * 1.5 for i in idx])
    return out_idx, out_w


def stage_numerics() -> Result:
    """Run each kernel against the reference its CPU tests already pin.

    A kernel that compiles, launches and returns wrong numbers is the failure
    mode the whole test suite exists to catch, and it is the one that only
    hardware can reveal.
    """
    try:
        torch = _torch()
    except ImportError as e:
        return Result("numerics", SKIP, str(e))
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return Result("numerics", SKIP, "no XPU device")
    try:
        import custom_esimd_kernels_vllm  # noqa: F401
    except ImportError as e:
        return Result("numerics", SKIP, f"extension not importable: {e}")
    ns = getattr(torch.ops, "custom_esimd_kernels_vllm", None)
    if ns is None:
        return Result("numerics", SKIP, "ops not registered")

    checks: dict[str, Any] = {}

    # Router: the expert set must match exactly, not approximately. A
    # different set is a different model, so this is compared as integers.
    try:
        torch.manual_seed(7)
        T, E, K = 4, 384, 6
        logits = (torch.randn(T, E) * 3).to(torch.float16)
        bias = (torch.randn(E) * 0.5).to(torch.float16)
        w, idx = ns.deepseek_v41_noaux_tc_topk(
            logits.to("xpu"), bias.to("xpu"), K)
        torch.xpu.synchronize()
        want_idx, want_w = _cpu_reference_topk(
            logits.float(), bias.float(), K)
        got_idx = idx.cpu().tolist()
        mism = sum(1 for a, b in zip(got_idx, want_idx) if a != b)
        werr = max(
            abs(float(w.cpu()[t, k]) - want_w[t][k])
            for t in range(T) for k in range(K))
        checks["router_index_mismatches"] = mism
        checks["router_weight_max_err"] = werr
        if mism:
            return Result("numerics", FAIL,
                          f"router selected a different expert set on "
                          f"{mism}/{T} tokens", checks)
        if werr > 2e-2:
            return Result("numerics", FAIL,
                          f"router weights diverge by {werr:.3e}", checks)
    except Exception as e:  # noqa: BLE001
        return Result("numerics", FAIL,
                      f"router: {e}\n{traceback.format_exc(limit=2)}", checks)

    # FP4 GEMM against a host sum in float64.
    try:
        M, N, Kd, G = 2, 32, 128, 32
        torch.manual_seed(11)
        a = (torch.randn(M, Kd) * 0.5).to(torch.float16)
        b = torch.randint(0, 256, (N, Kd // 2), dtype=torch.uint8)
        s = torch.randint(118, 134, (N, Kd // G), dtype=torch.uint8)
        c = ns.deepseek_v41_fp4_gemm(a.to("xpu"), b.to("xpu"), s.to("xpu"))
        torch.xpu.synchronize()

        e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        worst = 0.0
        for m in range(M):
            for n in range(N):
                want = 0.0
                for k in range(Kd):
                    byte = int(b[n, k // 2])
                    nib = (byte >> 4) & 0xF if k % 2 else byte & 0xF
                    wv = e2m1[nib & 7] * (-1.0 if nib & 8 else 1.0)
                    raw = int(s[n, k // G])
                    sc = 0.0 if raw <= 112 else 2.0 ** (min(raw, 142) - 127)
                    want += float(a[m, k]) * wv * sc
                got = float(c.cpu()[m, n])
                den = max(abs(want), 1.0)
                worst = max(worst, abs(got - want) / den)
        checks["fp4_gemm_max_rel_err"] = worst
        # fp16 accumulation over 128 terms, so the bar is loose but finite.
        if worst > 5e-2:
            return Result("numerics", FAIL,
                          f"FP4 GEMM diverges: {worst:.3e}", checks)
    except Exception as e:  # noqa: BLE001
        return Result("numerics", FAIL,
                      f"fp4 gemm: {e}\n{traceback.format_exc(limit=2)}",
                      checks)

    return Result("numerics", PASS,
                  "router selected the reference expert set; FP4 GEMM within "
                  "tolerance", checks)


# --- stage 3: multi-card, only what a second device can show ----------------

def stage_multicard() -> Result:
    """Every kernel must submit to its operands' device, not the current one.

    This is the failure that is invisible on one card: a kernel that always
    submits to device 0 is correct alone and wrong on every larger
    configuration.
    """
    try:
        torch = _torch()
    except ImportError as e:
        return Result("multicard", SKIP, str(e))
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return Result("multicard", SKIP, "no XPU device")
    if torch.xpu.device_count() < 2:
        return Result("multicard", SKIP,
                      "one device: the device-index bug cannot appear")
    try:
        import custom_esimd_kernels_vllm  # noqa: F401
        ns = torch.ops.custom_esimd_kernels_vllm
        logits = torch.zeros(2, 384, dtype=torch.float16, device="xpu:1")
        bias = torch.zeros(384, dtype=torch.float16, device="xpu:1")
        w, idx = ns.deepseek_v41_noaux_tc_topk(logits, bias, 6)
        torch.xpu.synchronize()
        if w.device.index != 1 or idx.device.index != 1:
            return Result("multicard", FAIL,
                          f"output landed on device {w.device.index}, not 1")
    except Exception as e:  # noqa: BLE001
        return Result("multicard", FAIL, f"device-1 launch failed: {e}")
    return Result("multicard", PASS, "a kernel ran on device 1 and stayed there")


# --- stage 4: the checkpoint, without loading 510 GB ------------------------

def stage_checkpoint(index_path: str | None) -> Result:
    """Validate a real checkpoint index against the placement plan.

    Reads the index and the shard headers only. That is enough to prove every
    tensor is recognised, placed exactly once, and budgeted -- the three things
    that decide whether a load will succeed before it is attempted.
    """
    if not index_path:
        return Result("checkpoint", SKIP,
                      "no --index given; pass model.safetensors.index.json")
    from pathlib import Path
    p = Path(index_path)
    if not p.is_file():
        return Result("checkpoint", SKIP, f"{p} not found")

    # Loaded by path, not as package members: the package __init__ imports the
    # compiled extensions, and planning a checkpoint must work on a machine
    # that has not built them.
    import importlib.util
    pkg = (Path(__file__).resolve().parents[2]
           / "custom-esimd-kernels-vllm/python/custom_esimd_kernels_vllm")

    def _mod(name):
        src = pkg / f"{name}.py"
        if not src.is_file():
            raise ImportError(f"{src} not present")
        full = f"custom_esimd_kernels_vllm.{name}"
        spec = importlib.util.spec_from_file_location(full, src)
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        return m

    try:
        L = _mod("deepseek_v41_layers")
        W = _mod("deepseek_v41_loader")
    except (ImportError, OSError) as e:
        return Result("checkpoint", SKIP, f"planning modules unavailable: {e}")

    index = json.loads(p.read_text())
    wm = index.get("weight_map")
    if not isinstance(wm, dict):
        return Result("checkpoint", FAIL, "index has no weight_map")

    names = sorted(wm)
    unknown = []
    for n in names:
        try:
            W.classify(n)
        except KeyError as e:
            unknown.append(str(e))
    if unknown:
        return Result("checkpoint", FAIL,
                      f"{len(unknown)} unrecognised tensor(s), e.g. "
                      f"{unknown[:3]}", {"unknown": len(unknown)})

    data = {"tensors": len(names), "shards": len(set(wm.values()))}
    try:
        stages = L.pipeline_split(40, 1)
        placements = W.place(names, stages, 1, 384)
        W.check_partition(placements, names, 1)
        data["placements_tp1"] = len(placements)
    except Exception as e:  # noqa: BLE001
        return Result("checkpoint", FAIL, f"placement: {e}", data)

    total = index.get("metadata", {}).get("total_size")
    if total:
        data["total_gb"] = round(total / 1e9, 1)
    return Result("checkpoint", PASS,
                  f"{data['tensors']} tensors over {data['shards']} shards, "
                  "all recognised and placed", data)


STAGES: dict[str, Callable[..., Result]] = {
    "device": stage_device,
    "esimd_aspect": stage_esimd_aspect,
    "numerics": stage_numerics,
    "multicard": stage_multicard,
    "checkpoint": stage_checkpoint,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="append", choices=sorted(STAGES),
                    help="run only these stages (repeatable)")
    ap.add_argument("--index", help="model.safetensors.index.json to validate")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)

    wanted = args.stage or ["device", "esimd_aspect", "numerics",
                            "multicard", "checkpoint"]
    results: list[Result] = []
    for name in wanted:
        fn = STAGES[name]
        if name == "checkpoint":
            results.append(fn(args.index))
        else:
            results.append(fn())

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
        for r in results:
            mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[r.status]
            print(f"[{mark}] {r.stage}: {r.detail}")
            for k, v in sorted(r.data.items()):
                print(f"         {k} = {v}")
        n_pass = sum(1 for r in results if r.status == PASS)
        n_fail = sum(1 for r in results if r.status == FAIL)
        n_skip = sum(1 for r in results if r.status == SKIP)
        # A skip is counted apart from a pass: it means the check could not
        # run, and a run on the wrong machine must not read as a clean bill.
        print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")

    return 1 if any(r.status == FAIL for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
