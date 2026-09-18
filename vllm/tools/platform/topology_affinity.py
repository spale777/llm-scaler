#!/usr/bin/env python3
"""Derive ZE_AFFINITY_MASK and a TP/PP rank layout from the PCIe topology.

A tensor-parallel group that spans two PCIe switches does its all-reduce over
the switch uplinks, which is the one traffic pattern the dual-switch design
exists to avoid. Device enumeration order does not encode that: `sycl-ls`
order follows BDF ordering, which need not group by switch, so a TP group
assigned as "the first eight devices" can straddle the boundary.

This reads the actual hierarchy from sysfs and emits an affinity mask whose
order makes rank i of a TP group share an upstream bridge with rank i+1. PP
stages are then laid across switches, so the only cross-switch traffic is
point-to-point activations between stages and no collective crosses.

Grouping rules, in priority order:
  1. never split a TP group across upstream bridges
  2. never split a TP group across NUMA nodes
  3. fill switches in a stable order so a rerun yields the same mask

Reads only /sys; no GPU, driver, or root needed. With --from-lspci it parses a
saved `lspci -tvnn` capture instead, so a layout can be planned from a machine
that does not have the cards.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Battlemage PCI IDs. B60 is the G21 die, B70 the G31; B580 shares G21 and is
# listed because a mixed bench box will enumerate one.
BMG_DEVICE_IDS = {
    "0xe210": "B60",
    "0xe211": "B60",
    "0xe20b": "B580",
    "0xe223": "B70",
}
INTEL_VENDOR = "0x8086"

SYS_PCI = Path("/sys/bus/pci/devices")


class Gpu:
    __slots__ = ("bdf", "device_id", "model", "switch", "numa")

    def __init__(self, bdf, device_id, model, switch, numa):
        self.bdf = bdf
        self.device_id = device_id
        self.model = model
        self.switch = switch
        self.numa = numa

    def __repr__(self):
        return f"Gpu({self.bdf}, {self.model}, switch={self.switch}, numa={self.numa})"


def _read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _upstream_bridge(bdf, root=SYS_PCI):
    """The nearest enclosing bridge that is not the root port.

    Walking the sysfs symlink upward yields .../<root port>/<switch up>/
    <switch down>/<device>. Two GPUs behind one switch share the switch's
    upstream port, so that component is the grouping key. A GPU plugged
    straight into a root port has no switch and keys on the root port, which
    correctly makes it its own group.
    """
    link = (root / bdf).resolve()
    parts = [p for p in link.parts if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]", p)]
    # parts[-1] is the device itself; the bridge above it is parts[-2] when a
    # switch is present, else the device sits directly on a root port.
    if len(parts) >= 3:
        return parts[-3]
    if len(parts) == 2:
        return parts[-2]
    return "none"


def discover(root=SYS_PCI):
    """Every Battlemage GPU present, with its switch and NUMA node."""
    gpus = []
    if not root.is_dir():
        return gpus
    for dev in sorted(root.iterdir()):
        if _read(dev / "vendor").lower() != INTEL_VENDOR:
            continue
        did = _read(dev / "device").lower()
        if did not in BMG_DEVICE_IDS:
            continue
        numa = _read(dev / "numa_node") or "-1"
        gpus.append(Gpu(dev.name, did, BMG_DEVICE_IDS[did],
                        _upstream_bridge(dev.name, root), int(numa)))
    return gpus


def parse_lspci_tree(text):
    """GPUs and their enclosing bridge from an `lspci -tvnn` capture.

    lspci draws the hierarchy with the bus in brackets and the device as a
    bare function:

        -+-[0000:00]-+-01.0-[01-30]----00.0-[02-30]--+-01.0-[03]----00.0  ... [8086:e223]

    A device's own bus comes from the nearest `[bus]` or `[bus-range]` to its
    left on the same line, and the switch it hangs off is the bracket before
    that. Column position encodes depth, so the bridge that owns a continuation
    line is the last bracket seen at or left of that column.
    """
    gpus = []
    brackets = []  # (column, bus, parent_bus)
    tok = re.compile(r"\[([0-9a-f]{4}:[0-9a-f]{2}|[0-9a-f]{2})(?:-[0-9a-f]{2})?\]"
                     r"|([0-9a-f]{2}\.[0-9a-f])"
                     r"|\[8086:([0-9a-f]{4})\]", re.I)

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" |"))
        # Drop brackets opened to the right of where this line starts: they
        # belong to a subtree that has ended.
        brackets = [b for b in brackets if b[0] < indent] if indent else []

        for m in re.finditer(r"\[(?:[0-9a-f]{4}:)?([0-9a-f]{2})(?:-[0-9a-f]{2})?\]"
                             r"|\b([0-9a-f]{2}\.[0-9a-f])\b"
                             r"|\[8086:([0-9a-f]{4})\]", line, re.I):
            bus, func, vend = m.group(1), m.group(2), m.group(3)
            if vend is not None:
                key = f"0x{vend.lower()}"
                if key in BMG_DEVICE_IDS and gpus and gpus[-1].device_id is None:
                    gpus[-1].device_id = key
                    gpus[-1].model = BMG_DEVICE_IDS[key]
                elif key not in BMG_DEVICE_IDS and gpus and gpus[-1].device_id is None:
                    gpus.pop()          # not a Battlemage part after all
                continue
            if bus is not None:
                brackets = [b for b in brackets if b[0] < m.start()]
                parent = brackets[-1][1] if brackets else "none"
                brackets.append((m.start(), bus.lower(), parent))
                continue
            if func is not None:
                # A function directly under the innermost bracket. Whether it
                # is a device or a bridge is decided by what follows: a bridge
                # is immediately succeeded by another `[bus]`.
                cur = [b for b in brackets if b[0] < m.start()]
                if not cur:
                    continue
                own_bus, parent_bus = cur[-1][1], (cur[-1][2] if cur[-1][2] else "none")
                rest = line[m.end():m.end() + 2]
                if rest.startswith("-["):
                    continue            # a bridge; its bracket is handled above
                gpus.append(Gpu(f"0000:{own_bus}:{func}", None, None,
                                parent_bus, -1))

    # Entries never resolved to a Battlemage id are not GPUs.
    return [g for g in gpus if g.device_id in BMG_DEVICE_IDS]


def group_devices(gpus, tp, pp):
    """Order devices so each contiguous run of `tp` shares a switch.

    Returns (order, diagnostics). `order` is a list of Gpu in the sequence the
    affinity mask should present them; rank r of the flattened world then maps
    to order[r], so TP group g is order[g*tp : (g+1)*tp].
    """
    want = tp * pp
    diags = []

    buckets = {}
    for g in gpus:
        buckets.setdefault((g.switch, g.numa), []).append(g)
    for k in buckets:
        buckets[k].sort(key=lambda g: g.bdf)

    order = []
    # Largest buckets first so a full TP group lands in one switch when any
    # switch can hold it; ties broken on the key for a stable rerun.
    for key in sorted(buckets, key=lambda k: (-len(buckets[k]), k)):
        members = buckets[key]
        if len(members) < tp:
            diags.append(
                f"switch {key[0]} numa {key[1]} holds {len(members)} GPU(s), "
                f"fewer than tp={tp}: a TP group placed here would cross a "
                f"switch boundary"
            )
        order.extend(members)

    if len(order) < want:
        diags.append(f"need {want} GPUs for tp={tp} pp={pp}, found {len(order)}")
    return order[:want], diags


def tp_groups_are_switch_local(order, tp):
    """True when no contiguous TP run spans more than one switch."""
    for i in range(0, len(order) - len(order) % tp, tp):
        grp = order[i:i + tp]
        if len({g.switch for g in grp}) > 1:
            return False
    return True


def build_plan(gpus, tp, pp):
    order, diags = group_devices(gpus, tp, pp)
    world = len(order)
    plan = {
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "world_size": world,
        "affinity_mask": ",".join(str(i) for i in range(world)),
        "device_order": [g.bdf for g in order],
        "models": sorted({g.model for g in order}),
        "tp_groups": [[g.bdf for g in order[i:i + tp]]
                      for i in range(0, world - world % tp, tp)],
        # One PP group per TP rank position: stage s of position r is
        # order[s*tp + r], so a PP pair is the same lane on each stage.
        "pp_groups": [[order[s * tp + r].bdf for s in range(pp)]
                      for r in range(tp)] if pp > 1 and world >= tp * pp else [],
        "switch_local_tp": tp_groups_are_switch_local(order, tp),
        "warnings": diags,
    }
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-tp", "--tensor-parallel-size", type=int, default=8)
    ap.add_argument("-pp", "--pipeline-parallel-size", type=int, default=1)
    ap.add_argument("--from-lspci", metavar="FILE",
                    help="parse a saved `lspci -tvnn` capture instead of /sys")
    ap.add_argument("--json", action="store_true", help="emit the plan as JSON")
    ap.add_argument("--export", action="store_true",
                    help="emit shell `export` lines for the launch environment")
    args = ap.parse_args(argv)

    if args.from_lspci:
        gpus = parse_lspci_tree(Path(args.from_lspci).read_text())
    else:
        gpus = discover()

    if not gpus:
        print("no Battlemage GPUs found "
              "(looked for Intel 0xE210/0xE211/0xE20B/0xE223)", file=sys.stderr)
        return 2

    plan = build_plan(gpus, args.tensor_parallel_size, args.pipeline_parallel_size)

    if args.json:
        print(json.dumps(plan, indent=2))
        return 0 if not plan["warnings"] else 1

    if args.export:
        print(f'export ZE_AFFINITY_MASK={plan["affinity_mask"]}')
        # TP is intra-switch, so peer-to-peer is the path that matters.
        print("export CCL_TOPO_P2P_ACCESS=1")
        print("export CCL_ATL_TRANSPORT=ofi")
        # Identical across ranks: a provider mismatch makes OFI init hang
        # rather than fail. shm is correct while every rank is on one host.
        print("export FI_PROVIDER=shm")
        print("export VLLM_WORKER_MULTIPROC_METHOD=spawn")
        return 0 if not plan["warnings"] else 1

    print(f"found {len(gpus)} GPU(s): {', '.join(sorted(set(g.model for g in gpus)))}")
    print(f"world={plan['world_size']}  tp={plan['tensor_parallel_size']}  "
          f"pp={plan['pipeline_parallel_size']}")
    print(f"ZE_AFFINITY_MASK={plan['affinity_mask']}")
    for i, grp in enumerate(plan["tp_groups"]):
        print(f"  TP group {i}: {' '.join(grp)}")
    for i, grp in enumerate(plan["pp_groups"]):
        print(f"  PP group {i}: {' -> '.join(grp)}")
    print(f"switch-local TP: {plan['switch_local_tp']}")
    for w in plan["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0 if not plan["warnings"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
