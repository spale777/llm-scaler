"""Contract tests for the TP/PP topology planner.

The rule the dual-switch design rests on is that a tensor-parallel group never
spans two PCIe switches: its all-reduce would then run over the switch uplinks,
which is the traffic the topology exists to keep off. Device enumeration order
does not encode switch membership, so "the first eight devices" can straddle
the boundary while looking correct.

These build synthetic hierarchies in a temporary sysfs tree and in lspci
captures, so the grouping is exercised without any GPU present.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = (Path(__file__).resolve().parents[3]
         / "vllm/tools/platform/topology_affinity.py")

sys.path.insert(0, str(_TOOL.parent))
ta = pytest.importorskip("topology_affinity")


def _mk_sysfs(tmp_path, layout):
    """Build a fake /sys/bus/pci/devices.

    layout: list of (bdf, device_id, switch_bdf or None, numa)
    The resolved symlink path is what the tool walks, so the device directory
    is created *under* its bridges and linked back from the flat directory,
    exactly as the kernel presents it.
    """
    flat = tmp_path / "devices"
    flat.mkdir(parents=True, exist_ok=True)
    tree = tmp_path / "tree"
    for bdf, did, switch, numa in layout:
        if switch:
            # root port / switch upstream / switch downstream / device
            real = tree / "0000:00:01.0" / switch / f"{switch[:-1]}1" / bdf
        else:
            real = tree / "0000:00:01.0" / bdf
        real.mkdir(parents=True, exist_ok=True)
        (real / "vendor").write_text("0x8086\n")
        (real / "device").write_text(f"{did}\n")
        (real / "numa_node").write_text(f"{numa}\n")
        link = flat / bdf
        if not link.exists():
            link.symlink_to(real)
    return flat


def _gpus(tmp_path, layout):
    return ta.discover(root=_mk_sysfs(tmp_path, layout))


def test_discovers_only_battlemage_parts(tmp_path):
    flat = _mk_sysfs(tmp_path, [
        ("0000:03:00.0", "0xe223", "0000:02:00.0", 0),
        ("0000:04:00.0", "0xe211", "0000:02:00.0", 0),
    ])
    # A non-Battlemage Intel device and a non-Intel device must be ignored.
    for bdf, vendor, did in (("0000:05:00.0", "0x8086", "0x1234"),
                             ("0000:06:00.0", "0x10de", "0xe223")):
        d = tmp_path / "tree" / "0000:00:01.0" / bdf
        d.mkdir(parents=True, exist_ok=True)
        (d / "vendor").write_text(vendor + "\n")
        (d / "device").write_text(did + "\n")
        (d / "numa_node").write_text("0\n")
        (flat / bdf).symlink_to(d)

    gpus = ta.discover(root=flat)
    assert {g.bdf for g in gpus} == {"0000:03:00.0", "0000:04:00.0"}
    assert {g.model for g in gpus} == {"B70", "B60"}


def test_two_switches_keep_each_tp_group_switch_local(tmp_path):
    """The design target: 16 cards, 8 per switch, tp=8 pp=2."""
    layout = []
    for i in range(8):
        layout.append((f"0000:1{i:x}:00.0", "0xe223", "0000:10:00.0", 0))
    for i in range(8):
        layout.append((f"0000:2{i:x}:00.0", "0xe223", "0000:20:00.0", 1))

    plan = ta.build_plan(_gpus(tmp_path, layout), tp=8, pp=2)
    assert plan["world_size"] == 16
    assert plan["switch_local_tp"], plan["warnings"]
    assert len(plan["tp_groups"]) == 2
    for grp in plan["tp_groups"]:
        prefixes = {b.split(":")[1][0] for b in grp}
        assert len(prefixes) == 1, f"TP group spans switches: {grp}"
    # One PP pair per TP rank position, each pair crossing the switch boundary.
    assert len(plan["pp_groups"]) == 8
    for pair in plan["pp_groups"]:
        assert len(pair) == 2
        assert pair[0].split(":")[1][0] != pair[1].split(":")[1][0]


def test_interleaved_enumeration_is_regrouped(tmp_path):
    """BDF order alternating between switches must not become the rank order.

    This is the case the tool exists for: taking devices in enumeration order
    would put four cards from each switch in every TP group.
    """
    layout = []
    for i in range(8):
        switch = "0000:10:00.0" if i % 2 == 0 else "0000:20:00.0"
        layout.append((f"0000:{0x30 + i:x}:00.0", "0xe223", switch, 0))

    gpus = _gpus(tmp_path, layout)
    naive = sorted(gpus, key=lambda g: g.bdf)
    assert not ta.tp_groups_are_switch_local(naive, tp=4), (
        "enumeration order should straddle switches here, or this test is "
        "not exercising the regrouping"
    )

    plan = ta.build_plan(gpus, tp=4, pp=2)
    assert plan["switch_local_tp"], plan["warnings"]


def test_warns_when_no_switch_can_hold_a_tp_group(tmp_path):
    """Four cards split two-and-two cannot host a tp=4 group intact."""
    layout = [
        ("0000:11:00.0", "0xe211", "0000:10:00.0", 0),
        ("0000:12:00.0", "0xe211", "0000:10:00.0", 0),
        ("0000:21:00.0", "0xe211", "0000:20:00.0", 0),
        ("0000:22:00.0", "0xe211", "0000:20:00.0", 0),
    ]
    plan = ta.build_plan(_gpus(tmp_path, layout), tp=4, pp=1)
    assert plan["warnings"], "a TP group forced across switches must warn"
    assert any("cross" in w for w in plan["warnings"])
    assert not plan["switch_local_tp"]


def test_numa_split_is_not_merged_into_one_tp_group(tmp_path):
    """Cross-socket pairs route over UPI and are a worse tier than one switch."""
    layout = [
        ("0000:11:00.0", "0xe223", "0000:10:00.0", 0),
        ("0000:12:00.0", "0xe223", "0000:10:00.0", 0),
        ("0000:81:00.0", "0xe223", "0000:80:00.0", 1),
        ("0000:82:00.0", "0xe223", "0000:80:00.0", 1),
    ]
    plan = ta.build_plan(_gpus(tmp_path, layout), tp=2, pp=2)
    assert plan["switch_local_tp"], plan["warnings"]
    for grp in plan["tp_groups"]:
        nodes = {b.split(":")[1] for b in grp}
        assert len(nodes) == 1 or all(n.startswith("1") for n in nodes) \
            or all(n.startswith("8") for n in nodes)


def test_single_gpu_plan_is_degenerate_but_valid(tmp_path):
    layout = [("0000:03:00.0", "0xe211", None, 0)]
    plan = ta.build_plan(_gpus(tmp_path, layout), tp=1, pp=1)
    assert plan["world_size"] == 1
    assert plan["affinity_mask"] == "0"
    assert plan["switch_local_tp"]
    assert plan["pp_groups"] == []


def test_lspci_capture_is_parsed_without_sysfs():
    """A layout can be planned from a capture taken on another machine."""
    cap = """\
-[0000:00]-+-01.0-[01-30]----00.0-[02-30]--+-01.0-[03]----00.0  Intel Corporation Device [8086:e223]
           |                               +-02.0-[04]----00.0  Intel Corporation Device [8086:e223]
           |                               \\-03.0-[05]----00.0  Intel Corporation Device [8086:e223]
           \\-02.0-[31-60]----00.0-[32-60]--+-01.0-[33]----00.0  Intel Corporation Device [8086:e223]
                                           \\-02.0-[34]----00.0  Intel Corporation Device [8086:e223]
"""
    gpus = ta.parse_lspci_tree(cap)
    assert len(gpus) == 5, [g.bdf for g in gpus]
    assert all(g.model == "B70" for g in gpus)
    assert len({g.switch for g in gpus}) == 2, (
        "the two switch subtrees must be distinguished"
    )


def test_cli_emits_exports_and_reports_missing_hardware():
    """Running with no Battlemage present must fail loudly, not emit a mask."""
    r = subprocess.run([sys.executable, str(_TOOL), "--export"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        # A machine that really has the cards; the mask must still be present.
        assert "ZE_AFFINITY_MASK" in r.stdout
    else:
        assert "no Battlemage GPUs found" in r.stderr
        assert "ZE_AFFINITY_MASK" not in r.stdout


def test_json_plan_is_machine_readable(tmp_path):
    layout = [(f"0000:1{i:x}:00.0", "0xe223", "0000:10:00.0", 0) for i in range(4)]
    plan = ta.build_plan(_gpus(tmp_path, layout), tp=4, pp=1)
    round_tripped = json.loads(json.dumps(plan))
    assert round_tripped["tensor_parallel_size"] == 4
    assert round_tripped["affinity_mask"] == "0,1,2,3"
