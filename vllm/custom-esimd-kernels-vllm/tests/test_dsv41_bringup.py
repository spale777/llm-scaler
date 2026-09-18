"""The hardware bring-up script must behave correctly without hardware.

Its whole value is that someone can run one command when a B70 arrives. That
only holds if it does something sane here too: skip what it cannot check, keep
a skip distinct from a pass, and still validate a checkpoint index, which needs
no GPU at all.

The trap it is written against is a harness that reports success on the wrong
machine. A run with no device must not read as a clean bill.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "vllm/tools/platform/dsv41_bringup.py"

pytestmark = pytest.mark.skipif(not _SCRIPT.exists(),
                                reason="bring-up script not present")


def _run(*args):
    return subprocess.run([sys.executable, str(_SCRIPT), *args],
                          capture_output=True, text=True, timeout=600,
                          cwd=str(_ROOT))


def test_it_runs_without_a_device_and_skips():
    """No GPU is not an error; it is a skip with a reason."""
    r = _run("--json")
    assert r.returncode == 0, r.stderr
    results = json.loads(r.stdout)
    assert results, "no stages ran"
    for stage in results:
        assert stage["status"] in ("pass", "fail", "skip")
        if stage["status"] == "skip":
            assert stage["detail"], f"{stage['stage']} skipped with no reason"


def test_a_skip_is_not_counted_as_a_pass():
    """A harness that prints 'all passed' on a machine with no device is
    worse than no harness."""
    r = _run()
    assert r.returncode == 0
    tail = r.stdout.strip().splitlines()[-1]
    assert "skipped" in tail
    assert tail.startswith("0 passed") or " 0 failed" in tail


def test_the_gpu_stages_skip_rather_than_fail_here():
    """A failure would mean the kernels are broken; they are untested."""
    r = _run("--json")
    by = {s["stage"]: s for s in json.loads(r.stdout)}
    for stage in ("device", "esimd_aspect", "numerics", "multicard"):
        assert by[stage]["status"] == "skip", (
            f"{stage} reported {by[stage]['status']} on a host with no XPU")


def test_the_checkpoint_stage_needs_no_gpu(tmp_path):
    """Validating an index is pure planning, so it must work anywhere."""
    index = {
        "metadata": {"total_size": 4096},
        "weight_map": {
            "embed.weight": "a.safetensors",
            "norm.weight": "a.safetensors",
            "head.weight": "a.safetensors",
            "layers.0.attn_norm.weight": "a.safetensors",
            "layers.0.ffn.experts.0.w1.weight": "b.safetensors",
        },
    }
    p = tmp_path / "model.safetensors.index.json"
    p.write_text(json.dumps(index))
    r = _run("--stage", "checkpoint", "--index", str(p), "--json")
    assert r.returncode == 0, r.stdout + r.stderr
    got = json.loads(r.stdout)[0]
    assert got["status"] == "pass", got
    assert got["data"]["tensors"] == 5
    assert got["data"]["shards"] == 2


def test_an_unrecognised_tensor_fails_the_checkpoint_stage(tmp_path):
    """A tensor the placement plan does not know is silently absent at
    runtime, so it must stop the run here instead."""
    index = {"weight_map": {"layers.0.attn.brand_new_thing.weight": "a.st"}}
    p = tmp_path / "model.safetensors.index.json"
    p.write_text(json.dumps(index))
    r = _run("--stage", "checkpoint", "--index", str(p), "--json")
    assert r.returncode == 1, "an unknown tensor must fail the run"
    got = json.loads(r.stdout)[0]
    assert got["status"] == "fail"
    assert "unrecognised" in got["detail"]


def test_a_missing_index_skips_rather_than_crashing(tmp_path):
    r = _run("--stage", "checkpoint", "--index",
             str(tmp_path / "nope.json"), "--json")
    assert r.returncode == 0
    assert json.loads(r.stdout)[0]["status"] == "skip"


def test_stages_can_be_selected():
    r = _run("--stage", "device", "--json")
    assert r.returncode == 0
    results = json.loads(r.stdout)
    assert len(results) == 1 and results[0]["stage"] == "device"


def test_it_measures_nothing():
    """Every timing claim in this repo needs a benchmark behind it, and this
    is not one; a stage that reported throughput would read as one."""
    src = _SCRIPT.read_text()
    assert "No stage measures throughput" in src
    for banned in ("perf_counter", "time.time(", "GB/s", "TFLOP"):
        assert banned not in src, (
            f"{banned} suggests this harness has started timing things")
