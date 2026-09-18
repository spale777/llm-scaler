"""Static guard: every kernel must submit to its operands' device.

`c10::xpu::getCurrentXPUStream()` with no argument returns the stream of the
*current* device, which on a multi-card run is not necessarily where the input
tensors live. The result is correct on one GPU and wrong on every larger
configuration, so this is a prerequisite for tensor- or pipeline-parallelism
rather than a perf item.
"""

import re
from pathlib import Path

import pytest

from srctext import code, tokens

_ROOT = Path(__file__).resolve().parents[3]
_SOURCES = sorted(
    list((_ROOT / "sglang/custom-esimd-kernels/csrc").rglob("*.sycl"))
    + list((_ROOT / "sglang/custom-esimd-kernels/csrc").rglob("*.h"))
    + list((_ROOT / "vllm/custom-esimd-kernels-vllm/csrc").rglob("*.sycl"))
    + list((_ROOT / "vllm/custom-esimd-kernels-vllm/csrc").rglob("*.h"))
)


def _rel(p):
    return str(p.relative_to(_ROOT))


@pytest.mark.skipif(not _SOURCES, reason="kernel sources not present")
def test_no_device_index_less_stream():
    offenders = []
    for path in _SOURCES:
        src = path.read_text(errors="ignore")
        for m in re.finditer(r"getCurrentXPUStream\(\s*(?:\)|c10::xpu::current_device\(\)|at::xpu::current_device\(\))", src):
            line = src[: m.start()].count("\n") + 1
            offenders.append(f"{_rel(path)}:{line}")
    assert not offenders, (
        "getCurrentXPUStream() without a device index submits to the current "
        "device, not the tensors' device:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.skipif(not _SOURCES, reason="kernel sources not present")
def test_no_default_gpu_selector_outside_allreduce():
    """A fresh queue from gpu_selector_v picks the default GPU, not the rank's.

    torch_extension_ar.cc is excluded; its queue handling is pinned by
    test_custom_ar_contract.
    """
    offenders = []
    for path in _SOURCES:
        if path.name == "torch_extension_ar.cc":
            continue
        src = path.read_text(errors="ignore")
        for m in re.finditer(r"gpu_selector_v|gpu_selector\{\}", src):
            line = src[: m.start()].count("\n") + 1
            offenders.append(f"{_rel(path)}:{line}")
    assert not offenders, (
        "constructing a queue from the default GPU selector ignores the rank's "
        "device:\n  " + "\n  ".join(offenders)
    )
