"""Pin the transport facts pipeline parallelism rests on.

PP over two switches needs point-to-point send/recv between the stages, and
nothing else: no collective crosses the PP boundary, which is what makes a
cross-switch collective hang structurally impossible.

The rendered PyTorch docs table lists P2P as unsupported for XCCL. That table
is stale -- ProcessGroupXCCL defines both operations and the shipped
libtorch_xpu.so exports them. This is checked against the installed binary
rather than restated from a document, because the whole PP design is void if
it is ever untrue for a pinned torch.

No GPU and no initialized process group are needed: symbol presence and the
Python API surface are both static properties of the install.
"""

import glob
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _torch_lib_dir():
    return Path(torch.__file__).resolve().parent / "lib"


def test_distributed_exposes_point_to_point():
    import torch.distributed as dist

    missing = [n for n in ("send", "recv", "isend", "irecv", "batch_isend_irecv")
               if not hasattr(dist, n)]
    assert not missing, f"torch.distributed is missing P2P entry points: {missing}"


def test_xccl_backend_is_available():
    import torch.distributed as dist

    if not hasattr(dist, "is_xccl_available"):
        pytest.skip("this torch predates is_xccl_available()")
    assert dist.is_xccl_available(), (
        "XCCL is not available in this torch build, so no XPU process group "
        "can be formed and neither TP nor PP can run"
    )


def test_process_group_xccl_exports_send_and_recv():
    """The stale-docs claim, checked against the binary that ships."""
    libs = glob.glob(str(_torch_lib_dir() / "libtorch_xpu.so"))
    if not libs:
        pytest.skip("libtorch_xpu.so not present in this install")
    nm = subprocess.run(
        ["nm", "-D", "--defined-only", libs[0]],
        capture_output=True, text=True,
    )
    if nm.returncode != 0:
        pytest.skip("nm unavailable or could not read the library")

    symbols = nm.stdout
    for op in ("send", "recv"):
        # Itanium mangling: N4c10d16ProcessGroupXCCL<len><name>E
        needle = f"c10d16ProcessGroupXCCL{len(op)}{op}E"
        assert needle in symbols, (
            f"ProcessGroupXCCL::{op} is not exported by libtorch_xpu.so; "
            "pipeline parallelism has no transport on this build"
        )
