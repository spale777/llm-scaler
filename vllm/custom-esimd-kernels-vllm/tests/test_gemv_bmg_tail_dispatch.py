"""Contract test for select_bmg / GEMV_fp8_pert_bmg_host dispatch.

An index-arithmetic test, not a numerical one: it mirrors the host dispatcher's
shape selection in Python and asserts that no (N, K) pair reaches a launch
configuration that drops part of the K dimension. A residue outside
{8,16,32,64,128} must route to the masked-tail kernel (vl_tail = -1); the
no-tail kernel silently loses kp % vl_big elements per thread.
"""

import re
from pathlib import Path

import pytest

_HEADERS = [
    Path(__file__).resolve().parents[1] / "csrc/xpu/esimd_kernels/fp8_GEMV_bmg.h",
    Path(__file__).resolve().parents[3]
    / "sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels/fp8_GEMV_bmg.h",
]

_POW2_TAILS = (8, 16, 32, 64, 128)


def _hw_threads():
    """Read the occupancy target out of the header, so the mirror cannot
    drift from the dispatcher it stands in for."""
    m = re.search(r"BMG_HW_THREADS\s*=\s*(\d+)", _HEADERS[0].read_text())
    assert m, "BMG_HW_THREADS not found in fp8_GEMV_bmg.h"
    return int(m.group(1))


BMG_HW_THREADS = _hw_threads()


def select_bmg(n, k):
    """Python mirror of select_bmg() in fp8_GEMV_bmg.h.

    Returns (ks, vl_big, vl_tail) where vl_tail == -1 means "masked tail".
    """
    if n * 8 <= BMG_HW_THREADS:
        target_ks = 8
    elif n * 4 <= BMG_HW_THREADS:
        target_ks = 4
    elif n * 2 <= BMG_HW_THREADS:
        target_ks = 2
    else:
        target_ks = 1

    ks = 1
    s = target_ks
    while s >= 1:
        if k % s == 0:
            ks = s
            break
        s //= 2

    kp = k // ks
    for c in (256, 128, 64, 32):
        if kp >= c:
            tail = kp - (kp // c) * c
            if tail == 0:
                return ks, c, 0
            if tail in _POW2_TAILS:
                return ks, c, tail
    return ks, 32, -1


def covered_elements(k, ks, vl_big, vl_tail):
    """How many of the K elements a launch actually accumulates."""
    kp = k // ks
    kp_full = (kp // vl_big) * vl_big
    if vl_tail == 0:
        per_thread = kp_full          # no-tail kernel: residue is dropped
    elif vl_tail > 0:
        per_thread = kp_full + vl_tail
    else:
        per_thread = kp               # masked tail covers the whole chunk
    return per_thread * ks


@pytest.mark.parametrize("k", [1044, 1056, 1152, 2816, 5120, 4096, 33, 60, 100, 8191])
@pytest.mark.parametrize("n", [512, 1024, 3072, 4096])
def test_dispatch_covers_every_k_element(n, k):
    ks, vl_big, vl_tail = select_bmg(n, k)
    covered = covered_elements(k, ks, vl_big, vl_tail)
    assert covered == k, (
        f"N={n} K={k}: dispatch (ks={ks}, vl_big={vl_big}, vl_tail={vl_tail}) "
        f"covers {covered}/{k} elements — {k - covered} silently dropped"
    )


def test_no_k_in_range_drops_data():
    """Exhaustive sweep over K: coverage must be total, not typical."""
    dropped = []
    for k in range(32, 8193):
        ks, vl_big, vl_tail = select_bmg(4096, k)
        if covered_elements(k, ks, vl_big, vl_tail) != k:
            dropped.append(k)
    assert not dropped, f"{len(dropped)} K values still drop data, e.g. {dropped[:8]}"


@pytest.mark.parametrize("path", _HEADERS, ids=lambda p: p.parents[3].name)
def test_header_has_masked_tail_kernel(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    assert "GEMV_fp8_pert_bmg_masked_tail_kernel" in src, (
        "masked-tail kernel missing — arbitrary residues cannot be handled"
    )
    assert "vl_tail = -1" in src, "select_bmg no longer signals the masked path"


@pytest.mark.parametrize("path", _HEADERS, ids=lambda p: p.parents[3].name)
def test_no_silent_notail_fallback(path):
    """Every `else` fallback must reach a tail-aware launch: a catch-all
    `else { LAUNCH_NOTAIL(...) }` drops the K residue."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    for m in re.finditer(r"else\s*\{\s*(LAUNCH_\w+)", src):
        assert m.group(1) != "LAUNCH_NOTAIL", (
            f"{path.name}: catch-all `else` falls back to LAUNCH_NOTAIL, which "
            "drops the K residue"
        )
