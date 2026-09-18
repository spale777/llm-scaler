"""Execute the DeepSeek V4.1 arithmetic on a real device.

Every other test in this family checks the kernels by mirroring them in Python.
This one compiles SYCL and runs it, so the numerics are validated by execution.

It does not run the shipped kernels. Those are ESIMD, which needs a device
reporting ``ext_intel_esimd``; a CPU OpenCL device does not, so on a host
without a Battlemage card they compile and cannot execute -- verified, not
assumed: the aspect query is printed by the binary and asserted here. What runs
is the same arithmetic in plain SYCL, which is enough to catch a wrong LUT, a
wrong scale decode, a wrong nibble order or a router that selects before it
transforms.

Skips when the compiler is absent, so a checkout without oneAPI still passes.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent / "exec/dsv41_fp4_reference.cpp"
_ICPX = os.environ.get("ICPX", "/opt/intel/oneapi/compiler/2026.1/bin/icpx")


def _have_compiler() -> bool:
    return Path(_ICPX).is_file() and os.access(_ICPX, os.X_OK)


pytestmark = pytest.mark.skipif(
    not _SRC.exists() or not _have_compiler(),
    reason="reference source or icpx not present",
)


_SETVARS = Path("/opt/intel/oneapi/setvars.sh")


def _oneapi_env():
    """The runtime environment the binary needs.

    pytest does not inherit the oneAPI shell setup, and the CPU OpenCL backend
    needs more than libsycl: TBB, UMF and TCM are all on its library path, and
    a partial path builds fine and then reports no device. Rather than
    reproduce that list, the environment is read out of setvars.sh, which owns
    it. Falls back to the compiler's own lib directory when setvars is absent.
    """
    env = dict(os.environ)
    if _SETVARS.is_file():
        r = subprocess.run(
            ["bash", "-c", f"source {_SETVARS} >/dev/null 2>&1 && env -0"],
            capture_output=True, timeout=300,
        )
        if r.returncode == 0:
            for entry in r.stdout.split(b"\0"):
                if not entry:
                    continue
                k, _, v = entry.decode("utf-8", "replace").partition("=")
                if k in ("LD_LIBRARY_PATH", "OCL_ICD_FILENAMES",
                         "OCL_ICD_VENDORS", "PATH"):
                    env[k] = v
            return env
    lib = Path(_ICPX).resolve().parent.parent / "lib"
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib}:{prev}" if prev else str(lib)
    return env


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    """Build once; the arithmetic is small but SYCL compilation is not."""
    out = tmp_path_factory.mktemp("dsv41_exec") / "ref"
    r = subprocess.run(
        [_ICPX, "-fsycl", "-O2", "-x", "c++", str(_SRC), "-o", str(out)],
        capture_output=True, text=True, timeout=900, env=_oneapi_env(),
    )
    if r.returncode != 0:
        pytest.fail(f"reference failed to build:\n{r.stderr[-2000:]}")
    return out


@pytest.fixture(scope="module")
def run(binary):
    r = subprocess.run([str(binary)], capture_output=True, text=True,
                       timeout=900, env=_oneapi_env())
    if r.returncode != 0:
        if "No device of requested type" in (r.stdout + r.stderr):
            pytest.skip("no SYCL device available to execute on")
        pytest.fail(f"reference failed at runtime:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def test_the_reference_transcribes_the_shipped_bit_trick():
    """The executable must carry the kernel's arithmetic, not a LUT.

    A reference that decoded E2M1 from a table would pass its own checks while
    telling us nothing about the expression fp4_dequant.h actually ships.
    """
    src = _SRC.read_text()
    kernel = (Path(__file__).resolve().parents[1]
              / "csrc/deepseek_v41/fp4_dequant.h").read_text()
    assert "0x3C00 + ((m - 2) << 9)" in src, (
        "the reference no longer uses the shipped affine expression"
    )
    assert "0x3C00 + ((m - 2) << 9)" in kernel, (
        "the kernel's expression changed; the reference is now checking "
        "something else"
    )
    # The kernel patches m==1 and m==0 with a vector merge and the reference
    # with a scalar assignment, so the constants and their conditions are
    # compared rather than the syntax around them.
    import re as _re
    for bits, m in (("0x3800", 1), ("0x0000", 0)):
        assert _re.search(rf"{bits}.*m == {m}|m == {m}.*{bits}", kernel), (
            f"the kernel no longer patches m=={m} with {bits}")
        assert _re.search(rf"m == {m}\) res = {bits}", src), (
            f"the reference no longer patches m=={m} with {bits}")
    # The saturation that keeps a scale below the fp16 Inf encoding.
    assert "142" in src and "142" in kernel


def test_it_ran_on_a_device(run):
    assert "device:" in run
    assert "all checks passed on a real device" in run


def test_the_esimd_aspect_is_reported(run):
    """Whether the shipped kernels could run here is a fact worth recording.

    A CPU device reports 0, which is why the ESIMD kernels are checked by
    compilation and CPU mirrors rather than by execution.
    """
    assert "ext_intel_esimd:" in run


def test_e2m1_decodes_exactly_on_device(run):
    """All 16 nibbles, decoded by the shipped bit trick, not a table."""
    assert "E2M1: 16 nibbles decoded on device, exact" in run


def test_ue8m0_decodes_exactly_on_device(run):
    """All 256 bytes, including the saturation that keeps a scale below the
    fp16 Inf encoding -- an Inf scale against an E2M1 zero gives NaN."""
    assert "UE8M0: 256 bytes decoded on device, exact" in run


def test_the_fp4_gemm_matches_a_host_reference(run):
    """The device result is compared against a double-precision host sum
    computed independently, so a wrong nibble order or a mispaired scale shows
    up as error rather than as a differently-shaped tensor."""
    line = next(l for l in run.splitlines() if l.startswith("FP4 GEMM:"))
    err = float(line.rsplit(" ", 1)[1])
    assert err < 1e-5, f"FP4 GEMM diverges on device: {line}"


def test_the_router_runs_over_the_real_expert_count(run):
    """384 experts and top-6, which is the model's own setting."""
    assert "noaux_tc router: 4 tokens over 384 experts executed on device" in run
