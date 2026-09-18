"""Incremental SYCL builds must incorporate changes made only to headers."""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

# oneAPI installs under /opt/intel and is not on PATH, so keying the skip on
# shutil.which("icpx") alone skips this test on a machine that has the compiler.
_ICPX = shutil.which("icpx") or "/opt/intel/oneapi/compiler/2026.1/bin/icpx"


def _oneapi_env():
    """The generated ninja file invokes `icpx` by bare name, and the linked
    probe needs oneAPI's runtime libraries to start (exit 127 otherwise)."""
    env = dict(os.environ)
    bindir = Path(_ICPX).parent
    if str(bindir) not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    libs = [bindir.parent / "lib", bindir.parent.parent.parent / "lib"]
    have = env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    for lib in libs:
        if lib.is_dir() and str(lib) not in have:
            env["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + env.get(
                "LD_LIBRARY_PATH", "")
    return env


@pytest.mark.skipif(
    not os.path.exists(_ICPX)
    or shutil.which("ninja") is None
    or importlib.util.find_spec("setuptools") is None,
    reason="requires the oneAPI compiler, ninja and setuptools",
)
def test_sycl_header_change_rebuilds_object(tmp_path):
    build_module = Path(__file__).resolve().parents[1] / "esimd_build_extention.py"
    spec = importlib.util.spec_from_file_location("esimd_build_for_test", build_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    header = tmp_path / "value.h"
    source = tmp_path / "probe.sycl"
    obj = tmp_path / "probe.o"
    header.write_text("#define HEADER_VALUE 7\n")
    source.write_text('#include "value.h"\nint main() { return HEADER_VALUE; }\n')
    module._write_ninja_file(
        path=str(tmp_path / "build.ninja"),
        cflags=[], post_cflags=[], cuda_cflags=[], cuda_post_cflags=[],
        cuda_dlink_post_cflags=[], sycl_cflags=["-fsycl"],
        sycl_post_cflags=[], sycl_dlink_post_cflags=[],
        sources=[str(source)], objects=[str(obj)], ldflags=[],
        library_target=None, with_cuda=False, with_sycl=True,
    )

    def build_and_run(expected):
        subprocess.run(["ninja", "-C", str(tmp_path)], check=True,
                       capture_output=True, env=_oneapi_env())
        executable = tmp_path / f"probe_{expected}"
        subprocess.run(
            [_ICPX, "-fsycl", str(obj), "-o", str(executable)],
            check=True, capture_output=True, env=_oneapi_env(),
        )
        assert subprocess.run([str(executable)],
                              env=_oneapi_env()).returncode == expected
        # Generated temporary SYCL headers must not cause perpetual rebuilds.
        dry_run = subprocess.check_output(
            ["ninja", "-C", str(tmp_path), "-n"], text=True
        )
        assert "no work to do" in dry_run

    build_and_run(7)
    header.write_text("#define HEADER_VALUE 9\n")
    build_and_run(9)
