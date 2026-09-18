import ast
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Skip on the missing setuptools only; importorskip() here would also swallow a
# genuine ImportError from esimd_build_extention itself.
try:
    import esimd_build_extention as build_extension  # noqa: E402
except ImportError as exc:
    if (getattr(exc, "name", "") or "") != "setuptools":
        raise
    pytest.skip(
        "setuptools is not installed on this host", allow_module_level=True
    )


def test_sycl_dlink_uses_common_compile_target(monkeypatch):
    monkeypatch.setenv("TORCH_XPU_ARCH_LIST", "bmg-g21")

    flags = build_extension._get_sycl_dlink_flags(
        build_extension._COMMON_SYCL_FLAGS)

    assert "-fsycl-targets=spir64_gen,spir64" in flags
    assert '-Xs "-device bmg-g21"' in flags


def test_sycl_dlink_uses_extension_target_override(monkeypatch):
    monkeypatch.setenv("TORCH_XPU_ARCH_LIST", "bmg-g21")
    compile_flags = [
        *build_extension._COMMON_SYCL_FLAGS,
        "-fsycl-targets=spir64_gen",
    ]

    flags = build_extension._get_sycl_dlink_flags(compile_flags)

    assert "-fsycl-targets=spir64_gen" in flags
    assert "-fsycl-targets=spir64_gen,spir64" not in flags
    assert flags.count("-fsycl-targets=spir64_gen") == 1


# --- static audit of the setup files themselves (no GPU, no compiler needed) ---

_SETUP_FILES = [
    Path(__file__).resolve().parents[1] / "setup.py",
    Path(__file__).resolve().parents[1] / "setup_sycl.py",
    Path(__file__).resolve().parents[3] / "sglang" / "custom-esimd-kernels" / "setup.py",
]


def _sycl_arg_lists(path):
    """Yield every extra_compile_args["sycl"] list literal in a setup file."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and key.value == "sycl"
                    and isinstance(value, ast.List)):
                yield [
                    elt.value for elt in value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]


@pytest.mark.parametrize("path", _SETUP_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_xclang_before_driver_flags(path):
    """-funroll-loops and -fno-sycl-early-optimizations are driver flags.

    Forwarding them to cc1 with -Xclang makes icpx reject them as unknown
    arguments, so the whole extension fails to build.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    for args in _sycl_arg_lists(path):
        for i, arg in enumerate(args[:-1]):
            if arg == "-Xclang":
                assert args[i + 1] not in (
                    "-funroll-loops",
                    "-fno-sycl-early-optimizations",
                ), f"{path.name}: -Xclang must not precede {args[i + 1]}"


@pytest.mark.parametrize("path", _SETUP_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_duplicate_aot_target(path):
    """A duplicated -device bmg is forwarded to ocloc twice."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    for args in _sycl_arg_lists(path):
        assert args.count("-device bmg") <= 1, (
            f"{path.name}: -device bmg repeated in one arg list: {args}"
        )
