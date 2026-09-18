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


def _arg_text(elt):
    """The text of one compile-arg element, or None if it is not a string.

    Args are not all plain literals: the device list and the include path are
    f-strings. Rendering the interpolations as their placeholder name keeps
    every flag visible to the audits below -- a helper that skipped f-strings
    would quietly stop examining exactly the args that carry the AOT target.
    """
    if isinstance(elt, ast.Constant):
        return elt.value if isinstance(elt.value, str) else None
    if isinstance(elt, ast.JoinedStr):
        out = []
        for part in elt.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
            elif isinstance(part, ast.FormattedValue):
                v = part.value
                out.append(v.id if isinstance(v, ast.Name) else "{...}")
        return "".join(out)
    return None


def _sycl_arg_lists(path):
    """Yield every extra_compile_args["sycl"] list literal in a setup file."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and key.value == "sycl"
                    and isinstance(value, ast.List)):
                yield [a for a in (_arg_text(elt) for elt in value.elts)
                       if a is not None]


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
        n = sum(a.count("-device ") for a in args)
        assert n <= 1, (
            f"{path.name}: -device repeated in one arg list: {args}"
        )


@pytest.mark.parametrize("path", _SETUP_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_module_names_an_aot_target(path):
    """A module without -fsycl-targets JITs its kernels on first call.

    The two trees build the same sources: sglang AOT-compiled all of its
    modules while vllm left five on the JIT path, including moe_int4.sycl
    (4151 lines) and eagle.sycl, both of which run per decoded token. The
    first call to each then pays a full device compile.

    The target names both dies: B60 is BMG-G21 and B70 is BMG-G31. ocloc
    accepts the pair and emits an image per die, so each part gets code built
    for it rather than one family binary. Note the AOT step needs `ocloc`,
    which the devel base image provides and a bare build host may not.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    jit = []
    for args in _sycl_arg_lists(path):
        if not any("-fsycl-targets" in a for a in args):
            jit.append(args)
    assert not jit, (
        f"{path.name}: {len(jit)} module(s) name no AOT target and will JIT: "
        f"{jit}"
    )


@pytest.mark.parametrize("path", _SETUP_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_at_most_one_xs_options_group(path):
    """ocloc takes one option string; two -Xs groups each carrying -options
    leaves it undefined which survives.

    sglang passed `-Xs "-device bmg -options -doubleGRF"` and, later in the
    same list, `-Xs "-options -cl-intel-enable-auto-fma"`. Whether ocloc merges
    them or the second replaces the first decides whether -doubleGRF reaches
    the compiler at all, so the flag a module is documented to need may not be
    the flag it gets.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    for args in _sycl_arg_lists(path):
        with_options = [
            args[i + 1] for i, a in enumerate(args[:-1])
            if a == "-Xs" and "-options" in args[i + 1]
        ]
        assert len(with_options) <= 1, (
            f"{path.name}: {len(with_options)} -Xs groups carry -options "
            f"({with_options}); merge them into one string"
        )


@pytest.mark.parametrize("path", _SETUP_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_one_env_var_has_one_default(path):
    """Two modules reading OMNI_XPU_DEVICE disagreed on its default.

    One defaulted to ptl-u (Xe3) and its neighbour to bmg, so with the variable
    unset one module was AOT-compiled for an architecture this repo does not
    ship, and setting it for one module silently retargeted the other.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    import re as _re
    src = path.read_text()
    # The default is either a quoted literal or a named constant; matching only
    # the quoted form would stop examining the file the moment it is factored
    # into a constant, which is exactly when a drift could reappear unseen.
    defaults = _re.findall(
        r'os\.environ\.get\(\s*"OMNI_XPU_DEVICE"\s*,\s*(?:"([^"]+)"|([A-Za-z_]\w*))\s*\)',
        src)
    defaults = [a or b for a, b in defaults]
    if _re.search(r'os\.environ\.get\(\s*"OMNI_XPU_DEVICE"', src):
        assert defaults, (
            f"{path.name}: OMNI_XPU_DEVICE is read but no default was matched; "
            "the anchor moved and this test stopped checking it"
        )
    assert len(set(defaults)) <= 1, (
        f"{path.name}: OMNI_XPU_DEVICE has conflicting defaults {sorted(set(defaults))}; "
        "one variable cannot select two architectures"
    )


@pytest.mark.parametrize("path", _SETUP_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_aot_target_names_both_battlemage_dies(path):
    """B60 and B70 are different dies, so name both.

    `bmg` is the family target: it builds one image that runs on either part,
    which is correct but forgoes per-die specialization, and any tuning keyed
    on `intel_gpu_bmg_g31` would silently not match. ocloc takes a comma list
    and emits an image per die -- verified by building the MoE translation
    unit for `bmg-g21,bmg-g31`, which reports a build per die and produces an
    object 2.3x the single-die size.

    The list is defined once per setup file so the modules cannot drift apart.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    assert 'BMG_DEVICES = "bmg-g21,bmg-g31"' in src, (
        f"{path.name}: the AOT device list must be a single named constant "
        "covering both dies"
    )
    checked = 0
    for args in _sycl_arg_lists(path):
        for a in args:
            if "-device " not in a:
                continue
            checked += 1
            assert "BMG_DEVICES" in a or "_DEV" in a, (
                f"{path.name}: `{a}` names a device literally instead of "
                "using BMG_DEVICES, so it will not track the die list"
            )
    assert checked >= 3, (
        f"{path.name}: examined {checked} device flag(s); the anchor moved "
        "and this test stopped checking them"
    )
