"""Make the suite runnable on a host without a built extension or an XPU.

Most tests import `custom_esimd_kernels_sglang` at module scope, so without a
compiled .so they abort *collection* and the CPU-only contract tests never run.
Nothing here hides a real failure: an ImportError for any other module, or any
assertion failure, still fails the run.
"""

import pytest

_EXT_MODULES = (
    "custom_esimd_kernels_vllm",
    "custom_esimd_kernels_sglang",
    "custom_esimd_kernels_vllm_ar",
    "q4_0_quant_ops",
    "esimd_topk_v2",
)


def _is_missing_extension(excinfo) -> bool:
    if not isinstance(excinfo.value, ImportError):
        return False
    missing = getattr(excinfo.value, "name", "") or str(excinfo.value)
    return any(mod in missing for mod in _EXT_MODULES)


def _extension_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("custom_esimd_kernels_sglang") is not None


# Collected once: this hook runs for every file in the directory.
_HAVE_EXT = _extension_available()
_IGNORED = []


def pytest_ignore_collect(collection_path, config):
    """Skip modules that import the extension when it is not built."""
    if _HAVE_EXT:
        return False
    path = getattr(collection_path, "name", str(collection_path))
    if not path.startswith("test_"):
        return False
    try:
        src = collection_path.read_text(errors="ignore")
    except (OSError, AttributeError):
        return False
    if any(f"import {mod}" in src or f"from {mod}" in src for mod in _EXT_MODULES):
        _IGNORED.append(path)
        return True
    return False


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Name the dropped files: a count that excludes most of the directory
    otherwise reads as a full run."""
    if not _IGNORED:
        return
    import pathlib

    here = pathlib.Path(__file__).parent
    all_tests = {p.name for p in here.glob("test_*.py")}
    collected = {
        pathlib.Path(str(item.fspath)).name
        for item in terminalreporter.stats.get("", [])
        + terminalreporter.stats.get("passed", [])
        + terminalreporter.stats.get("failed", [])
        + terminalreporter.stats.get("skipped", [])
    }
    silent = sorted(all_tests - set(_IGNORED) - collected)

    terminalreporter.write_sep(
        "-", f"{len(_IGNORED)} file(s) not collected: extension not built"
    )
    for name in sorted(_IGNORED):
        terminalreporter.write_line(f"  {name}")
    # Every file here is a DEVICE test, so the whole directory drops on a CPU
    # host. Name where this tree's static contracts live, so an empty run is
    # not read as a hole in coverage.
    terminalreporter.write_line(
        "  note: sglang static contracts are enforced from "
        "vllm/custom-esimd-kernels-vllm/tests (12 modules assert over this "
        "tree); the files above need torch + a built extension + an XPU."
    )
    if silent:
        # A file with no test_* functions is reported by nobody, so it reads
        # as covered while asserting nothing.
        terminalreporter.write_sep(
            "-", f"{len(silent)} file(s) collected no tests"
        )
        for name in silent:
            terminalreporter.write_line(f"  {name}")


def pytest_collection_modifyitems(config, items):
    """Skip XPU-marked tests when no XPU device is available."""
    try:
        import torch

        has_xpu = bool(getattr(torch, "xpu", None)) and torch.xpu.is_available()
    except Exception:
        has_xpu = False

    if has_xpu:
        return
    skip = pytest.mark.skip(reason="no XPU device on this host")
    for item in items:
        if "xpu" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "xpu: requires a real XPU device")


@pytest.fixture
def requires_xpu():
    try:
        import torch

        if getattr(torch, "xpu", None) and torch.xpu.is_available():
            return
    except Exception:
        pass
    pytest.skip("no XPU device on this host")
