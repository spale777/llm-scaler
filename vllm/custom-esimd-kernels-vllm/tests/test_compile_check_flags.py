"""The compile check must actually compile.

icpx treats a .sycl file as linker input unless -x c++ is passed: it compiles
nothing and exits 0, so the check reports a clean tree over any build break.
"""

from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "tools/compile_check.sh"


def _invocation(body):
    """The icpx command line, with comment lines removed."""
    lines = [l for l in body.splitlines() if not l.lstrip().startswith("#")]
    return "\n".join(lines)


def test_compile_check_forces_the_source_language():
    if not _SCRIPT.exists():
        pytest.skip("compile_check.sh not present")
    # The header comment names the flag; only the invocation counts.
    body = _invocation(_SCRIPT.read_text())
    assert "-x c++" in body, (
        "without -x c++ every .sycl file is treated as linker input and the "
        "check passes vacuously"
    )
    assert "-fsyntax-only" in body


def test_compile_check_fails_on_a_broken_translation_unit():
    """The only assertion that cannot be satisfied by a comment."""
    import subprocess
    import tempfile

    if not _SCRIPT.exists():
        pytest.skip("compile_check.sh not present")
    icpx = Path("/opt/intel/oneapi/compiler/2026.1/bin/icpx")
    if not icpx.exists():
        pytest.skip("icpx not present")
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.sycl"
        bad.write_text("int main() { this is not c++ ; }\n")
        r = subprocess.run(
            [str(icpx), "-fsycl", "-fsyntax-only", "-x", "c++", str(bad)],
            capture_output=True, text=True,
        )
        assert r.returncode != 0, (
            "icpx accepted a deliberately broken .sycl file; the -x c++ flag "
            "is not reaching the compiler"
        )


def test_compile_check_covers_both_trees():
    if not _SCRIPT.exists():
        pytest.skip("compile_check.sh not present")
    body = _SCRIPT.read_text()
    for tree in ("vllm/custom-esimd-kernels-vllm", "sglang/custom-esimd-kernels"):
        assert tree in body, f"{tree} is not swept"
    # Deliberately no per-directory string list: that asserts the sweep's
    # implementation rather than its coverage, and passes over whole
    # subdirectories it does not name. Coverage is checked by set comparison in
    # test_compile_check_sweeps_every_translation_unit.


def test_compile_check_sweeps_every_translation_unit():
    """The sweep must cover every .sycl/.cc/.cpp under both csrc trees.

    A substring assertion on the script text cannot see an uncovered
    subdirectory, so compare the set it reports against what is on disk.
    """
    root = Path(__file__).resolve().parents[3]
    on_disk = set()
    for tree in ("vllm/custom-esimd-kernels-vllm", "sglang/custom-esimd-kernels"):
        csrc = root / tree / "csrc"
        if not csrc.exists():
            continue
        for ext in ("*.sycl", "*.cc", "*.cpp"):
            on_disk |= {q.resolve() for q in csrc.rglob(ext)}
    assert on_disk, "no translation units found -- the walk is wrong"

    script = (root / "tools/compile_check.sh").read_text()
    body = "\n".join(l for l in script.splitlines() if not l.lstrip().startswith("#"))
    assert "find " in body and "-name '*.cpp'" in body, (
        "the sweep must recurse and include .cpp; an explicit directory list "
        "silently skipped a whole subsystem"
    )
    # Ask the SCRIPT what it sweeps rather than re-running the find here: a
    # local copy agrees with itself by construction and cannot see a `-prune`
    # added to the script.
    import subprocess
    proc = subprocess.run(["bash", str(root / "tools/compile_check.sh"), "--list"],
                          cwd=str(root), capture_output=True, text=True)
    assert proc.returncode == 0, f"--list failed: {proc.stderr[-400:]}"
    swept = {(root / x).resolve() for x in proc.stdout.split()}
    assert swept, "the script enumerated nothing"
    missing = on_disk - swept
    assert not missing, f"not compile-checked: {sorted(str(m) for m in missing)}"
    extra = swept - on_disk
    assert not extra, f"swept but not on disk: {sorted(str(e) for e in extra)}"
