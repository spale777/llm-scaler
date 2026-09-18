"""The suite policing itself.

Two ways a test in this suite can be green while checking nothing, both
decidable from the AST: anchor loss (a loop that `continue`s past non-matching
sites and asserts only inside itself, so a rename in both trees empties it),
and a module that never opens a file and so asserts only about Python.
"""

import ast
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent

# Exempt from the post-loop counter: a rename in the sources cannot silently
# empty these. The first two iterate a fixed literal domain; the last two scan
# parsed source but assert on every match with no `continue`, so losing the
# anchor fails rather than passes.
_FIXED_DOMAIN = {
    "test_compile_check_sweeps_every_translation_unit",  # asserts set equality
    "test_patch_hunk_headers_match_their_bodies",        # walks the whole file
    "test_fp8_gemv_floor_matches_its_ladder",            # scans source; asserts per match
    "test_every_simd_local_array_fits_the_grf_budget",   # scans source; asserts per match
}

# Sweeps whose passing state is an empty result set: they collect violations and
# assert there are none, so a progress guard would assert the opposite of the
# contract. Their anchor is the defect pattern rather than a healthy site, so
# losing it cannot hide a regression.
_ABSENCE_SWEEPS = {
    "test_no_gate_proxies_a_dtype_by_its_width",
    "test_gate_comments_do_not_quote_superseded_kernel_messages",
    "test_shape_pinned_kernels_have_a_shape_term_in_their_gate",
    "test_no_default_gpu_selector_outside_allreduce",
    "test_no_void_op_declares_zero_mutable_slots",
}


_GUARD_NAMES = {"seen", "found", "checked", "n_checked", "hits", "count",
                "examined", "hosts"}


def _skips_before_asserting(fn):
    """True when a `continue` can skip past an assertion in the same loop.

    The iterable is not consulted: the dominant shape is
    `for root in (_VLLM, _SGL):` with the scan in the body, so it never names a
    scan call. Anchor loss is decidable from the body alone.
    """
    for node in ast.walk(fn):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        conts, asserts = [], []
        for inner in ast.walk(node):
            if isinstance(inner, ast.Continue):
                conts.append(inner.lineno)
            elif isinstance(inner, ast.Assert):
                asserts.append(inner.lineno)
        if conts and any(a > min(conts) for a in asserts):
            return True
    return False


def _has_progress_guard(fn):
    """The counter must be ASSERTED, not merely mentioned: matching the name
    anywhere accepts a loop variable called `count`, or `checked = 0` with no
    assertion at all."""
    asserted = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            asserted |= {n.id for n in ast.walk(node.test)
                         if isinstance(n, ast.Name)}
    written = set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            written |= {t.id for t in tgts if isinstance(t, ast.Name)}
    if asserted & written & _GUARD_NAMES:
        return True
    # A named counter is not the only sound form: `assert any(...)` or
    # `assert len(forms) == 2` over the collection the loop filled proves the
    # loop ran just as well, and demanding a name from a fixed list would be
    # the same presence-vs-use weakness this file polices.
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assert):
            continue
        for call in ast.walk(node.test):
            if isinstance(call, ast.Call) and getattr(
                    call.func, "id", "") in ("any", "all", "len"):
                # Position is what makes it a witness: inside the skipping
                # loop the same `continue` skips the assertion too, so it must
                # sit past the end of the loop that owns that `continue`.
                if node.lineno > _skipping_loop_end(fn):
                    return True
    return False


def _skipping_loop_end(fn):
    """Last line of the loop whose `continue` can skip the SOURCE SCAN.

    Only a loop that both reads a source file and skips sites can be emptied by
    a rename, so only its end is the bar a witness must clear; a wider window
    flags a correct test whose witness sits in a later, unrelated loop.
    """
    def owns(loop, cont):
        """True when `cont` binds to `loop` -- i.e. no nearer loop between."""
        for mid in ast.walk(loop):
            if mid is loop or not isinstance(mid, (ast.For, ast.While)):
                continue
            if any(n is cont for n in ast.walk(mid)):
                return False        # a nested loop catches it first
        return True

    end = 0
    for loop in ast.walk(fn):
        if not isinstance(loop, (ast.For, ast.While)):
            continue
        body = ast.dump(loop)
        if "read_text" not in body and "exists" not in body:
            continue
        # Only a `continue` that binds here can empty this loop: an outer loop
        # containing an inner loop's `continue` is not skippable by it.
        if not any(owns(loop, c) for c in ast.walk(loop)
                   if isinstance(c, ast.Continue)):
            continue
        end = max(end, max(n.lineno for n in ast.walk(loop)
                           if hasattr(n, "lineno")))
    return end


@pytest.mark.parametrize("path", sorted(_TESTS.glob("test_*.py")),
                         ids=lambda p: p.name)
def test_source_scanning_loops_assert_they_examined_something(path):
    """A loop over parsed sources must prove it ran: without a witness past
    the loop, a rename in both trees empties it silently."""
    if path.name == Path(__file__).name:
        pytest.skip("this file")
    tree = ast.parse(path.read_text())
    offenders = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]:
        if fn.name in _FIXED_DOMAIN or fn.name in _ABSENCE_SWEEPS:
            continue
        if not _skips_before_asserting(fn):
            continue
        if not _has_progress_guard(fn):
            offenders.append(f"{fn.name} (line {fn.lineno})")
    assert not offenders, (
        f"{path.name}: these tests loop over parsed source, skip non-matching "
        "sites with `continue`, and never assert that any site was examined. "
        "Renaming the anchor in both trees disarms them silently: "
        + ", ".join(offenders)
    )


@pytest.mark.parametrize("path", sorted(_TESTS.glob("test_*.py")),
                         ids=lambda p: p.name)
def test_every_test_module_reads_something_it_asserts_about(path):
    """A module of pure-Python arithmetic asserts nothing about the tree."""
    if path.name in (Path(__file__).name, "conftest.py"):
        pytest.skip("this file")
    src = path.read_text()
    if "read_text" in src or "importorskip" in src or "import torch" in src:
        return
    tree = ast.parse(src)
    has_test = any(isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                   for n in ast.walk(tree))
    assert not has_test, (
        f"{path.name} defines tests but never reads a source file; a test that "
        "recomputes a formula in Python and compares it to a Python constant "
        "is a tautology that reads as coverage"
    )
