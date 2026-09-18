"""Every patch must be well formed enough for git to parse and apply it.

A hunk whose `@@ -a,b +c,d @@` counts disagree with the lines beneath it is
rejected outright, so an edit that adds or removes a line without updating the
header breaks the whole patch. The failure surfaces at deploy time, inside a
container build, long after the edit looked right.

`git apply --check` is the authority rather than a diff parser written here: a
hand-rolled one gets the trailing-blank-line and "\\ No newline" cases wrong and
then reports every vendored patch as broken.
"""

import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_PATCHES = sorted(
    list((_ROOT / "sglang/patches").glob("*.patch"))
    + list((_ROOT / "vllm/patches").glob("*.patch"))
)


@pytest.mark.skipif(not _PATCHES, reason="no patches present")
@pytest.mark.parametrize("path", _PATCHES, ids=lambda p: p.name)
def test_patch_is_structurally_valid(path):
    """`git apply --numstat` parses every hunk without needing the target tree.

    It reports the added/removed counts per file, which requires reading each
    hunk header and its body; a header that disagrees with its body makes it
    fail. It does not need the files the patch applies to, so this runs against
    a checkout that has never had the upstream sources.
    """
    text = path.read_text()
    hunks = text.count("\n@@ ")
    assert hunks > 0, (
        f"{path.name} contains no hunks; it is not a patch any more"
    )

    r = subprocess.run(
        ["git", "apply", "--numstat", str(path)],
        capture_output=True, text=True, cwd=str(_ROOT),
    )
    assert r.returncode == 0, (
        f"{path.name}: git cannot parse its {hunks} hunk(s), so the patch "
        f"will not apply:\n{r.stderr.strip()}"
    )
