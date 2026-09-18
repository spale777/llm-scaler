#!/usr/bin/env bash
# Syntax-check every kernel translation unit in both trees.
#
# icpx does not recognise .sycl as a source extension: without -x c++ it treats
# the file as linker input, compiles nothing and exits 0. Keep that flag.
set -uo pipefail

ICPX=${ICPX:-/opt/intel/oneapi/compiler/2026.1/bin/icpx}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${VENV:-"$ROOT/.venv"}
TI=$("$VENV/bin/python" -c "import torch,os;print(os.path.dirname(torch.__file__)+'/include')")
PYI=$("$VENV/bin/python" -c "import sysconfig;print(sysconfig.get_paths()['include'])")
if [ -z "$TI" ] || [ -z "$PYI" ]; then
  echo "FAIL: could not resolve torch/python include paths" >&2
  exit 1
fi
if ! [ -x "$ICPX" ]; then
  echo "FAIL: compiler not executable: $ICPX" >&2
  exit 1
fi

cd "$ROOT"
RESULTS=$(mktemp)
JOBS=${JOBS:-18}

check_one() {
  tree=$1
  tu=$2
  # Failure is the exit status, not the word "error:": an ICE, a missing
  # compiler and a SIGKILL all emit no "error:" line.
  out=$("$ICPX" -fsycl -fsyntax-only -x c++ \
        -I "$tree" -I "$tree/csrc" -I "$tree/include" \
        -I"$TI" -I"$TI/torch/csrc/api/include" -I"$PYI" \
        "$tu" 2>&1)
  rc=$?
  n=$(printf '%s' "$out" | grep -c "error:")
  if [ "$rc" != "0" ] || [ "$n" != "0" ]; then
    echo "FAIL(rc=$rc errors=$n): $tu"
  else
    echo "ok: $tu"
  fi
}
export -f check_one
export ICPX TI PYI

# Single source of truth for which translation units exist. `--list` exposes it
# so the coverage test diffs THIS enumeration against the tree instead of
# re-implementing the walk and agreeing with itself.
list_tus() {
for tree in vllm/custom-esimd-kernels-vllm sglang/custom-esimd-kernels; do
  [ -d "$tree" ] || continue
  # Must stay recursive and keep .cpp: subsystems live in subdirectories of
  # csrc/xpu, and an explicit directory list silently drops them.
  find "$tree/csrc" \
       \( -name '*.sycl' -o -name '*.cc' -o -name '*.cpp' \) -print \
     2>/dev/null | sed "s|^|$tree |"
done
}

if [ "${1:-}" = "--list" ]; then
  list_tus | awk '{print $2}'
  exit 0
fi

list_tus | xargs -P "$JOBS" -n 2 bash -c 'check_one "$0" "$1"' > "$RESULTS"

fail=$(grep -c '^FAIL' "$RESULTS" || true)
pass=$(grep -c '^ok:' "$RESULTS" || true)
grep '^FAIL' "$RESULTS" || true
echo "PASS=$pass FAIL=$fail"
rm -f "$RESULTS"
[ "$fail" = "0" ]
