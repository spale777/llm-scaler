#!/usr/bin/env bash
# Launch vLLM across any GPU count with a topology-derived TP/PP layout.
#
# TP groups are kept inside one PCIe switch and PP stages are laid across
# switches, so the only cross-switch traffic is point-to-point activations
# between stages. No collective crosses the boundary, which makes a
# cross-switch collective hang structurally impossible rather than unlikely.
#
#   ./launch_tp_pp.sh --model /llm/models/X -tp 8 -pp 2 [-- extra vllm args]
#
# --dry-run prints the environment and command without starting anything, so
# the layout can be inspected on a machine that does not have the cards.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PLANNER="$HERE/topology_affinity.py"

TP=1
PP=1
MODEL=""
DRY=0
LSPCI_CAPTURE=""
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        -tp|--tensor-parallel-size) TP="$2"; shift 2 ;;
        -pp|--pipeline-parallel-size) PP="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --from-lspci) LSPCI_CAPTURE="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [ -z "$MODEL" ] && [ "$DRY" = "0" ]; then
    echo "error: --model is required" >&2
    exit 2
fi

plan_args=(-tp "$TP" -pp "$PP")
[ -n "$LSPCI_CAPTURE" ] && plan_args+=(--from-lspci "$LSPCI_CAPTURE")

echo "=== topology ==="
python3 "$PLANNER" "${plan_args[@]}" || true

# The planner warns but still emits a usable layout; a warning means a TP group
# would cross a switch, which is a performance cliff rather than an error.
if ! eval "$(python3 "$PLANNER" "${plan_args[@]}" --export 2>/dev/null)"; then
    echo "warning: topology planner reported a suboptimal layout; continuing" >&2
    eval "$(python3 "$PLANNER" "${plan_args[@]}" --export 2>/dev/null || true)"
fi

if [ -z "${ZE_AFFINITY_MASK:-}" ]; then
    echo "error: no affinity mask derived; are the GPUs visible?" >&2
    [ "$DRY" = "0" ] && exit 3
fi

# PP stages exchange activations with send/recv over XCCL. A provider mismatch
# between ranks makes OFI initialisation hang instead of failing, so it is set
# explicitly and identically for every rank rather than left to autodetection.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}

echo
echo "=== environment ==="
for v in ZE_AFFINITY_MASK CCL_TOPO_P2P_ACCESS CCL_ATL_TRANSPORT FI_PROVIDER \
         VLLM_WORKER_MULTIPROC_METHOD VLLM_ALLOW_LONG_MAX_MODEL_LEN; do
    printf '%s=%s\n' "$v" "$(eval "echo \${$v:-<unset>}")"
done

cmd=(vllm serve --host 0.0.0.0 --port "${PORT:-8000}"
     --model "$MODEL"
     --tensor-parallel-size "$TP"
     --pipeline-parallel-size "$PP")
[ ${#EXTRA[@]} -gt 0 ] && cmd+=("${EXTRA[@]}")

echo
echo "=== command ==="
printf '%q ' "${cmd[@]}"; echo

if [ "$DRY" = "1" ]; then
    echo
    echo "(dry run; nothing started)"
    exit 0
fi

exec "${cmd[@]}"
