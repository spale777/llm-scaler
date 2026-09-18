# SPDX-License-Identifier: Apache-2.0
"""Per-rank slicing of a DeepSeek V4.1 tensor, and the CSA2 forward order.

Two things the placement plan does not do: cut a tensor, and decide what runs
in what order.

**Cutting.** A quantized weight and its scale do not slice alike. The weight is
divided along the sharded axis; the scale is divided along the *same* axis but
in units of its block, so a 5120-row weight at block 32 has a 160-row scale and
a rank taking rows [0, 2560) needs scale rows [0, 80). Slicing the scale with
the weight's own bounds silently pairs every row with the wrong scale -- the
same failure class as the Q4_K stride defect, which produced full, plausible,
wrong output.

FP8 scales are blocked on *both* axes, FP4 scales only on the input axis. The
two are not interchangeable and the shapes do not distinguish them, so the
block layout is passed in rather than inferred.

**Ordering.** CSA2 gives each layer a mode, and the modes have a dependency:
a reuse layer reads what its source published *this step*, so the source must
run first, and the candidate source must run before anything confined to its
pool. The order is the layer order -- but only because the source lists happen
to be ascending, and nothing enforces that. `forward_order` checks it rather
than assuming, because a config whose sources were not ascending would produce
a layer reading a cache written later in the same step, which is last step's
data: plausible, and wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

# Scale layouts. Which one a tensor uses is a property of its quantization,
# not of its shape, so it is named rather than guessed.
SCALE_NONE = "none"          # unquantized: no scale tensor
SCALE_BLOCK_2D = "block2d"   # fp8: [ceil(out/B), ceil(in/B)]
SCALE_ROW_GROUP = "rowgroup" # fp4: [out, in/B]


@dataclass(frozen=True)
class Slice:
    """A half-open range on one axis of one tensor."""

    axis: int
    start: int
    stop: int

    def __post_init__(self):
        if self.start < 0 or self.stop < self.start:
            raise ValueError(f"degenerate slice [{self.start}, {self.stop})")

    @property
    def length(self) -> int:
        return self.stop - self.start


def shard_axis(mode: str) -> int | None:
    """Which axis a mode divides. None when the tensor is not divided."""
    from custom_esimd_kernels_vllm.deepseek_v41_loader import (
        COLUMN, EXPERT, REPLICATED, ROW)
    if mode == COLUMN:
        return 0      # output channels
    if mode == ROW:
        return 1      # input channels, the reduction axis
    if mode in (REPLICATED, EXPERT):
        return None
    raise ValueError(f"unknown shard mode: {mode}")


def weight_slice(
    shape: Sequence[int],
    mode: str,
    tp_rank: int,
    tp_size: int,
) -> Slice | None:
    """The slice of a weight this rank owns, or None when it takes all of it."""
    axis = shard_axis(mode)
    if axis is None:
        return None
    if tp_size <= 0 or not (0 <= tp_rank < tp_size):
        raise ValueError(f"rank {tp_rank} outside tp_size {tp_size}")
    if axis >= len(shape):
        raise ValueError(
            f"mode {mode} splits axis {axis} of a {len(shape)}D tensor")
    n = shape[axis]
    if n % tp_size:
        raise ValueError(
            f"axis {axis} of length {n} does not divide across {tp_size} "
            "ranks; a ragged split leaves the ranks disagreeing on widths")
    per = n // tp_size
    return Slice(axis, tp_rank * per, (tp_rank + 1) * per)


def scale_slice(
    weight_shape: Sequence[int],
    scale_layout: str,
    block_size: int,
    mode: str,
    tp_rank: int,
    tp_size: int,
) -> Slice | None:
    """The matching slice of the scale tensor.

    The scale is cut on the same axis as the weight, but in blocks. Reusing the
    weight's bounds is the mistake: at block 32 a rank taking weight rows
    [2560, 5120) needs scale rows [80, 160), and taking [2560, 5120) of a
    160-row scale is out of range or, worse, silently clamped.
    """
    if scale_layout == SCALE_NONE:
        return None
    axis = shard_axis(mode)
    if axis is None:
        return None
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    w = weight_slice(weight_shape, mode, tp_rank, tp_size)
    assert w is not None

    # An fp4 scale is per row on the output axis and blocked only on the input
    # axis, so a column split cuts it one row per weight row.
    blocked = (scale_layout == SCALE_BLOCK_2D) or (axis == 1)
    if not blocked:
        return Slice(axis, w.start, w.stop)

    if w.start % block_size or w.length % block_size:
        raise ValueError(
            f"rank {tp_rank} takes [{w.start}, {w.stop}) of axis {axis}, which "
            f"is not a whole number of {block_size}-wide scale blocks; the "
            "rank would need a fraction of a scale")
    return Slice(axis, w.start // block_size, w.stop // block_size)


def apply_slice(tensor: Any, sl: Slice | None) -> Any:
    """Narrow a tensor to a slice. Returns it unchanged when there is none.

    Uses ``narrow`` rather than indexing so the result is a view: a 510 GB
    checkpoint cannot afford a copy per shard.
    """
    if sl is None:
        return tensor
    return tensor.narrow(sl.axis, sl.start, sl.length)


# --- CSA2 forward order -----------------------------------------------------

FULL = "full"        # compresses its own KV and indexes for itself
REINDEX = "reindex"  # reads another layer's KV, runs its own indexer
REUSE = "reuse"      # reads another layer's KV and its published indices
DENSE = "dense"      # sliding window only, no compressed positions


def layer_mode(plan: Any) -> str:
    """The CSA2 mode a layer plan describes."""
    if plan.is_dense:
        return DENSE
    if plan.is_kv_source:
        return FULL
    if plan.is_index_source:
        return REINDEX
    return REUSE


@dataclass(frozen=True)
class Step:
    """One layer's place in the forward order, and what it must find ready."""

    layer_id: int
    mode: str
    # The layer whose compressed KV this one reads; itself when FULL.
    reads_kv_from: int | None
    # The layer whose top-k indices this one reuses; itself when it indexes.
    reads_indices_from: int | None
    # True when this layer publishes state a later layer depends on.
    publishes_kv: bool
    publishes_indices: bool
    publishes_candidates: bool


def forward_order(plans: Sequence[Any]) -> list[Step]:
    """The order layers run in, with each one's dependencies made explicit.

    Layer order is the answer, but it is checked rather than assumed: every
    dependency must be satisfied by a layer that has already run, or by the
    layer itself. A source that appeared after its consumers would have the
    consumer read the previous step's cache -- real numbers, one token stale,
    and no error anywhere.
    """
    published_kv: set[int] = set()
    published_idx: set[int] = set()
    candidates_ready = False

    steps: list[Step] = []
    for p in plans:
        mode = layer_mode(p)

        if mode != DENSE:
            src = p.kv_source_layer
            if src is None:
                raise ValueError(f"layer {p.layer_id} has no KV source")
            if src != p.layer_id and src not in published_kv:
                raise ValueError(
                    f"layer {p.layer_id} reads KV from layer {src}, which has "
                    "not run yet; it would read the previous step's cache")
            isrc = p.index_source_layer
            if isrc is None:
                raise ValueError(f"layer {p.layer_id} has no index source")
            if isrc != p.layer_id and isrc not in published_idx:
                raise ValueError(
                    f"layer {p.layer_id} reuses indices from layer {isrc}, "
                    "which has not run yet")
            if p.uses_candidates and not candidates_ready:
                raise ValueError(
                    f"layer {p.layer_id} is confined to a candidate pool that "
                    "has not been built yet")

        steps.append(Step(
            layer_id=p.layer_id,
            mode=mode,
            reads_kv_from=p.kv_source_layer,
            reads_indices_from=p.index_source_layer,
            publishes_kv=p.is_kv_source,
            publishes_indices=p.is_index_source,
            publishes_candidates=p.is_candidate_source,
        ))

        if p.is_kv_source:
            published_kv.add(p.layer_id)
        if p.is_index_source:
            published_idx.add(p.layer_id)
        if p.is_candidate_source:
            candidates_ready = True

    return steps


def mode_counts(steps: Sequence[Step]) -> dict[str, int]:
    """How many layers run in each mode. The shape of the model's cost."""
    out = {FULL: 0, REINDEX: 0, REUSE: 0, DENSE: 0}
    for s in steps:
        out[s.mode] += 1
    return out


def stage_steps(steps: Sequence[Step], layers: range) -> list[Step]:
    """The steps one pipeline stage runs.

    A stage that does not contain a step's KV source still runs that step: the
    compressed cache crosses the stage boundary with the activations, which is
    why the boundary is chosen by layer count and not by source membership.
    """
    want = set(layers)
    return [s for s in steps if s.layer_id in want]


def crossing_dependencies(
    steps: Sequence[Step],
    pp_stages: Sequence[range],
) -> list[tuple[int, int, int, str]]:
    """Dependencies that cross a pipeline boundary.

    Each is (consumer_layer, producer_layer, producer_stage, kind). These are
    the tensors a stage must receive alongside the activations; a PP split that
    ignores them starves the consumer, and the consumer reads an empty cache
    rather than failing.
    """
    stage_of: dict[int, int] = {}
    for i, r in enumerate(pp_stages):
        for lid in r:
            stage_of[lid] = i

    out: list[tuple[int, int, int, str]] = []
    for s in steps:
        for src, kind in ((s.reads_kv_from, "kv"),
                          (s.reads_indices_from, "indices")):
            if src is None or src == s.layer_id:
                continue
            if stage_of[src] != stage_of[s.layer_id]:
                out.append((s.layer_id, src, stage_of[src], kind))
    return out
