# SPDX-License-Identifier: Apache-2.0
"""CSA2 layer plan and weight layout for DeepSeek V4.1.

Two things a loader has to get right before any kernel runs, and both are
silent when wrong:

**Which layer owns what.** ``compress_ratios`` gives each layer a mode, but a
non-zero ratio does not mean the layer compresses its own KV -- only the layers
in ``kv_source_layer_ids`` do. Every other layer *reads* the cache its source
published. A loader that allocates a compressed cache per layer wastes 36
caches' worth of memory and, worse, leaves them empty: the layer would attend
over zeros and still return a full tensor.

**Where each tensor lives.** A source layer publishes a compressed KV cache and
an index-key cache; the layers between sources hold neither. Asking a reuse
layer for its own cache is how a plausible-but-empty attention happens.

This module computes that plan from the config and nothing else, so it is
testable without weights, without a GPU, and without vLLM.

Sourced from deepseek-ai/DeepSeek-V4.1-Flash ``config.json`` and
``inference/model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A layer's compress_ratio of 0 means it runs dense sliding-window attention
# only: no compressed positions, so no indexer and no compressed cache.
DENSE = 0


@dataclass(frozen=True)
class LayerPlan:
    """What one decoder layer does and which caches it touches."""

    layer_id: int
    compress_ratio: int
    # Owns and writes the compressed KV cache the reuse layers read.
    is_kv_source: bool
    # Runs its own indexer; the layers between sources reuse the published
    # top-k indices instead of scoring again.
    is_index_source: bool
    # Builds the candidate pool that every later indexing layer is confined to.
    is_candidate_source: bool
    # Restricted to the pool built by candidate_source_layer_id.
    uses_candidates: bool
    # The layer whose compressed cache this one reads. Itself when it is a
    # source; None when it is dense.
    kv_source_layer: int | None
    # The layer whose top-k indices this one reuses. Itself when it is an
    # index source; None when it is dense.
    index_source_layer: int | None

    @property
    def is_dense(self) -> bool:
        return self.compress_ratio == DENSE

    @property
    def needs_compressed_cache(self) -> bool:
        """Only a source allocates one. A reuse layer reading its own would
        attend over an empty cache and return a full tensor."""
        return self.is_kv_source

    @property
    def needs_index_cache(self) -> bool:
        return self.is_kv_source and not self.is_dense


def _last_at_or_before(sources: list[int], layer_id: int) -> int | None:
    """The most recent source at or before this layer.

    At-or-before, not strictly before: a source layer reads the cache it has
    just written this step, which is what makes the publish-then-read ordering
    in Attention._compress_kv work.
    """
    best = None
    for s in sources:
        if s <= layer_id:
            best = s if best is None else max(best, s)
    return best


def build_layer_plans(
    num_hidden_layers: int,
    compress_ratios: list[int],
    kv_source_layer_ids: list[int],
    index_source_layer_ids: list[int],
    candidate_source_layer_id: int,
) -> list[LayerPlan]:
    """The per-layer plan CSA2 implies.

    `compress_ratios` in the published config is longer than the layer count --
    it carries three trailing entries past the last layer -- so it is indexed,
    never zipped.
    """
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if len(compress_ratios) < num_hidden_layers:
        raise ValueError(
            f"compress_ratios has {len(compress_ratios)} entries for "
            f"{num_hidden_layers} layers")

    kv_sources = sorted(kv_source_layer_ids)
    index_sources = sorted(index_source_layer_ids)

    plans = []
    for lid in range(num_hidden_layers):
        ratio = compress_ratios[lid]
        dense = ratio == DENSE
        is_kv = lid in kv_sources
        is_idx = lid in index_sources

        if is_kv and dense:
            raise ValueError(
                f"layer {lid} is a KV source with compress_ratio 0; a dense "
                "layer has no compressed cache to publish")

        kv_src = None if dense else _last_at_or_before(kv_sources, lid)
        idx_src = None if dense else _last_at_or_before(index_sources, lid)

        if not dense and kv_src is None:
            raise ValueError(
                f"layer {lid} compresses at ratio {ratio} but no KV source "
                "precedes it, so there is no cache for it to read")

        plans.append(LayerPlan(
            layer_id=lid,
            compress_ratio=ratio,
            is_kv_source=is_kv,
            is_index_source=is_idx,
            is_candidate_source=(lid == candidate_source_layer_id),
            uses_candidates=(0 <= candidate_source_layer_id < lid
                             and not dense),
            kv_source_layer=kv_src,
            index_source_layer=idx_src,
        ))
    return plans


def plans_from_config(text_config: Any) -> list[LayerPlan]:
    """Build the plan from a HF text config."""
    return build_layer_plans(
        num_hidden_layers=getattr(text_config, "num_hidden_layers"),
        compress_ratios=list(getattr(text_config, "compress_ratios")),
        kv_source_layer_ids=list(getattr(text_config, "kv_source_layer_ids")),
        index_source_layer_ids=list(
            getattr(text_config, "index_source_layer_ids")),
        candidate_source_layer_id=getattr(
            text_config, "candidate_source_layer_id", -1),
    )


@dataclass(frozen=True)
class CacheSpec:
    """One cache a source layer allocates, in elements not bytes."""

    layer_id: int
    kind: str          # "compressed_kv" | "index_k" | "window_kv"
    rows: int
    cols: int

    @property
    def elements(self) -> int:
        return self.rows * self.cols


def cache_specs(
    plans: list[LayerPlan],
    max_seq_len: int,
    head_dim: int,
    index_head_dim: int,
    window_size: int,
) -> list[CacheSpec]:
    """Every cache that must be allocated, and by which layer.

    A compressed cache holds one row per group, so its depth is the sequence
    divided by the layer's ratio -- not the sequence. Sizing it at full length
    is the mistake that makes the 890 bytes/token figure unreachable.
    """
    specs = []
    for p in plans:
        # Every layer keeps its own sliding window of raw KV.
        specs.append(CacheSpec(p.layer_id, "window_kv", window_size, head_dim))
        if p.needs_compressed_cache:
            rows = max_seq_len // p.compress_ratio
            specs.append(CacheSpec(p.layer_id, "compressed_kv", rows, head_dim))
        if p.needs_index_cache:
            rows = max_seq_len // p.compress_ratio
            specs.append(CacheSpec(p.layer_id, "index_k", rows, index_head_dim))
    return specs


@dataclass(frozen=True)
class ShardPlan:
    """How one layer's tensors divide across a tensor-parallel group."""

    tp_size: int
    n_local_heads: int
    n_local_groups: int
    n_local_index_heads: int
    # Experts are not sharded by TP here: the MoE is expert-parallel, so each
    # rank holds whole experts rather than slices of every expert.
    experts_per_rank: int


def shard_plan(
    tp_size: int,
    num_attention_heads: int,
    o_groups: int,
    index_n_heads: int,
    n_routed_experts: int,
    expert_parallel: bool = True,
) -> ShardPlan:
    """Split the per-layer widths across a TP group.

    Each of these must divide evenly. A ragged split does not fail loudly: the
    ranks disagree on their slice widths and the all-reduce sums mismatched
    shapes, or worse, matching shapes holding different channels.
    """
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    for name, value in (("num_attention_heads", num_attention_heads),
                        ("o_groups", o_groups),
                        ("index_n_heads", index_n_heads)):
        if value % tp_size:
            raise ValueError(
                f"{name}={value} does not divide across tp_size={tp_size}")

    if expert_parallel:
        if n_routed_experts % tp_size:
            raise ValueError(
                f"n_routed_experts={n_routed_experts} does not divide across "
                f"tp_size={tp_size}")
        experts = n_routed_experts // tp_size
    else:
        experts = n_routed_experts

    return ShardPlan(
        tp_size=tp_size,
        n_local_heads=num_attention_heads // tp_size,
        n_local_groups=o_groups // tp_size,
        n_local_index_heads=index_n_heads // tp_size,
        experts_per_rank=experts,
    )


def pipeline_split(num_hidden_layers: int, pp_size: int) -> list[range]:
    """Which layers each pipeline stage owns.

    The remainder goes to the earlier stages rather than the last: the final
    stage also carries the LM head, so giving it the extra layers is the split
    that runs out of memory first.
    """
    if pp_size <= 0:
        raise ValueError("pp_size must be positive")
    if pp_size > num_hidden_layers:
        raise ValueError(
            f"pp_size={pp_size} exceeds {num_hidden_layers} layers")
    base, extra = divmod(num_hidden_layers, pp_size)
    out = []
    start = 0
    for stage in range(pp_size):
        n = base + (1 if stage < extra else 0)
        out.append(range(start, start + n))
        start += n
    return out
