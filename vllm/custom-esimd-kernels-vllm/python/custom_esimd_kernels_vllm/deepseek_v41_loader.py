# SPDX-License-Identifier: Apache-2.0
"""Weight placement for a DeepSeek V4.1 checkpoint.

The published checkpoint is 96,085 tensors across 48 shards, 510 GB on disk.
Deciding which rank owns which tensor, and which slice of it, is the step
between the kernels and a running model. Every failure here is quiet:

  - a tensor assigned to no rank is silently absent, and the layer it belongs
    to computes over whatever its buffer was initialised with
  - a tensor assigned to two ranks is loaded twice and the memory budget is
    wrong by that much, which surfaces as an OOM at a random later layer
  - a row-parallel tensor sliced along the wrong axis gives every rank a full
    tensor of the wrong channels, and the all-reduce sums them into plausible
    garbage

So placement is computed, checked for exact partition, and budgeted before any
file is opened.

Names are from ``model.safetensors.index.json``. The layer sets are not
guessed: ``attn.compressor.*`` appears on layers [2, 8, 14, 20] and
``attn.indexer.wq_b.*`` on [2, 8, 14, 20, 24, 28, 32, 36], which is exactly
what the CSA2 plan derives from config.json.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

# How a tensor divides across a tensor-parallel group.
REPLICATED = "replicated"   # every rank holds the whole thing
COLUMN = "column"           # split along dim 0 (output channels)
ROW = "row"                 # split along dim 1 (input channels)
EXPERT = "expert"           # whole experts to one rank, never sliced

_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
_MTP_RE = re.compile(r"^mtp\.(\d+)\.")
_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(.+)$")

# Suffix -> shard mode for the per-layer tensors. Checked against the
# reference's module definitions, not inferred from the name.
_SHARD_MODES: dict[str, str] = {
    # Attention: heads are the parallel axis, so the q/kv projections split by
    # output and the output projection by input.
    "attn.wq_a.weight": COLUMN,
    "attn.wq_a.scale": COLUMN,
    "attn.wq_b.weight": COLUMN,
    "attn.wq_b.scale": COLUMN,
    "attn.wkv.weight": REPLICATED,   # MQA: one KV head, so nothing to split
    "attn.wkv.scale": REPLICATED,
    "attn.wo_a.weight": COLUMN,      # block diagonal over o_groups
    "attn.wo_a.scale": COLUMN,
    "attn.wo_b.weight": ROW,
    "attn.wo_b.scale": ROW,
    "attn.attn_sink": COLUMN,        # one logit per local head
    "attn.q_norm.weight": REPLICATED,
    "attn.kv_norm.weight": REPLICATED,
    "attn_norm.weight": REPLICATED,
    "ffn_norm.weight": REPLICATED,
    # Compressor and indexer, present only on their source layers.
    "attn.compressor.wkv.weight": REPLICATED,
    "attn.compressor.wgate.weight": REPLICATED,
    "attn.compressor.norm.weight": REPLICATED,
    "attn.indexer.wk.weight": REPLICATED,
    "attn.indexer.k_norm.weight": REPLICATED,
    "attn.indexer.wq_b.weight": COLUMN,   # index heads are the parallel axis
    "attn.indexer.wq_b.scale": COLUMN,
    "attn.indexer.weights_proj.weight": COLUMN,
    # Router: every rank scores every expert, then keeps the ones it owns.
    "ffn.gate.weight": REPLICATED,
    "ffn.gate.bias": REPLICATED,
    "ffn.gate.bias_vl": REPLICATED,
    # The shared expert runs on every rank.
    "ffn.shared_experts.w1.weight": COLUMN,
    "ffn.shared_experts.w1.scale": COLUMN,
    "ffn.shared_experts.w3.weight": COLUMN,
    "ffn.shared_experts.w3.scale": COLUMN,
    "ffn.shared_experts.w2.weight": ROW,
    "ffn.shared_experts.w2.scale": ROW,
    # Hyper-connection mixes are tiny and every rank needs all of them.
    "hc_attn_fn": REPLICATED,
    "hc_attn_base": REPLICATED,
    "hc_attn_scale": REPLICATED,
    "hc_ffn_fn": REPLICATED,
    "hc_ffn_base": REPLICATED,
    "hc_ffn_scale": REPLICATED,
    # Engram, present only on its own layers.
    "engram.embed.weight": COLUMN,   # the hash table shards by row
    "engram.embed.scale": COLUMN,
    "engram.wkv.weight": COLUMN,
    "engram.wkv.scale": COLUMN,
    "engram.q_weight": REPLICATED,
    "engram.k_weight": REPLICATED,
}

# Not part of the text model. DSpark MTP is deferred past v1 and the vision
# tower is a separate stack; loading either into the text ranks is memory the
# decoder then does not have.
_SKIP_PREFIXES = ("mtp.", "vision.", "aligner.")
# Vision span markers: top-level names with no prefix to match on. They belong
# to the vision stack, not the decoder.
_SKIP_EXACT = frozenset({"image_start", "image_end", "image_newline"})
_NON_LAYER = {
    "embed.weight": COLUMN,
    "head.weight": COLUMN,
    "norm.weight": REPLICATED,
}


@dataclass(frozen=True)
class Placement:
    """Where one checkpoint tensor goes."""

    name: str
    pp_stage: int
    tp_rank: int        # -1 when replicated across the whole TP group
    mode: str
    layer_id: int | None
    expert_id: int | None

    @property
    def is_replicated(self) -> bool:
        return self.tp_rank < 0


def _suffix(name: str) -> str | None:
    m = _LAYER_RE.match(name)
    return m.group(2) if m else None


def _layer_of(name: str) -> int | None:
    m = _LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def classify(name: str) -> str | None:
    """The shard mode for a tensor, or None when it is not part of the text
    model. Unrecognised layer tensors raise rather than defaulting: a new
    tensor silently treated as replicated is loaded on every rank."""
    if name.startswith(_SKIP_PREFIXES) or name in _SKIP_EXACT:
        return None
    if name in _NON_LAYER:
        return _NON_LAYER[name]
    if _EXPERT_RE.match(name):
        return EXPERT
    suf = _suffix(name)
    if suf is None:
        # A top-level tensor that is not in the table: refuse rather than guess.
        raise KeyError(f"unrecognised checkpoint tensor: {name}")
    if suf not in _SHARD_MODES:
        raise KeyError(f"unrecognised layer tensor suffix: {suf} (from {name})")
    return _SHARD_MODES[suf]


def place(
    names: Iterable[str],
    pp_stages: list[range],
    tp_size: int,
    n_routed_experts: int,
) -> list[Placement]:
    """Assign every text-model tensor to a (pp_stage, tp_rank).

    Experts go whole to one rank -- expert parallelism -- so an expert tensor
    is never sliced. Everything else is either replicated across the TP group
    or split along one axis.
    """
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if n_routed_experts % tp_size:
        raise ValueError(
            f"n_routed_experts={n_routed_experts} does not divide across "
            f"tp_size={tp_size}; some rank would own a fraction of an expert")

    stage_of: dict[int, int] = {}
    for s, r in enumerate(pp_stages):
        for lid in r:
            stage_of[lid] = s

    per_rank = n_routed_experts // tp_size
    out: list[Placement] = []
    for name in names:
        mode = classify(name)
        if mode is None:
            continue

        lid = _layer_of(name)
        if lid is None:
            # embed / head / final norm live on the first and last stages.
            stage = 0 if name == "embed.weight" else len(pp_stages) - 1
        else:
            if lid not in stage_of:
                raise ValueError(
                    f"{name}: layer {lid} is outside the pipeline split")
            stage = stage_of[lid]

        em = _EXPERT_RE.match(name)
        if em:
            eid = int(em.group(2))
            if eid >= n_routed_experts:
                raise ValueError(
                    f"{name}: expert {eid} beyond n_routed_experts="
                    f"{n_routed_experts}")
            out.append(Placement(name, stage, eid // per_rank, EXPERT, lid, eid))
            continue

        if mode == REPLICATED:
            out.append(Placement(name, stage, -1, mode, lid, None))
        else:
            for r in range(tp_size):
                out.append(Placement(name, stage, r, mode, lid, None))
    return out


def check_partition(
    placements: list[Placement],
    names: Iterable[str],
    tp_size: int,
) -> None:
    """Every text tensor is placed exactly once per rank that needs it.

    A tensor placed nowhere is absent at runtime with no error; one placed
    twice doubles its memory. Neither shows up as a wrong number until much
    later, so it is checked here.
    """
    wanted = {n for n in names if classify(n) is not None}
    seen: dict[str, int] = {}
    for p in placements:
        seen[p.name] = seen.get(p.name, 0) + 1

    missing = wanted - set(seen)
    if missing:
        raise ValueError(
            f"{len(missing)} tensor(s) placed on no rank, e.g. "
            f"{sorted(missing)[:3]}")
    extra = set(seen) - wanted
    if extra:
        raise ValueError(
            f"{len(extra)} placed tensor(s) are not in the checkpoint, e.g. "
            f"{sorted(extra)[:3]}")

    for p in placements:
        if p.mode == EXPERT and seen[p.name] != 1:
            raise ValueError(
                f"{p.name}: an expert belongs to one rank, placed "
                f"{seen[p.name]} times")
        if p.mode == REPLICATED and seen[p.name] != 1:
            raise ValueError(
                f"{p.name}: replicated tensors are recorded once, placed "
                f"{seen[p.name]} times")
        if p.mode in (COLUMN, ROW) and seen[p.name] != tp_size:
            raise ValueError(
                f"{p.name}: split across {seen[p.name]} ranks, expected "
                f"{tp_size}")


def rank_bytes(
    placements: list[Placement],
    sizes: dict[str, int],
    tp_size: int,
) -> dict[tuple[int, int], int]:
    """Bytes each (pp_stage, tp_rank) must hold.

    A replicated tensor counts against every rank in its stage, not once: that
    is the difference between a budget that fits and one that OOMs on the
    second rank.
    """
    out: dict[tuple[int, int], int] = {}
    for p in placements:
        n = sizes.get(p.name)
        if n is None:
            raise KeyError(f"no size recorded for {p.name}")
        if p.is_replicated:
            for r in range(tp_size):
                out[(p.pp_stage, r)] = out.get((p.pp_stage, r), 0) + n
        elif p.mode == EXPERT:
            key = (p.pp_stage, p.tp_rank)
            out[key] = out.get(key, 0) + n
        else:
            key = (p.pp_stage, p.tp_rank)
            out[key] = out.get(key, 0) + n // tp_size
    return out


def plan_fits(
    placements: list[Placement],
    sizes: dict[str, int],
    tp_size: int,
    bytes_per_card: int,
    reserve_bytes: int,
) -> tuple[bool, int, tuple[int, int]]:
    """Whether every rank's weights fit, with the heaviest named.

    `reserve_bytes` is the KV cache plus activations plus allocator slack. It
    is a parameter rather than a fraction because the cache size follows from
    the CSA2 plan, which the caller already has.
    """
    per = rank_bytes(placements, sizes, tp_size)
    if not per:
        return True, 0, (0, 0)
    worst_key = max(per, key=lambda k: per[k])
    worst = per[worst_key]
    return (worst + reserve_bytes) <= bytes_per_card, worst, worst_key


def minimum_cards(
    sizes: dict[str, int],
    names: Iterable[str],
    bytes_per_card: int,
    reserve_bytes: int,
    tp_size: int,
    n_routed_experts: int,
    num_hidden_layers: int,
    max_pp: int | None = None,
) -> int | None:
    """The fewest pipeline stages at this TP width whose weights fit.

    Returns None when no split up to `max_pp` fits, which is the honest answer
    for a checkpoint larger than the machine rather than a plan that will OOM.
    """
    from custom_esimd_kernels_vllm.deepseek_v41_layers import pipeline_split

    names = list(names)
    # A stage cannot be emptier than one layer, so that is the natural ceiling
    # rather than an arbitrary cap.
    ceiling = num_hidden_layers if max_pp is None else min(max_pp, num_hidden_layers)
    for pp in range(1, ceiling + 1):
        stages = pipeline_split(num_hidden_layers, pp)
        placements = place(names, stages, tp_size, n_routed_experts)
        ok, _, _ = plan_fits(
            placements, sizes, tp_size, bytes_per_card, reserve_bytes)
        if ok:
            return pp * tp_size
    return None


def load_index(index: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    """Tensor names and their byte sizes from a safetensors index.

    The index records a weight_map and a total size but not per-tensor sizes,
    so the caller supplies sizes separately when it has them; this returns an
    empty size map rather than inventing one.
    """
    wm = index.get("weight_map")
    if not isinstance(wm, dict):
        raise ValueError("index has no weight_map")
    return sorted(wm), {}
