# SPDX-License-Identifier: Apache-2.0
"""Read a rank's share of a DeepSeek V4.1 checkpoint.

The placement plan says which tensor each rank owns and which slice of it; this
reads exactly that and nothing else. The distinction matters at this size: the
checkpoint is 510 GB across 48 shards, so a rank that opens every shard to find
its own tensors pays the whole read, and one that materialises a tensor before
slicing it pays the whole tensor.

Two properties the reader has to hold:

**Only touch the shards you need.** Tensors are grouped into shards by the
index, and a rank's tensors are spread across them. Opening a shard once and
draining every tensor it holds for this rank turns 48 opens per rank into 48
total, and lets the file close before the next one opens -- which is the
difference between a bounded resident set and one that grows with the shard
count.

**Slice before materialising.** safetensors exposes a lazy slice: taking
``f.get_slice(name)`` and narrowing it reads only the bytes in range. Calling
``get_tensor`` first and narrowing after reads the whole tensor, which for a
5120x5120 fp8 weight is 26 MB a rank does not want and, across 40 layers and
8 ranks, is most of the checkpoint read eight times.

The header of a safetensors file is JSON, so the shapes and dtypes are
readable without loading anything. That is what `probe_shapes` uses, and it is
how a placement plan can be budgeted before a single weight is read.
"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# safetensors dtype -> bytes per element. Only what this checkpoint uses.
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


@dataclass(frozen=True)
class TensorInfo:
    """Shape and dtype of one tensor, read from a shard header."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str

    @property
    def nbytes(self) -> int:
        n = _DTYPE_BYTES.get(self.dtype)
        if n is None:
            raise KeyError(f"{self.name}: unknown dtype {self.dtype}")
        out = n
        for d in self.shape:
            out *= d
        return out


def read_header(path: Path) -> dict[str, Any]:
    """The JSON header of a safetensors file, without loading any tensor.

    The format is an 8-byte little-endian length followed by that many bytes
    of JSON. Reading it costs one seek, so a plan can be budgeted against the
    real shapes before anything is materialised.
    """
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: too short to be a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        blob = f.read(n)
        if len(blob) != n:
            raise ValueError(f"{path}: header claims {n} bytes, got {len(blob)}")
    return json.loads(blob)


def probe_shapes(
    shard_dir: Path,
    weight_map: dict[str, str],
    shards: Iterable[str] | None = None,
) -> dict[str, TensorInfo]:
    """Shapes and dtypes for every tensor, by reading headers only.

    `shards` limits the read to the files a rank actually needs, which is the
    point: probing all 48 to size one rank's share is 47 files of no interest.
    """
    wanted = set(shards) if shards is not None else set(weight_map.values())
    out: dict[str, TensorInfo] = {}
    for shard in sorted(wanted):
        header = read_header(shard_dir / shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if weight_map.get(name) != shard:
                # The index is authoritative about ownership; a shard holding
                # a tensor the index attributes elsewhere is a corrupt pair.
                continue
            out[name] = TensorInfo(
                name=name,
                dtype=meta["dtype"],
                shape=tuple(meta["shape"]),
                shard=shard,
            )
    return out


def shards_for(names: Iterable[str], weight_map: dict[str, str]) -> list[str]:
    """The shards holding these tensors, each once.

    A rank opens only these. Iterating the tensors and opening per tensor
    reopens a shard once per tensor it holds, which at 96,085 tensors over 48
    shards is thousands of opens for the same files.
    """
    out = set()
    for n in names:
        shard = weight_map.get(n)
        if shard is None:
            raise KeyError(f"{n} is not in the weight map")
        out.add(shard)
    return sorted(out)


def group_by_shard(
    names: Iterable[str], weight_map: dict[str, str]
) -> dict[str, list[str]]:
    """Tensor names grouped by the shard that holds them.

    This is the read order: one open per shard, every tensor drained, then
    closed before the next. It bounds the resident set to one shard.
    """
    out: dict[str, list[str]] = defaultdict(list)
    for n in names:
        shard = weight_map.get(n)
        if shard is None:
            raise KeyError(f"{n} is not in the weight map")
        out[shard].append(n)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _narrow_slice(handle: Any, sl: Any) -> Any:
    """Read only the bytes a slice covers.

    safetensors' slice object takes python slicing and reads the covered range
    rather than the whole tensor, so the axis is built as a tuple of slices
    with `slice(None)` everywhere else.
    """
    if sl is None:
        return handle[:]
    idx: list[Any] = [slice(None)] * len(handle.get_shape())
    idx[sl.axis] = slice(sl.start, sl.stop)
    return handle[tuple(idx)]


def load_rank(
    shard_dir: Path,
    weight_map: dict[str, str],
    placements: Sequence[Any],
    pp_stage: int,
    tp_rank: int,
    tp_size: int,
    scale_layout_of: Callable[[str], tuple[str, int]],
    shapes: dict[str, TensorInfo] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load exactly this rank's tensors, sliced.

    `scale_layout_of` returns (layout, block_size) for a tensor name, so the
    caller owns the quantization knowledge rather than this reader guessing it
    from a shape.

    Requires safetensors. Raises rather than falling back to a full read: a
    silent whole-tensor read here is the difference between a rank loading its
    eighth of the weights and loading all of them.
    """
    from safetensors import safe_open

    from custom_esimd_kernels_vllm.deepseek_v41_loader import EXPERT, REPLICATED
    from custom_esimd_kernels_vllm.deepseek_v41_shard import (
        scale_slice, weight_slice)

    mine = [p for p in placements
            if p.pp_stage == pp_stage
            and (p.is_replicated or p.tp_rank == tp_rank)]
    by_shard = group_by_shard((p.name for p in mine), weight_map)
    plan_of = {p.name: p for p in mine}

    if shapes is None:
        shapes = probe_shapes(shard_dir, weight_map, by_shard.keys())

    out: dict[str, Any] = {}
    for shard, names in by_shard.items():
        with safe_open(str(shard_dir / shard), framework="pt",
                       device=device) as f:
            for name in names:
                p = plan_of[name]
                info = shapes.get(name)
                if info is None:
                    raise KeyError(f"no shape probed for {name}")

                if p.mode in (REPLICATED, EXPERT):
                    sl = None
                elif name.endswith(".scale"):
                    layout, block = scale_layout_of(name)
                    base = info.shape
                    # The scale's own shape is not the weight's, so the slice
                    # is derived from the weight it belongs to.
                    wname = name[: -len(".scale")] + ".weight"
                    winfo = shapes.get(wname)
                    if winfo is None:
                        raise KeyError(
                            f"{name}: cannot slice a scale without its weight "
                            f"{wname}")
                    sl = scale_slice(winfo.shape, layout, block, p.mode,
                                     tp_rank, tp_size)
                    if sl is not None and sl.stop > base[sl.axis]:
                        raise ValueError(
                            f"{name}: slice [{sl.start}, {sl.stop}) past axis "
                            f"{sl.axis} of length {base[sl.axis]}")
                else:
                    sl = weight_slice(info.shape, p.mode, tp_rank, tp_size)

                out[name] = _narrow_slice(f.get_slice(name), sl)
    return out


def rank_read_bytes(
    placements: Sequence[Any],
    shapes: dict[str, TensorInfo],
    pp_stage: int,
    tp_rank: int,
    tp_size: int,
) -> int:
    """Bytes this rank actually reads, given the slicing.

    A sliced read is the tensor's size divided by the group; a replicated or
    expert tensor is read whole. This is the number that says whether a load
    fits in the time and bandwidth available, and it is not the same as the
    rank's resident bytes when a tensor is read once and shared.
    """
    from custom_esimd_kernels_vllm.deepseek_v41_loader import EXPERT, REPLICATED

    total = 0
    for p in placements:
        if p.pp_stage != pp_stage:
            continue
        if not (p.is_replicated or p.tp_rank == tp_rank):
            continue
        info = shapes.get(p.name)
        if info is None:
            raise KeyError(f"no shape for {p.name}")
        if p.mode in (REPLICATED, EXPERT):
            total += info.nbytes
        else:
            total += info.nbytes // tp_size
    return total
