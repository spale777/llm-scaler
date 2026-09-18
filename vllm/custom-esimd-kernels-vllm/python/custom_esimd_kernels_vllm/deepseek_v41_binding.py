# SPDX-License-Identifier: Apache-2.0
"""Guarded adapter binding the ESIMD DeepSeek V4.1 kernels to a vLLM model.

Every entry point returns ``None`` when it cannot serve the call, and the
caller falls back to its own path. That is the only safe shape here: the
kernels are dimensioned for one specific architecture, and a silent mismatch
would return a full, plausible, wrong tensor rather than raising.

The guards check the architecture against the published ``config.json``
constants rather than trusting the caller, because every one of these numbers
is a silent failure if wrong:

  - ``index_n_heads`` / ``index_head_dim`` size the indexer's query rows
  - ``head_dim`` 512 with ``num_key_value_heads`` 1 means one KV row per
    position shared by all 64 query heads, not a per-head cache
  - ``n_routed_experts`` 384 with no ``n_group`` means routing is ungrouped
  - the expert weight block is 32x32, not the 128x128 the older DeepSeek
    checkpoints use

Nothing here has executed: the kernels compile and their arithmetic is checked
against the reference on CPU, but no forward pass has run on a GPU.
"""

from __future__ import annotations

import os
from typing import Any

import torch

# Published architecture constants. A model whose config disagrees is not
# served: the kernels are instantiated for these and would read past their
# operands rather than fail.
N_ROUTED_EXPERTS = 384
NUM_EXPERTS_PER_TOK = 6
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
INDEX_TOPK = 512
CANDIDATE_BLOCK_SIZE = 8
CANDIDATE_TOPK_BLOCKS = 2048
HEAD_DIM = 512
NUM_KEY_VALUE_HEADS = 1
WEIGHT_BLOCK_SIZE = 32
ROUTED_SCALING_FACTOR = 1.5

_SUPPORTED_TOP_K = (4, 6, 8)


def _disabled() -> bool:
    return os.environ.get("VLLM_XPU_DISABLE_ESIMD_DSV41", "0") == "1"


def _ops() -> Any | None:
    """The registered op namespace, or None when the extension is absent.

    Import failure is not an error: a build without the deepseek module should
    fall back, not crash the engine.
    """
    if _disabled():
        return None
    try:
        import custom_esimd_kernels_vllm.deepseek_v41  # noqa: F401
    except ImportError:
        return None
    ns = getattr(torch.ops, "custom_esimd_kernels_vllm", None)
    if ns is None:
        return None
    return ns if hasattr(ns, "deepseek_v41_noaux_tc_topk") else None


def _xpu_fp16(t: torch.Tensor | None) -> torch.Tensor | None:
    if t is None or t.device.type != "xpu":
        return None
    t = t if t.dtype == torch.float16 else t.to(torch.float16)
    return t if t.is_contiguous() else t.contiguous()


def try_noaux_tc_topk(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Route one batch of tokens. Returns (weights, indices) or None.

    The bias is required: noaux_tc selects on ``score + bias`` and weights by
    the unbiased score, so a missing bias is a different router rather than a
    default.
    """
    ops = _ops()
    if ops is None or bias is None:
        return None
    if top_k not in _SUPPORTED_TOP_K:
        return None
    if logits.dim() != 2 or logits.size(1) != N_ROUTED_EXPERTS:
        return None
    if bias.dim() != 1 or bias.size(0) != N_ROUTED_EXPERTS:
        return None

    lg = _xpu_fp16(logits)
    bs = _xpu_fp16(bias)
    if lg is None or bs is None:
        return None
    try:
        return ops.deepseek_v41_noaux_tc_topk(lg, bs, top_k)
    except (RuntimeError, torch.xpu.OutOfMemoryError):
        return None


def try_indexer_topk(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    compress_len: int,
    index_topk: int = INDEX_TOPK,
    use_candidates: bool = False,
) -> torch.Tensor | None:
    """Score compressed positions and return the selected indices, or None.

    ``weights`` must already carry ``softmax_scale * n_heads**-0.5``; the
    kernel applies no scale of its own, so pre-scaling twice is silent.

    Positions at or past ``compress_len`` are unreachable and score negative
    infinity, which is what the candidate stage reads as "not yet visible".
    The returned indices are sorted into position order with unreachable slots
    marked -1, matching the reference's contract with sparse attention.
    """
    ops = _ops()
    if ops is None:
        return None
    if q.dim() != 3 or q.size(1) != INDEX_N_HEADS or q.size(2) != INDEX_HEAD_DIM:
        return None
    if index_k.dim() != 2 or index_k.size(1) != INDEX_HEAD_DIM:
        return None
    if weights.dim() != 2 or weights.shape != (q.size(0), INDEX_N_HEADS):
        return None
    if not (0 <= compress_len <= index_k.size(0)):
        return None

    qh, kh, wh = _xpu_fp16(q), _xpu_fp16(index_k), _xpu_fp16(weights)
    if qh is None or kh is None or wh is None:
        return None

    try:
        scores = ops.deepseek_v41_lightning_indexer(qh, kh, wh, compress_len)
        if use_candidates:
            keep = ops.deepseek_v41_candidate_blocks(
                scores, CANDIDATE_BLOCK_SIZE, CANDIDATE_TOPK_BLOCKS,
                compress_len)
            # Expand the per-block mask to positions and mask the rest out, so
            # the second level scores only inside the first level's pool.
            mask = keep.repeat_interleave(CANDIDATE_BLOCK_SIZE, dim=-1)
            mask = mask[..., : scores.size(1)]
            scores = scores.masked_fill(mask == 0, float("-inf"))
    except (RuntimeError, torch.xpu.OutOfMemoryError):
        return None

    k = min(index_topk, compress_len) if compress_len else 0
    if k <= 0:
        return torch.empty((q.size(0), 0), dtype=torch.int32, device=q.device)

    # Sorted into position order, as the reference does: sparse attention walks
    # the list in order and an unsorted list is still correct but loses the
    # sequential locality of the gathered KV rows.
    idxs = scores.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < compress_len, idxs, -1).int()


def try_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor | None,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor | None:
    """Gather-by-index attention over the selected positions, or None.

    ``kv`` is [N, D]: one row per position shared by every query head, which is
    what ``num_key_value_heads: 1`` means. A per-head cache passed here would
    be read as though its heads were positions.
    """
    ops = _ops()
    if ops is None:
        return None
    if q.dim() != 3 or kv.dim() != 2:
        return None
    if kv.size(1) != q.size(2):
        return None
    if topk_idxs.dim() != 2 or topk_idxs.size(0) != q.size(0):
        return None
    if topk_idxs.dtype != torch.int32:
        return None

    qh, kvh = _xpu_fp16(q), _xpu_fp16(kv)
    if qh is None or kvh is None:
        return None
    sink = attn_sink
    if sink is not None:
        if sink.numel() != q.size(1):
            return None
        sink = sink.to(torch.float32).contiguous()
    else:
        sink = torch.empty(0, dtype=torch.float32, device=q.device)

    try:
        return ops.deepseek_v41_sparse_attn(
            qh, kvh, sink, topk_idxs.contiguous(), float(softmax_scale))
    except (RuntimeError, torch.xpu.OutOfMemoryError):
        return None


def try_fp4_expert_gemm(
    a: torch.Tensor,
    b_fp4: torch.Tensor,
    b_scales: torch.Tensor,
) -> torch.Tensor | None:
    """One expert's FP4 GEMM, or None.

    The scale group is 32 wide, matching ``weight_block_size``; a checkpoint
    quantised at 128 would be scaled four times too coarsely and every weight
    past the first block would pair with the wrong scale.
    """
    ops = _ops()
    if ops is None:
        return None
    if a.dim() != 2 or b_fp4.dim() != 2 or b_scales.dim() != 2:
        return None
    if a.size(1) != b_fp4.size(1) * 2:
        return None
    k = a.size(1)
    if k % WEIGHT_BLOCK_SIZE != 0:
        return None
    if b_scales.shape != (b_fp4.size(0), k // WEIGHT_BLOCK_SIZE):
        return None
    if b_fp4.dtype != torch.uint8 or b_scales.dtype != torch.uint8:
        return None

    ah = _xpu_fp16(a)
    if ah is None:
        return None
    try:
        return ops.deepseek_v41_fp4_gemm(
            ah, b_fp4.contiguous(), b_scales.contiguous())
    except (RuntimeError, torch.xpu.OutOfMemoryError):
        return None


def try_o_group_proj(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    o_groups: int,
) -> torch.Tensor | None:
    """Block-diagonal output projection, or None.

    ``wo_a`` is one block per group. Passing a dense weight would mix the
    groups and still return a full tensor.
    """
    ops = _ops()
    if ops is None or o_groups <= 0:
        return None
    if x.dim() != 3 or x.size(1) != o_groups:
        return None
    if wo_a.dim() != 3 or wo_a.size(0) != o_groups:
        return None
    if wo_a.size(2) != x.size(2):
        return None
    if x.device.type != "xpu":
        return None

    xf = x.to(torch.float32).contiguous()
    wf = wo_a.to(torch.float32).contiguous()
    try:
        return ops.deepseek_v41_o_group_proj(xf, wf)
    except (RuntimeError, torch.xpu.OutOfMemoryError):
        return None


def supports_config(text_config: Any) -> bool:
    """True when these kernels are dimensioned for the given model config.

    Checked rather than assumed: the kernels are instantiated for one
    architecture and would read past their operands on another.
    """
    if _ops() is None:
        return False

    def g(name, default=None):
        return getattr(text_config, name, default)

    if g("n_routed_experts") != N_ROUTED_EXPERTS:
        return False
    if g("num_experts_per_tok") not in _SUPPORTED_TOP_K:
        return False
    if g("scoring_func") != "sqrtsoftplus":
        return False
    if g("topk_method") != "noaux_tc":
        return False
    # Their absence is what makes selection ungrouped; a config that carries
    # them wants a group-limited router this binding does not request.
    if g("n_group") is not None or g("topk_group") is not None:
        return False
    if g("head_dim") != HEAD_DIM:
        return False
    if g("num_key_value_heads") != NUM_KEY_VALUE_HEADS:
        return False
    if g("index_n_heads") != INDEX_N_HEADS:
        return False
    if g("index_head_dim") != INDEX_HEAD_DIM:
        return False
    return True
