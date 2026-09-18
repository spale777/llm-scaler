"""The DeepSeek V4.1 kernel binding must refuse what it cannot serve.

Every kernel here is instantiated for one architecture. Handed another, it
would read past its operands and return a full, plausible, wrong tensor rather
than raise -- so the binding's guards are the safety property, not a
convenience. These tests drive the guards with mismatched shapes and configs
and assert a refusal.

No XPU is needed: every path checked here refuses before it reaches an op.
"""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Loaded by path, not as a package member: importing the package runs its
# __init__, which needs the compiled extensions. The binding itself has no such
# dependency -- that is the point of it -- so it must be testable without them.
_BINDING = (Path(__file__).resolve().parents[1]
            / "python/custom_esimd_kernels_vllm/deepseek_v41_binding.py")
if not _BINDING.exists():
    pytest.skip("binding module not present", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("dsv41_binding", _BINDING)
binding = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(binding)


class _Cfg:
    """A config carrying exactly the published V4.1 values."""

    def __init__(self, **over):
        self.n_routed_experts = 384
        self.num_experts_per_tok = 6
        self.scoring_func = "sqrtsoftplus"
        self.topk_method = "noaux_tc"
        self.n_group = None
        self.topk_group = None
        self.head_dim = 512
        self.num_key_value_heads = 1
        self.index_n_heads = 32
        self.index_head_dim = 128
        for k, v in over.items():
            setattr(self, k, v)


def test_constants_match_the_published_config():
    """These numbers size the kernels; a drift is silent at runtime."""
    assert binding.N_ROUTED_EXPERTS == 384
    assert binding.NUM_EXPERTS_PER_TOK == 6
    assert binding.INDEX_N_HEADS == 32
    assert binding.INDEX_HEAD_DIM == 128
    assert binding.INDEX_TOPK == 512
    assert binding.CANDIDATE_BLOCK_SIZE == 8
    assert binding.CANDIDATE_TOPK_BLOCKS == 2048
    assert binding.HEAD_DIM == 512
    # MQA: one KV row per position shared by every query head.
    assert binding.NUM_KEY_VALUE_HEADS == 1
    # 32x32, not the 128x128 of the older DeepSeek checkpoints.
    assert binding.WEIGHT_BLOCK_SIZE == 32
    assert binding.ROUTED_SCALING_FACTOR == 1.5


@pytest.mark.parametrize("field,value", [
    ("n_routed_experts", 256),
    ("scoring_func", "softmax"),
    ("topk_method", "greedy"),
    ("head_dim", 128),
    ("num_key_value_heads", 8),
    ("index_n_heads", 64),
    ("index_head_dim", 64),
    ("num_experts_per_tok", 2),
])
def test_supports_config_rejects_a_mismatched_architecture(field, value):
    assert binding.supports_config(_Cfg(**{field: value})) is False


@pytest.mark.parametrize("field", ["n_group", "topk_group"])
def test_supports_config_rejects_a_grouped_router(field):
    """A config carrying grouping wants a router this binding does not request.

    V4.1's config has neither field, and that absence is what makes selection
    range over every expert.
    """
    assert binding.supports_config(_Cfg(**{field: 8})) is False


def test_noaux_tc_requires_a_bias():
    """The bias is what the selection is made on, not an optional offset."""
    logits = torch.zeros(2, 384, dtype=torch.float16)
    assert binding.try_noaux_tc_topk(logits, None, 6) is None


@pytest.mark.parametrize("top_k", [1, 2, 3, 5, 7, 16])
def test_noaux_tc_refuses_an_uninstantiated_top_k(top_k):
    """Only 4, 6 and 8 have kernel arms; the rest would throw from device code."""
    logits = torch.zeros(2, 384, dtype=torch.float16)
    bias = torch.zeros(384, dtype=torch.float16)
    assert binding.try_noaux_tc_topk(logits, bias, top_k) is None


def test_noaux_tc_refuses_a_wrong_expert_count():
    logits = torch.zeros(2, 256, dtype=torch.float16)
    bias = torch.zeros(256, dtype=torch.float16)
    assert binding.try_noaux_tc_topk(logits, bias, 6) is None


@pytest.mark.parametrize("shape", [
    (4, 16, 128),   # wrong head count
    (4, 32, 64),    # wrong head dim
    (4, 32),        # missing a dimension
])
def test_indexer_refuses_a_wrong_query_geometry(shape):
    q = torch.zeros(*shape, dtype=torch.float16)
    k = torch.zeros(64, 128, dtype=torch.float16)
    w = torch.zeros(shape[0], 32, dtype=torch.float16)
    assert binding.try_indexer_topk(q, k, w, 64) is None


def test_indexer_refuses_a_compress_len_past_the_cache():
    q = torch.zeros(4, 32, 128, dtype=torch.float16)
    k = torch.zeros(64, 128, dtype=torch.float16)
    w = torch.zeros(4, 32, dtype=torch.float16)
    assert binding.try_indexer_topk(q, k, w, 65) is None


def test_sparse_attn_refuses_a_per_head_kv_cache():
    """kv is [N, D]: one row per position, shared across heads.

    A [N, H, D] cache passed here would have its heads read as positions.
    """
    q = torch.zeros(4, 64, 512, dtype=torch.float16)
    kv = torch.zeros(128, 64, 512, dtype=torch.float16)
    idx = torch.zeros(4, 512, dtype=torch.int32)
    assert binding.try_sparse_attn(q, kv, None, idx, 0.04) is None


def test_sparse_attn_refuses_non_int32_indices():
    q = torch.zeros(4, 64, 512, dtype=torch.float16)
    kv = torch.zeros(128, 512, dtype=torch.float16)
    idx = torch.zeros(4, 512, dtype=torch.int64)
    assert binding.try_sparse_attn(q, kv, None, idx, 0.04) is None


def test_sparse_attn_refuses_a_sink_of_the_wrong_width():
    q = torch.zeros(4, 64, 512, dtype=torch.float16)
    kv = torch.zeros(128, 512, dtype=torch.float16)
    idx = torch.zeros(4, 512, dtype=torch.int32)
    sink = torch.zeros(32, dtype=torch.float32)
    assert binding.try_sparse_attn(q, kv, sink, idx, 0.04) is None


def test_fp4_gemm_refuses_a_128_wide_scale_group():
    """weight_block_size is 32. A 128-quantised checkpoint scaled as 32 pairs
    every weight past the first block with the wrong scale."""
    k = 256
    a = torch.zeros(2, k, dtype=torch.float16)
    b = torch.zeros(16, k // 2, dtype=torch.uint8)
    coarse = torch.zeros(16, k // 128, dtype=torch.uint8)
    assert binding.try_fp4_expert_gemm(a, b, coarse) is None


def test_fp4_gemm_refuses_a_k_that_is_not_packed_in_half():
    a = torch.zeros(2, 256, dtype=torch.float16)
    b = torch.zeros(16, 256, dtype=torch.uint8)   # not K/2
    s = torch.zeros(16, 8, dtype=torch.uint8)
    assert binding.try_fp4_expert_gemm(a, b, s) is None


def test_o_group_proj_refuses_a_dense_weight():
    """wo_a is one block per group; a dense weight mixes them."""
    x = torch.zeros(4, 8, 128, dtype=torch.float32)
    dense = torch.zeros(1024, 1024, dtype=torch.float32)
    assert binding.try_o_group_proj(x, dense, 8) is None


def test_o_group_proj_refuses_a_group_count_mismatch():
    x = torch.zeros(4, 8, 128, dtype=torch.float32)
    w = torch.zeros(4, 64, 128, dtype=torch.float32)
    assert binding.try_o_group_proj(x, w, 8) is None


def test_every_entry_point_returns_none_without_the_extension(monkeypatch):
    """A build without the module must fall back, not crash the engine."""
    monkeypatch.setattr(binding, "_ops", lambda: None)
    logits = torch.zeros(2, 384, dtype=torch.float16)
    bias = torch.zeros(384, dtype=torch.float16)
    q = torch.zeros(4, 32, 128, dtype=torch.float16)
    k = torch.zeros(64, 128, dtype=torch.float16)
    w = torch.zeros(4, 32, dtype=torch.float16)

    assert binding.try_noaux_tc_topk(logits, bias, 6) is None
    assert binding.try_indexer_topk(q, k, w, 64) is None
    assert binding.try_sparse_attn(
        torch.zeros(4, 64, 512, dtype=torch.float16),
        torch.zeros(128, 512, dtype=torch.float16),
        None, torch.zeros(4, 512, dtype=torch.int32), 0.04) is None
    assert binding.try_fp4_expert_gemm(
        torch.zeros(2, 256, dtype=torch.float16),
        torch.zeros(16, 128, dtype=torch.uint8),
        torch.zeros(16, 8, dtype=torch.uint8)) is None
    assert binding.try_o_group_proj(
        torch.zeros(4, 8, 128, dtype=torch.float32),
        torch.zeros(8, 64, 128, dtype=torch.float32), 8) is None
    assert binding.supports_config(_Cfg()) is False


def test_disable_env_var_turns_the_binding_off(monkeypatch):
    monkeypatch.setenv("VLLM_XPU_DISABLE_ESIMD_DSV41", "1")
    assert binding._ops() is None
    assert binding.supports_config(_Cfg()) is False
