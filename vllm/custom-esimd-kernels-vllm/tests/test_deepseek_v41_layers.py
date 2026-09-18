"""The CSA2 layer plan, checked against the published config.

The plan decides which layer owns a cache and which reads one. Both failures
are silent: a layer that allocates a cache it never fills attends over zeros
and returns a full tensor, and a layer pointed at the wrong source reads real
numbers from the wrong place. Neither raises.

Pure arithmetic over the config -- no weights, no GPU, no vLLM.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_MOD = (Path(__file__).resolve().parents[1]
        / "python/custom_esimd_kernels_vllm/deepseek_v41_layers.py")
if not _MOD.exists():
    pytest.skip("layer plan module not present", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("dsv41_layers", _MOD)
L = importlib.util.module_from_spec(_spec)
sys.modules["dsv41_layers"] = L
_spec.loader.exec_module(L)

# config.json, verbatim.
NUM_LAYERS = 40
COMPRESS_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
KV_SOURCES = [2, 8, 14, 20]
INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36]
CANDIDATE_SOURCE = 20


@pytest.fixture
def plans():
    return L.build_layer_plans(
        NUM_LAYERS, COMPRESS_RATIOS, KV_SOURCES, INDEX_SOURCES,
        CANDIDATE_SOURCE)


def test_compress_ratios_is_longer_than_the_layer_count():
    """The published list carries three entries past the last layer.

    Zipping it against the layers instead of indexing would silently drop the
    tail or, with a shorter list, shift every layer's mode by one.
    """
    assert len(COMPRESS_RATIOS) == 43
    assert len(COMPRESS_RATIOS) > NUM_LAYERS


def test_only_the_source_layers_allocate_a_compressed_cache(plans):
    """Four caches, not forty.

    A per-layer cache is 10x the memory and empty in 36 of them, which reads
    as an attention over zeros rather than an error.
    """
    owners = [p.layer_id for p in plans if p.needs_compressed_cache]
    assert owners == KV_SOURCES
    reusers = [p for p in plans if not p.is_dense and not p.is_kv_source]
    assert len(reusers) == 34


def test_every_compressing_layer_resolves_to_a_source_at_or_before_it(plans):
    """A forward reference would read a cache this step has not written."""
    checked = 0
    for p in plans:
        if p.is_dense:
            assert p.kv_source_layer is None
            continue
        checked += 1
        assert p.kv_source_layer is not None
        assert p.kv_source_layer <= p.layer_id
        assert p.kv_source_layer in KV_SOURCES
    assert checked == NUM_LAYERS - 2, "the two dense layers are 0 and 1"


def test_a_source_layer_reads_its_own_cache(plans):
    """It publishes then reads in the same step, which is the ordering
    Attention._compress_kv relies on."""
    for lid in KV_SOURCES:
        assert plans[lid].kv_source_layer == lid


def test_index_sources_are_independent_of_kv_sources(plans):
    """A layer can index for itself while reading another layer's KV.

    Layers 24, 28, 32 and 36 do exactly that: KV from 20, indices their own.
    Collapsing the two source lists into one is the natural simplification and
    it changes which weights every one of those layers needs.
    """
    assert plans[24].kv_source_layer == 20
    assert plans[24].index_source_layer == 24
    assert plans[36].kv_source_layer == 20
    assert plans[36].index_source_layer == 36
    extra = [lid for lid in INDEX_SOURCES if lid not in KV_SOURCES]
    assert extra == [24, 28, 32, 36]


def test_the_candidate_source_does_not_consume_its_own_pool(plans):
    """Layer 20 builds the pool; only later layers are confined to it."""
    assert plans[CANDIDATE_SOURCE].is_candidate_source
    assert plans[CANDIDATE_SOURCE].uses_candidates is False
    users = [p.layer_id for p in plans if p.uses_candidates]
    assert users, "no layer uses the candidate pool"
    assert all(u > CANDIDATE_SOURCE for u in users)


def test_dense_layers_hold_no_compressed_state(plans):
    for lid in (0, 1):
        p = plans[lid]
        assert p.is_dense
        assert not p.needs_compressed_cache
        assert not p.needs_index_cache
        assert p.index_source_layer is None


def test_a_dense_kv_source_is_refused():
    """A layer with ratio 0 has no compressed cache to publish."""
    ratios = list(COMPRESS_RATIOS)
    with pytest.raises(ValueError, match="no compressed cache"):
        L.build_layer_plans(NUM_LAYERS, ratios, [0] + KV_SOURCES,
                            INDEX_SOURCES, CANDIDATE_SOURCE)


def test_a_compressing_layer_before_any_source_is_refused():
    """It would read a cache nothing has written."""
    ratios = [2] * 43
    with pytest.raises(ValueError, match="no KV source"):
        L.build_layer_plans(NUM_LAYERS, ratios, [8], INDEX_SOURCES,
                            CANDIDATE_SOURCE)


def test_a_short_compress_ratios_list_is_refused():
    with pytest.raises(ValueError, match="compress_ratios"):
        L.build_layer_plans(NUM_LAYERS, [0] * 10, KV_SOURCES, INDEX_SOURCES,
                            CANDIDATE_SOURCE)


def test_compressed_cache_depth_divides_by_the_ratio(plans):
    """A compressed cache holds one row per group, not per token.

    Sizing it at the full sequence is 2x the memory at ratio 2 and is what
    makes the published 890 bytes/token unreachable.
    """
    specs = L.cache_specs(plans, max_seq_len=65536, head_dim=512,
                          index_head_dim=128, window_size=128)
    comp = {s.layer_id: s for s in specs if s.kind == "compressed_kv"}
    assert set(comp) == set(KV_SOURCES)
    for lid, s in comp.items():
        assert s.rows == 65536 // plans[lid].compress_ratio
        assert s.cols == 512
    # Ratio 2 halves it; ratio 1 does not.
    assert comp[2].rows == 32768
    assert comp[20].rows == 65536


def test_every_layer_keeps_its_own_sliding_window(plans):
    specs = L.cache_specs(plans, 65536, 512, 128, window_size=128)
    windows = [s for s in specs if s.kind == "window_kv"]
    assert len(windows) == NUM_LAYERS
    assert all(s.rows == 128 for s in windows)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_shard_plan_divides_every_width(tp):
    sp = L.shard_plan(tp, num_attention_heads=64, o_groups=8,
                      index_n_heads=32, n_routed_experts=384)
    assert sp.n_local_heads * tp == 64
    assert sp.n_local_groups * tp == 8
    assert sp.n_local_index_heads * tp == 32
    assert sp.experts_per_rank * tp == 384


@pytest.mark.parametrize("tp", [3, 5, 6, 7, 16])
def test_shard_plan_refuses_a_ragged_split(tp):
    """A ragged split does not fail loudly on its own: the ranks disagree on
    slice widths and the collective sums mismatched channels."""
    with pytest.raises(ValueError, match="does not divide"):
        L.shard_plan(tp, num_attention_heads=64, o_groups=8,
                     index_n_heads=32, n_routed_experts=384)


def test_pipeline_split_covers_every_layer_exactly_once():
    for pp in (1, 2, 4, 8):
        stages = L.pipeline_split(NUM_LAYERS, pp)
        assert len(stages) == pp
        seen = [lid for r in stages for lid in r]
        assert seen == list(range(NUM_LAYERS))


def test_pipeline_split_puts_the_remainder_on_the_earlier_stages():
    """The last stage also carries the LM head, so giving it the extra layers
    is the split that runs out of memory first."""
    stages = L.pipeline_split(40, 3)
    sizes = [len(r) for r in stages]
    assert sizes == [14, 13, 13]
    assert sizes[0] >= sizes[-1]


def test_pipeline_split_refuses_more_stages_than_layers():
    with pytest.raises(ValueError, match="exceeds"):
        L.pipeline_split(4, 8)


def test_sixteen_card_layout_is_expressible():
    """PP=2 x TP=8 is the floor for a 552B model on 32 GB cards."""
    stages = L.pipeline_split(NUM_LAYERS, 2)
    assert [len(r) for r in stages] == [20, 20]
    sp = L.shard_plan(8, 64, 8, 32, 384)
    assert sp.n_local_heads == 8
    assert sp.experts_per_rank == 48
