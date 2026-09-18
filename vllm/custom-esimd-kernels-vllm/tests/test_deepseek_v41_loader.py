"""Weight placement, checked against the published checkpoint index.

Every failure in placement is quiet. A tensor placed on no rank is absent at
runtime and its layer computes over an uninitialised buffer; one placed twice
doubles a memory budget and OOMs somewhere unrelated; one split along the wrong
axis gives each rank a full tensor of the wrong channels, which the all-reduce
sums into plausible garbage. None of them raise on their own, so they are
checked here.

The tensor names are the real ones from model.safetensors.index.json
(96,085 tensors, 48 shards). They are listed here as a generator rather than
fetched, so this runs offline.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PY = Path(__file__).resolve().parents[1] / "python"


def _load(mod_name, rel):
    path = _PY / rel
    if not path.exists():
        pytest.skip(f"{rel} not present", allow_module_level=True)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


L = _load("custom_esimd_kernels_vllm.deepseek_v41_layers",
          "custom_esimd_kernels_vllm/deepseek_v41_layers.py")
W = _load("custom_esimd_kernels_vllm.deepseek_v41_loader",
          "custom_esimd_kernels_vllm/deepseek_v41_loader.py")

NUM_LAYERS = 40
N_EXPERTS = 384
KV_SOURCES = [2, 8, 14, 20]
INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36]
ENGRAM_LAYERS = [1, 14]

# Per-layer suffixes, transcribed from the index. Every layer carries these.
_EVERY_LAYER = [
    "attn.attn_sink", "attn.kv_norm.weight", "attn.q_norm.weight",
    "attn.wkv.weight", "attn.wkv.scale",
    "attn.wo_a.weight", "attn.wo_a.scale",
    "attn.wo_b.weight", "attn.wo_b.scale",
    "attn.wq_a.weight", "attn.wq_a.scale",
    "attn.wq_b.weight", "attn.wq_b.scale",
    "attn_norm.weight", "ffn_norm.weight",
    "ffn.gate.weight", "ffn.gate.bias", "ffn.gate.bias_vl",
    "ffn.shared_experts.w1.weight", "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w2.weight", "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w3.weight", "ffn.shared_experts.w3.scale",
    "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
    "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
]
# Only on kv_source_layer_ids.
_KV_SOURCE_ONLY = [
    "attn.compressor.norm.weight", "attn.compressor.wgate.weight",
    "attn.compressor.wkv.weight",
    "attn.indexer.k_norm.weight", "attn.indexer.wk.weight",
]
# Only on index_source_layer_ids -- a superset of the KV sources.
_INDEX_SOURCE_ONLY = [
    "attn.indexer.wq_b.weight", "attn.indexer.wq_b.scale",
    "attn.indexer.weights_proj.weight",
]
_ENGRAM_ONLY = [
    "engram.embed.weight", "engram.embed.scale",
    "engram.wkv.weight", "engram.wkv.scale",
    "engram.q_weight", "engram.k_weight",
]


def _checkpoint_names():
    """The text-model half of the real index, plus the parts that must skip."""
    names = ["embed.weight", "head.weight", "norm.weight",
             "image_start", "image_end", "image_newline"]
    for lid in range(NUM_LAYERS):
        for suf in _EVERY_LAYER:
            names.append(f"layers.{lid}.{suf}")
        if lid in KV_SOURCES:
            names += [f"layers.{lid}.{s}" for s in _KV_SOURCE_ONLY]
        if lid in INDEX_SOURCES:
            names += [f"layers.{lid}.{s}" for s in _INDEX_SOURCE_ONLY]
        if lid in ENGRAM_LAYERS:
            names += [f"layers.{lid}.{s}" for s in _ENGRAM_ONLY]
        for eid in range(N_EXPERTS):
            for w in ("w1", "w2", "w3"):
                names.append(f"layers.{lid}.ffn.experts.{eid}.{w}.weight")
                names.append(f"layers.{lid}.ffn.experts.{eid}.{w}.scale")
    # Deferred stacks, which must classify as skipped.
    names += ["mtp.0.attn.wkv.weight", "vision.norm.weight",
              "aligner.w1.weight"]
    return sorted(names)


@pytest.fixture(scope="module")
def names():
    return _checkpoint_names()


def test_the_compressor_lives_only_on_the_kv_source_layers(names):
    """The checkpoint confirms what config.json implies.

    attn.compressor.* appears on [2, 8, 14, 20] and nowhere else, which is the
    independent evidence that 36 layers reuse rather than compress.
    """
    got = sorted({int(n.split(".")[1]) for n in names
                  if ".attn.compressor." in n})
    assert got == KV_SOURCES


def test_the_indexer_query_lives_on_a_superset_of_the_kv_sources(names):
    """Four layers index without compressing: 24, 28, 32 and 36.

    Treating the two lists as one is the natural simplification and it drops
    the weights those four layers need.
    """
    got = sorted({int(n.split(".")[1]) for n in names
                  if ".attn.indexer.wq_b." in n})
    assert got == INDEX_SOURCES
    assert set(KV_SOURCES) < set(INDEX_SOURCES)


def test_every_tensor_classifies_without_guessing(names):
    """An unrecognised name must raise, not default to replicated.

    A new tensor silently treated as replicated is loaded on every rank, which
    is a memory error that never names its cause.
    """
    seen = 0
    for n in names:
        mode = W.classify(n)
        seen += 1
        assert mode in (None, W.REPLICATED, W.COLUMN, W.ROW, W.EXPERT)
    assert seen == len(names)

    with pytest.raises(KeyError):
        W.classify("layers.0.attn.some_new_projection.weight")
    with pytest.raises(KeyError):
        W.classify("a_top_level_tensor_nobody_declared")


@pytest.mark.parametrize("name", [
    "mtp.0.attn.wkv.weight", "vision.norm.weight", "aligner.w1.weight",
    "image_start", "image_end", "image_newline",
])
def test_the_deferred_stacks_are_skipped(name):
    """MTP and the vision tower are not the text model.

    Loading either into the decoder ranks is memory the decoder then does not
    have, and the failure is an OOM at an unrelated layer.
    """
    assert W.classify(name) is None


def test_the_shard_table_states_a_mode_for_every_known_suffix():
    """Every entry is a deliberate choice, read out of the module source.

    The table is the whole safety property: a suffix missing from it raises,
    but a suffix present with the wrong mode is silent, so the two that decide
    the collective are pinned here against the source rather than recomputed.
    """
    src = (_PY / "custom_esimd_kernels_vllm/deepseek_v41_loader.py").read_text()
    assert '"attn.wkv.weight": REPLICATED' in src, (
        "MQA has one KV head; splitting it gives a rank nothing"
    )
    assert '"attn.wo_b.weight": ROW' in src, (
        "the output projection consumes what the heads produced, so it splits "
        "by input"
    )
    assert '"attn.wq_b.weight": COLUMN' in src
    # A default would load an unknown tensor on every rank.
    assert "raise KeyError" in src, (
        "an unrecognised tensor must raise rather than default to replicated"
    )


def test_mqa_kv_projection_is_not_split(names):
    """num_key_value_heads is 1: there is no KV head to give a second rank."""
    assert W.classify("layers.0.attn.wkv.weight") == W.REPLICATED
    assert W.classify("layers.0.attn.wkv.scale") == W.REPLICATED


def test_the_output_projection_splits_by_input_not_output():
    """wo_b consumes what the heads produced, so its split follows theirs.

    Splitting it by output instead gives each rank a slice of the wrong axis,
    and the all-reduce sums full tensors of different channels.
    """
    assert W.classify("layers.0.attn.wo_b.weight") == W.ROW
    assert W.classify("layers.0.attn.wq_b.weight") == W.COLUMN


@pytest.mark.parametrize("tp,pp", [
    (1, 1), (1, 2), (2, 1), (2, 2), (4, 1), (4, 2),
    (8, 1), (8, 2), (8, 4), (16, 1), (16, 2),
])
def test_placement_partitions_exactly_at_any_shape(names, tp, pp):
    """One card or thirty-two: every tensor placed exactly as its mode requires."""
    stages = L.pipeline_split(NUM_LAYERS, pp)
    placements = W.place(names, stages, tp, N_EXPERTS)
    W.check_partition(placements, names, tp)


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16])
def test_experts_are_owned_whole_and_evenly(names, tp):
    """Expert parallelism: a rank holds whole experts, never a slice of one."""
    stages = L.pipeline_split(NUM_LAYERS, 1)
    placements = W.place(names, stages, tp, N_EXPERTS)
    per_rank = {}
    for p in placements:
        if p.mode == W.EXPERT:
            per_rank[p.tp_rank] = per_rank.get(p.tp_rank, 0) + 1
    assert len(per_rank) == tp
    assert len(set(per_rank.values())) == 1, (
        f"experts unevenly owned: {per_rank}")


def test_a_tp_size_that_splits_an_expert_is_refused(names):
    """384 experts across 5 ranks would give someone 76.8 of them."""
    stages = L.pipeline_split(NUM_LAYERS, 1)
    with pytest.raises(ValueError, match="fraction of an expert"):
        W.place(names, stages, 5, N_EXPERTS)


def test_a_missing_placement_is_caught(names):
    """The partition check is the thing that makes a dropped tensor loud."""
    stages = L.pipeline_split(NUM_LAYERS, 1)
    placements = W.place(names, stages, 1, N_EXPERTS)
    dropped = [p for p in placements
               if p.name != "layers.0.attn_norm.weight"]
    with pytest.raises(ValueError, match="placed on no rank"):
        W.check_partition(dropped, names, 1)


def test_a_duplicated_placement_is_caught(names):
    stages = L.pipeline_split(NUM_LAYERS, 1)
    placements = W.place(names, stages, 1, N_EXPERTS)
    dup = placements + [placements[0]]
    with pytest.raises(ValueError, match="split across 2 ranks|placed"):
        W.check_partition(dup, names, 1)


def test_replicated_tensors_count_against_every_rank(names):
    """A replicated tensor is held by all of them, not once.

    Counting it once is the budget error that fits on paper and OOMs on the
    second rank.
    """
    stages = L.pipeline_split(NUM_LAYERS, 1)
    tp = 4
    placements = W.place(names, stages, tp, N_EXPERTS)
    sizes = {n: 1000 for n in names}
    per = W.rank_bytes(placements, sizes, tp)
    assert len(per) == tp
    # Identical work per rank, so identical budgets.
    assert len(set(per.values())) == 1
    total_replicated = sum(
        1000 for p in placements if p.is_replicated)
    assert per[(0, 0)] >= total_replicated


def test_the_budget_refuses_a_card_count_that_does_not_fit(names):
    """An honest refusal beats a plan that OOMs mid-load."""
    stages = L.pipeline_split(NUM_LAYERS, 1)
    placements = W.place(names, stages, 1, N_EXPERTS)
    sizes = {n: 10_000_000 for n in names}
    ok, worst, _ = W.plan_fits(
        placements, sizes, 1, bytes_per_card=1024, reserve_bytes=0)
    assert ok is False
    assert worst > 1024


def test_minimum_cards_scales_with_the_tp_width(names):
    """The search is over pipeline depth at a given width, not a fixed shape."""
    sizes = {n: 1_000_000 for n in names}
    card = 8 * 1024 ** 3
    found = {}
    for tp in (1, 2, 4, 8):
        n = W.minimum_cards(sizes, names, card, 0, tp_size=tp,
                            n_routed_experts=N_EXPERTS,
                            num_hidden_layers=NUM_LAYERS)
        found[tp] = n
    assert all(v is not None for v in found.values()), found
    # Widening TP never needs more pipeline stages for the same weights.
    stages = {tp: n // tp for tp, n in found.items()}
    assert stages[8] <= stages[1]


def test_minimum_cards_returns_none_rather_than_an_impossible_plan(names):
    """A checkpoint larger than any split of the machine has no answer."""
    sizes = {n: 10 ** 12 for n in names}
    assert W.minimum_cards(sizes, names, 1024, 0, tp_size=1,
                           n_routed_experts=N_EXPERTS,
                           num_hidden_layers=NUM_LAYERS) is None


def test_single_card_is_a_valid_shape(names):
    """tp=1 pp=1 must place everything on one rank, memory aside."""
    stages = L.pipeline_split(NUM_LAYERS, 1)
    placements = W.place(names, stages, 1, N_EXPERTS)
    W.check_partition(placements, names, 1)
    assert {p.pp_stage for p in placements} == {0}
    assert {p.tp_rank for p in placements} <= {0, -1}


def test_embedding_and_head_sit_at_the_ends_of_the_pipeline(names):
    """The head is on the last stage; putting it on the first would need the
    activations to travel back."""
    stages = L.pipeline_split(NUM_LAYERS, 4)
    placements = W.place(names, stages, 1, N_EXPERTS)
    by_name = {p.name: p for p in placements}
    assert by_name["embed.weight"].pp_stage == 0
    assert by_name["head.weight"].pp_stage == 3
    assert by_name["norm.weight"].pp_stage == 3


def test_every_layer_lands_on_exactly_one_stage(names):
    stages = L.pipeline_split(NUM_LAYERS, 4)
    placements = W.place(names, stages, 2, N_EXPERTS)
    seen = {}
    for p in placements:
        if p.layer_id is None:
            continue
        prev = seen.setdefault(p.layer_id, p.pp_stage)
        assert prev == p.pp_stage, (
            f"layer {p.layer_id} split across stages {prev} and {p.pp_stage}")
    assert sorted(seen) == list(range(NUM_LAYERS))
