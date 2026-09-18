"""Per-rank slicing and the CSA2 forward order.

Both failures here are silent. A scale sliced with the weight's own bounds
pairs every row with the wrong scale -- the Q4_K stride defect again, which
produced full, plausible, wrong output. A layer that runs before the source it
reads from gets the previous step's cache: real numbers, one token stale, no
error anywhere.

Arithmetic only, so no GPU and no checkpoint.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PY = Path(__file__).resolve().parents[1] / "python"


def _load(name, rel):
    path = _PY / rel
    if not path.exists():
        pytest.skip(f"{rel} not present", allow_module_level=True)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


L = _load("custom_esimd_kernels_vllm.deepseek_v41_layers",
          "custom_esimd_kernels_vllm/deepseek_v41_layers.py")
W = _load("custom_esimd_kernels_vllm.deepseek_v41_loader",
          "custom_esimd_kernels_vllm/deepseek_v41_loader.py")
S = _load("custom_esimd_kernels_vllm.deepseek_v41_shard",
          "custom_esimd_kernels_vllm/deepseek_v41_shard.py")

NUM_LAYERS = 40
COMPRESS_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
KV_SOURCES = [2, 8, 14, 20]
INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36]
CANDIDATE_SOURCE = 20
BLOCK = 32


@pytest.fixture
def plans():
    return L.build_layer_plans(NUM_LAYERS, COMPRESS_RATIOS, KV_SOURCES,
                               INDEX_SOURCES, CANDIDATE_SOURCE)


# --- slicing ----------------------------------------------------------------

def test_column_splits_output_and_row_splits_input():
    """The axes are not interchangeable.

    A row-parallel tensor cut on the output axis gives every rank a full
    tensor of the wrong channels, and the all-reduce sums them.
    """
    assert S.shard_axis(W.COLUMN) == 0
    assert S.shard_axis(W.ROW) == 1
    assert S.shard_axis(W.REPLICATED) is None
    assert S.shard_axis(W.EXPERT) is None


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16])
def test_weight_slices_tile_the_axis_exactly(tp):
    """Every element belongs to exactly one rank."""
    shape = (5120, 5120)
    covered = []
    for r in range(tp):
        sl = S.weight_slice(shape, W.COLUMN, r, tp)
        covered.append((sl.start, sl.stop))
    covered.sort()
    assert covered[0][0] == 0
    assert covered[-1][1] == shape[0]
    for (_, prev_stop), (start, _) in zip(covered, covered[1:]):
        assert prev_stop == start, "a gap or an overlap between ranks"


def test_a_ragged_split_is_refused():
    """Ranks disagreeing on slice widths is not a loud failure on its own."""
    with pytest.raises(ValueError, match="does not divide"):
        S.weight_slice((5120, 5120), W.COLUMN, 0, 3)


def test_replicated_and_expert_tensors_are_not_sliced():
    assert S.weight_slice((5120, 5120), W.REPLICATED, 0, 8) is None
    assert S.weight_slice((5120, 5120), W.EXPERT, 0, 8) is None


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_fp8_scale_is_sliced_in_blocks_not_rows(tp):
    """At block 32 a rank taking weight rows [2560, 5120) needs scale rows
    [80, 160). Reusing the weight's bounds indexes past the scale or, worse,
    clamps and pairs every row with the wrong one."""
    shape = (5120, 5120)
    for r in range(tp):
        w = S.weight_slice(shape, W.COLUMN, r, tp)
        sc = S.scale_slice(shape, S.SCALE_BLOCK_2D, BLOCK, W.COLUMN, r, tp)
        assert sc.axis == w.axis
        assert sc.start == w.start // BLOCK
        assert sc.stop == w.stop // BLOCK
        assert w.length // sc.length == BLOCK


def test_fp4_scale_is_per_row_on_the_output_axis():
    """An fp4 scale is [out, in/32]: blocked on the input axis only.

    Treating it like an fp8 scale divides the output axis by 32 and hands each
    rank a thirty-second of the rows it needs.
    """
    shape = (2304, 5120)
    for r in (0, 7):
        w = S.weight_slice(shape, W.COLUMN, r, 8)
        sc = S.scale_slice(shape, S.SCALE_ROW_GROUP, BLOCK, W.COLUMN, r, 8)
        assert (sc.start, sc.stop) == (w.start, w.stop), (
            "an fp4 scale has one row per weight row")


def test_fp4_scale_is_blocked_on_a_row_split():
    """The input axis is blocked in both layouts, so a ROW split divides it."""
    shape = (2304, 5120)
    w = S.weight_slice(shape, W.ROW, 1, 4)
    sc = S.scale_slice(shape, S.SCALE_ROW_GROUP, BLOCK, W.ROW, 1, 4)
    assert sc.axis == 1
    assert sc.start == w.start // BLOCK
    assert sc.stop == w.stop // BLOCK


def test_a_split_needing_a_fraction_of_a_scale_block_is_refused():
    """160 rows across 16 ranks is 10 rows each, which is not a whole block."""
    with pytest.raises(ValueError, match="whole number of 32-wide"):
        S.scale_slice((160, 5120), S.SCALE_BLOCK_2D, BLOCK, W.COLUMN, 1, 16)


def test_unquantized_tensors_have_no_scale_slice():
    assert S.scale_slice((5120, 5120), S.SCALE_NONE, BLOCK,
                         W.COLUMN, 0, 8) is None


def test_apply_slice_narrows_without_copying():
    torch = pytest.importorskip("torch")
    t = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    sl = S.weight_slice((8, 8), W.COLUMN, 1, 2)
    out = S.apply_slice(t, sl)
    assert out.shape == (4, 8)
    assert torch.equal(out, t[4:8])
    # A view, not a copy: a 510 GB checkpoint cannot afford one per shard.
    assert out.data_ptr() == t[4:8].data_ptr()
    assert S.apply_slice(t, None) is t


# --- forward order ----------------------------------------------------------

def test_modes_partition_the_layers(plans):
    steps = S.forward_order(plans)
    counts = S.mode_counts(steps)
    assert sum(counts.values()) == NUM_LAYERS
    assert counts[S.FULL] == len(KV_SOURCES)
    assert counts[S.DENSE] == 2
    # Four layers index without compressing.
    assert counts[S.REINDEX] == len(INDEX_SOURCES) - len(KV_SOURCES)
    assert counts[S.REUSE] == 30


def test_every_dependency_is_satisfied_before_it_is_read(plans):
    """A source running after its consumer hands over last step's cache."""
    steps = S.forward_order(plans)
    assert len(steps) == NUM_LAYERS
    ran_kv, ran_idx = set(), set()
    checked = 0
    for s in steps:
        if s.mode != S.DENSE:
            checked += 1
            assert s.reads_kv_from == s.layer_id or s.reads_kv_from in ran_kv
            assert (s.reads_indices_from == s.layer_id
                    or s.reads_indices_from in ran_idx)
        if s.publishes_kv:
            ran_kv.add(s.layer_id)
        if s.publishes_indices:
            ran_idx.add(s.layer_id)
    assert checked == NUM_LAYERS - 2


def test_a_source_after_its_consumer_is_refused(plans):
    """forward_order checks the order it is handed, not the one it assumes.

    build_layer_plans already refuses a layer with no source before it, so a
    plan list is reordered directly here: the two checks are independent and
    this one has to hold on its own. A consumer running first reads the
    previous step's cache -- real numbers, one token stale, no error.
    """
    # Move layer 20 (the last KV source) behind the layers that read it.
    reordered = [p for p in plans if p.layer_id != CANDIDATE_SOURCE]
    reordered.append(plans[CANDIDATE_SOURCE])
    with pytest.raises(ValueError, match="has not run yet"):
        S.forward_order(reordered)


def test_the_kv_and_index_checks_are_independently_load_bearing(plans):
    """Layer 21 reads both KV and indices from layer 20, so reordering it
    exercises whichever check runs first and leaves the other untested.

    Layer 25 separates them: its KV comes from 20 and its indices from 24. Move
    only 24 behind it and the index check must fire on its own; move only 20
    and the KV check must.
    """
    # Indices late, KV fine: only the index check can catch this.
    idx_late = [p for p in plans if p.layer_id != 24] + [plans[24]]
    with pytest.raises(ValueError, match="reuses indices from layer 24"):
        S.forward_order(idx_late)

    # KV late, indices fine. Layer 21 would trip the index check too, so take
    # the plans only up to 24: its index source is itself.
    head = [p for p in plans if p.layer_id <= 24]
    kv_late = [p for p in head if p.layer_id != 20] + [plans[20]]
    with pytest.raises(ValueError, match="reads KV from layer 20"):
        S.forward_order(kv_late)


def test_forward_order_accepts_the_plans_in_layer_order(plans):
    """The control for the test above: unreordered plans must pass."""
    steps = S.forward_order(plans)
    assert [s.layer_id for s in steps] == list(range(NUM_LAYERS))


def test_candidates_are_built_before_anything_is_confined_to_them(plans):
    steps = S.forward_order(plans)
    builder = next(s for s in steps if s.publishes_candidates)
    assert builder.layer_id == CANDIDATE_SOURCE
    users = [p.layer_id for p in plans if p.uses_candidates]
    assert users and all(u > builder.layer_id for u in users)


def test_a_reindex_layer_reads_foreign_kv_but_its_own_indices(plans):
    """Layers 24, 28, 32 and 36 are exactly this case."""
    steps = S.forward_order(plans)
    for lid in (24, 28, 32, 36):
        s = steps[lid]
        assert s.mode == S.REINDEX
        assert s.reads_kv_from == CANDIDATE_SOURCE
        assert s.reads_indices_from == lid
        assert s.publishes_indices and not s.publishes_kv


def test_dense_layers_depend_on_nothing(plans):
    steps = S.forward_order(plans)
    for lid in (0, 1):
        s = steps[lid]
        assert s.mode == S.DENSE
        assert s.reads_kv_from is None
        assert s.reads_indices_from is None


@pytest.mark.parametrize("pp", [1, 2, 4, 8])
def test_stage_steps_partition_the_layers(plans, pp):
    steps = S.forward_order(plans)
    stages = L.pipeline_split(NUM_LAYERS, pp)
    seen = []
    for r in stages:
        seen += [s.layer_id for s in S.stage_steps(steps, r)]
    assert sorted(seen) == list(range(NUM_LAYERS))


def test_a_single_stage_has_no_crossing_dependencies(plans):
    steps = S.forward_order(plans)
    stages = L.pipeline_split(NUM_LAYERS, 1)
    assert S.crossing_dependencies(steps, stages) == []


@pytest.mark.parametrize("pp", [2, 4, 8])
def test_crossing_dependencies_are_well_formed(plans, pp):
    """A stage reading a cache published on another needs it sent alongside
    the activations. Ignoring one starves the consumer, which then reads an
    empty cache rather than failing."""
    steps = S.forward_order(plans)
    stages = L.pipeline_split(NUM_LAYERS, pp)
    stage_of = {lid: i for i, r in enumerate(stages) for lid in r}
    for consumer, producer, producer_stage, kind in S.crossing_dependencies(
            steps, stages):
        assert kind in ("kv", "indices")
        assert stage_of[producer] == producer_stage
        assert stage_of[consumer] != producer_stage
        # A producer always runs before its consumer.
        assert producer < consumer


def test_pp2_needs_nothing_to_cross_the_boundary(plans):
    """The PP=2 split lands on layer 20, which is the last KV source.

    So stage 1 opens by publishing the cache every one of its own layers
    reads, and no compressed state crosses -- only the activations. That is a
    property of where the boundary falls, not of the model, so it is pinned:
    a split that moved it would silently need a transfer nothing sends.
    """
    steps = S.forward_order(plans)
    stages = L.pipeline_split(NUM_LAYERS, 2)
    assert stages[1].start == CANDIDATE_SOURCE
    assert S.crossing_dependencies(steps, stages) == []


def test_a_boundary_that_splits_a_source_from_its_readers_is_reported(plans):
    """Cutting between layer 20 and the layers that read it must show up."""
    steps = S.forward_order(plans)
    stages = [range(0, 21), range(21, NUM_LAYERS)]
    crossings = S.crossing_dependencies(steps, stages)
    kv = {(c, p) for c, p, _, k in crossings if k == "kv"}
    assert kv, "layers 21+ read layer 20's KV across this boundary"
    assert all(p == CANDIDATE_SOURCE for _, p in kv)
