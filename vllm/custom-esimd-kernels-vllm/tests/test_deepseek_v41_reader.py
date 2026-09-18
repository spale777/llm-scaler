"""Load a rank's share of a real safetensors checkpoint.

Unlike the rest of the DeepSeek tests, this one executes: it writes real
safetensors files, reads them back through the placement plan, and checks the
loaded values. The checkpoint is small, but the format, the slicing and the
per-shard read order are the real ones.

The values are chosen so a wrong slice is visible rather than merely
differently shaped: each weight is an arange, so element [0, 0] of a rank's
slice says exactly which rows it got. A scale sliced with the weight's own
bounds has the right dtype and a plausible shape and the wrong numbers, which
is the failure this is here to catch.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

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
R = _load("custom_esimd_kernels_vllm.deepseek_v41_reader",
          "custom_esimd_kernels_vllm/deepseek_v41_reader.py")

OUT, IN, BLOCK = 512, 256, 32


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Two shards carrying real V4.1 tensor names."""
    d = tmp_path_factory.mktemp("dsv41_ckpt")
    shard_a = {
        "layers.0.attn.wq_b.weight":
            torch.arange(OUT * IN, dtype=torch.float32).reshape(OUT, IN),
        "layers.0.attn.wq_b.scale":
            torch.arange((OUT // BLOCK) * (IN // BLOCK),
                         dtype=torch.float32).reshape(OUT // BLOCK,
                                                      IN // BLOCK),
        "layers.0.attn.wkv.weight": torch.ones(OUT, IN),
    }
    shard_b = {
        "layers.0.attn.wo_b.weight":
            torch.arange(OUT * IN, dtype=torch.float32).reshape(OUT, IN),
        "layers.0.ffn.experts.0.w1.weight": torch.full((64, IN), 7.0),
        "layers.0.ffn.experts.1.w1.weight": torch.full((64, IN), 9.0),
    }
    safetensors_torch.save_file(shard_a, d / "a.safetensors")
    safetensors_torch.save_file(shard_b, d / "b.safetensors")
    weight_map = {**{k: "a.safetensors" for k in shard_a},
                  **{k: "b.safetensors" for k in shard_b}}
    return d, weight_map


def _layout(_name):
    return (S.SCALE_BLOCK_2D, BLOCK)


def _placements(weight_map, tp_size):
    stages = L.pipeline_split(1, 1)
    return W.place(sorted(weight_map), stages, tp_size, n_routed_experts=2)


def test_header_gives_shapes_without_loading_a_tensor(checkpoint):
    """The header is JSON after an 8-byte length, so a plan can be budgeted
    against real shapes before a weight is read."""
    d, wm = checkpoint
    shapes = R.probe_shapes(d, wm)
    assert shapes["layers.0.attn.wq_b.weight"].shape == (OUT, IN)
    assert shapes["layers.0.attn.wq_b.scale"].shape == (OUT // BLOCK,
                                                        IN // BLOCK)
    assert shapes["layers.0.attn.wq_b.weight"].dtype == "F32"
    assert shapes["layers.0.attn.wq_b.weight"].nbytes == OUT * IN * 4


def test_only_the_needed_shards_are_named(checkpoint):
    """A rank opening all 48 shards to find its own tensors pays the whole
    read. One expert lives in one shard."""
    _, wm = checkpoint
    assert R.shards_for(["layers.0.ffn.experts.0.w1.weight"], wm) == [
        "b.safetensors"]
    assert len(R.shards_for(sorted(wm), wm)) == 2


def test_tensors_are_grouped_one_open_per_shard(checkpoint):
    """Opening per tensor reopens a shard once per tensor it holds."""
    _, wm = checkpoint
    grouped = R.group_by_shard(sorted(wm), wm)
    assert set(grouped) == {"a.safetensors", "b.safetensors"}
    assert sum(len(v) for v in grouped.values()) == len(wm)
    for shard, names in grouped.items():
        assert all(wm[n] == shard for n in names)


@pytest.mark.parametrize("rank", [0, 1])
def test_column_split_loads_the_rows_that_rank_owns(checkpoint, rank):
    """The weight is an arange, so element [0, 0] names the first row read."""
    d, wm = checkpoint
    got = R.load_rank(d, wm, _placements(wm, 2), 0, rank, 2, _layout)
    w = got["layers.0.attn.wq_b.weight"]
    assert w.shape == (OUT // 2, IN)
    assert w[0, 0].item() == rank * (OUT // 2) * IN


@pytest.mark.parametrize("rank", [0, 1])
def test_the_scale_is_sliced_in_blocks_not_with_the_weights_bounds(
        checkpoint, rank):
    """This is the failure the whole slicer exists for.

    Reusing the weight's bounds would ask for rows [256, 512) of a 16-row
    scale. The shape alone does not reveal it, so the values are checked.
    """
    d, wm = checkpoint
    got = R.load_rank(d, wm, _placements(wm, 2), 0, rank, 2, _layout)
    sc = got["layers.0.attn.wq_b.scale"]
    rows = OUT // BLOCK // 2
    assert sc.shape == (rows, IN // BLOCK)
    assert sc[0, 0].item() == rank * rows * (IN // BLOCK)


@pytest.mark.parametrize("rank", [0, 1])
def test_row_split_cuts_the_input_axis(checkpoint, rank):
    d, wm = checkpoint
    got = R.load_rank(d, wm, _placements(wm, 2), 0, rank, 2, _layout)
    w = got["layers.0.attn.wo_b.weight"]
    assert w.shape == (OUT, IN // 2)
    assert w[0, 0].item() == rank * (IN // 2)


@pytest.mark.parametrize("rank", [0, 1])
def test_the_mqa_kv_projection_arrives_whole(checkpoint, rank):
    """num_key_value_heads is 1: there is no second head to hand over."""
    d, wm = checkpoint
    got = R.load_rank(d, wm, _placements(wm, 2), 0, rank, 2, _layout)
    assert got["layers.0.attn.wkv.weight"].shape == (OUT, IN)


def test_each_rank_gets_whole_and_different_experts(checkpoint):
    """Expert parallelism: never a slice of an expert, and no overlap."""
    d, wm = checkpoint
    pl = _placements(wm, 2)
    r0 = R.load_rank(d, wm, pl, 0, 0, 2, _layout)
    r1 = R.load_rank(d, wm, pl, 0, 1, 2, _layout)
    e0 = {k for k in r0 if ".experts." in k}
    e1 = {k for k in r1 if ".experts." in k}
    assert e0 and e1
    assert e0.isdisjoint(e1)
    assert r0["layers.0.ffn.experts.0.w1.weight"].shape == (64, IN)
    assert r0["layers.0.ffn.experts.0.w1.weight"][0, 0].item() == 7.0
    assert r1["layers.0.ffn.experts.1.w1.weight"][0, 0].item() == 9.0


def test_the_two_ranks_reconstruct_the_original_weight(checkpoint):
    """Concatenating the column slices must give back exactly what was saved.

    A gap, an overlap or a transposed axis all fail here, and none of them
    would fail on shape alone.
    """
    d, wm = checkpoint
    pl = _placements(wm, 2)
    parts = [R.load_rank(d, wm, pl, 0, r, 2, _layout)[
        "layers.0.attn.wq_b.weight"] for r in (0, 1)]
    rebuilt = torch.cat(parts, dim=0)
    original = torch.arange(OUT * IN, dtype=torch.float32).reshape(OUT, IN)
    assert torch.equal(rebuilt, original)


def test_tp1_loads_everything_unsliced(checkpoint):
    """One card is a real shape, not a degenerate case."""
    d, wm = checkpoint
    got = R.load_rank(d, wm, _placements(wm, 1), 0, 0, 1, _layout)
    assert set(got) == set(wm)
    assert got["layers.0.attn.wq_b.weight"].shape == (OUT, IN)
    assert got["layers.0.attn.wo_b.weight"].shape == (OUT, IN)


def test_read_volume_falls_with_the_group_width(checkpoint):
    """A sliced read is the tensor divided by the group; replicated is whole.

    If this did not fall, every rank would be reading the whole checkpoint.
    """
    d, wm = checkpoint
    shapes = R.probe_shapes(d, wm)
    one = R.rank_read_bytes(_placements(wm, 1), shapes, 0, 0, 1)
    two = R.rank_read_bytes(_placements(wm, 2), shapes, 0, 0, 2)
    assert two < one
    # wkv is replicated and the experts are whole, so it is not a clean half.
    assert two > one // 4


def test_a_missing_tensor_is_named(checkpoint):
    _, wm = checkpoint
    with pytest.raises(KeyError, match="not in the weight map"):
        R.shards_for(["layers.99.nonexistent.weight"], wm)


def test_a_truncated_header_is_refused(tmp_path):
    """A short read must not be mistaken for an empty checkpoint."""
    p = tmp_path / "broken.safetensors"
    p.write_bytes(b"\x04\x00\x00")
    with pytest.raises(ValueError, match="too short"):
        R.read_header(p)


def test_a_header_longer_than_the_file_is_refused(tmp_path):
    import struct
    p = tmp_path / "lying.safetensors"
    p.write_bytes(struct.pack("<Q", 4096) + b"{}")
    with pytest.raises(ValueError, match="header claims"):
        R.read_header(p)
