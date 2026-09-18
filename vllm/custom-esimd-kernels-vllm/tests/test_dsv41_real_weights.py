"""Load a real DeepSeek V4.1 shard, not a fixture.

Everything else in this family reads checkpoints this repo generates. That
proves the code path but not the assumption underneath it: that the published
checkpoint is shaped the way the placement plan believes.

This reads a shard from the model itself. It is opt-in, because the file is
1.3 GB and is not vendored -- point DSV41_SHARD at a downloaded
model-000NN-of-00048.safetensors and it runs. The smallest shard (43 of 48)
carries head.weight and norm.weight, which is enough: the head is the largest
column-parallel tensor in the model and the norm is replicated, so one shard
exercises both modes against real data.

    curl -L -o /tmp/s.safetensors \\
      https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/model-00043-of-00048.safetensors
    DSV41_SHARD=/tmp/s.safetensors python -m pytest tests/test_dsv41_real_weights.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

_SHARD = os.environ.get("DSV41_SHARD")
pytestmark = pytest.mark.skipif(
    not _SHARD or not Path(_SHARD).is_file(),
    reason="set DSV41_SHARD to a real model-000NN-of-00048.safetensors",
)

_PY = Path(__file__).resolve().parents[1] / "python"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _PY / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


L = _load("custom_esimd_kernels_vllm.deepseek_v41_layers",
          "custom_esimd_kernels_vllm/deepseek_v41_layers.py")
W = _load("custom_esimd_kernels_vllm.deepseek_v41_loader",
          "custom_esimd_kernels_vllm/deepseek_v41_loader.py")
Sh = _load("custom_esimd_kernels_vllm.deepseek_v41_shard",
           "custom_esimd_kernels_vllm/deepseek_v41_shard.py")
R = _load("custom_esimd_kernels_vllm.deepseek_v41_reader",
          "custom_esimd_kernels_vllm/deepseek_v41_reader.py")

# config.json: vocab_size and hidden_size.
VOCAB, HIDDEN = 129280, 5120


@pytest.fixture(scope="module")
def shard():
    p = Path(_SHARD)
    header = R.read_header(p)
    names = [k for k in header if k != "__metadata__"]
    return p.parent, {n: p.name for n in names}, names


def _layout(_name):
    return (Sh.SCALE_NONE, 32)


def test_the_header_matches_the_published_config(shard):
    """The real tensors must be the shape the plan assumes.

    A head of [vocab, hidden] is the check that config.json and the checkpoint
    agree; if they did not, every placement derived from the config would be
    sized for a model that does not exist.
    """
    d, wm, _ = shard
    shapes = R.probe_shapes(d, wm)
    if "head.weight" in shapes:
        assert shapes["head.weight"].shape == (VOCAB, HIDDEN)
        assert shapes["head.weight"].dtype == "BF16"
    if "norm.weight" in shapes:
        assert shapes["norm.weight"].shape == (HIDDEN,)


def test_every_real_tensor_is_recognised(shard):
    """A tensor the plan does not know would be absent at runtime."""
    _, _, names = shard
    for n in names:
        W.classify(n)   # raises on an unknown name


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_real_weights_split_and_reconstruct_exactly(shard, tp):
    """Concatenating the ranks must reproduce the published tensor.

    A gap, an overlap or a transposed axis all fail here, and none of them
    would fail on shape alone.
    """
    from safetensors import safe_open

    d, wm, names = shard
    if "head.weight" not in wm:
        pytest.skip("this shard has no column-parallel tensor")

    stages = L.pipeline_split(40, 1)
    placements = W.place(sorted(wm), stages, tp, n_routed_experts=384)
    W.check_partition(placements, sorted(wm), tp)

    parts = []
    for rank in range(tp):
        got = R.load_rank(d, wm, placements, 0, rank, tp, _layout)
        h = got["head.weight"]
        assert h.shape == (VOCAB // tp, HIDDEN)
        assert h.dtype == torch.bfloat16
        parts.append(h)

    with safe_open(str(d / wm["head.weight"]), framework="pt") as f:
        original = f.get_tensor("head.weight")
    assert torch.equal(torch.cat(parts, dim=0), original)


def test_a_replicated_tensor_arrives_whole_on_every_rank(shard):
    """The final norm is not sharded; each rank holds all of it."""
    d, wm, _ = shard
    if "norm.weight" not in wm:
        pytest.skip("this shard has no replicated tensor")
    stages = L.pipeline_split(40, 1)
    placements = W.place(sorted(wm), stages, 4, n_routed_experts=384)
    for rank in range(4):
        got = R.load_rank(d, wm, placements, 0, rank, 4, _layout)
        assert got["norm.weight"].shape == (HIDDEN,)


def test_the_weights_are_not_all_zero(shard):
    """A reader that returned an empty buffer would pass every shape check."""
    d, wm, _ = shard
    name = "head.weight" if "head.weight" in wm else sorted(wm)[0]
    stages = L.pipeline_split(40, 1)
    placements = W.place(sorted(wm), stages, 1, n_routed_experts=384)
    got = R.load_rank(d, wm, placements, 0, 0, 1, _layout)[name]
    flat = got.flatten()[:4096].float()
    assert flat.abs().sum().item() > 0.0, "loaded a buffer of zeros"
