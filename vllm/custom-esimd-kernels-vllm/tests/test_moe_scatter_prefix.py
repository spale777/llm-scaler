"""Contract test for MoE_Scatter_Prefix_Kernel.

The kernel builds an exclusive prefix sum over experts_token_count[num_experts]
in 32-wide steps, so the load must be lane-masked: any num_experts that is not a
multiple of 32 (60 for DeepSeek-V2-Lite, 4/6/8 for small MoEs) otherwise folds
out-of-array garbage into running_sum and every later expert offset scatters its
tokens to the wrong rows. The index arithmetic is modelled in Python, so no GPU
is needed.
"""

from pathlib import Path

import pytest

from srctext import code, tokens

_HEADERS = [
    Path(__file__).resolve().parents[1] / "csrc/xpu/esimd_kernels/moe_ops.h",
    Path(__file__).resolve().parents[3]
    / "sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels/moe_ops.h",
]


def prefix_masked(counts, num_experts):
    """Model of the kernel: 32-wide steps, inactive lanes zeroed."""
    expert_start = [0] * (num_experts + 1)
    running = 0
    max_count = 0
    for base in range(0, num_experts, 32):
        valid = min(num_experts - base, 32)
        lanes = [counts[base + i] if i < valid else 0 for i in range(32)]
        for i in range(32):
            if i >= valid:
                break
            c = lanes[i]
            max_count = max(max_count, c)
            expert_start[base + i] = running
            running += c
    expert_start[num_experts] = running
    return expert_start, max_count


@pytest.mark.parametrize("num_experts", [1, 4, 6, 8, 31, 32, 33, 60, 64, 96, 128, 256])
def test_prefix_sum_matches_reference(num_experts):
    counts = [(i * 7) % 13 + 1 for i in range(num_experts)]
    got, got_max = prefix_masked(counts, num_experts)

    # Reference: plain exclusive scan.
    want = [0] * (num_experts + 1)
    acc = 0
    for i, c in enumerate(counts):
        want[i] = acc
        acc += c
    want[num_experts] = acc

    assert got == want, f"num_experts={num_experts}: prefix sum diverges"
    assert got_max == max(counts)
    assert got[num_experts] == sum(counts), "total token count must be exact"


# Only non-multiples of 32 can over-read; 32/64/96 are exactly covered.
@pytest.mark.parametrize("num_experts", [4, 6, 8, 31, 60, 100])
def test_unmasked_load_would_corrupt(num_experts):
    """An unmasked 32-lane read corrupts the scan, so the masking above
    cannot be 'simplified' back to a plain block_load."""
    counts = [1] * num_experts
    garbage = [999] * 32  # whatever follows the array in memory
    running = 0
    for base in range(0, num_experts, 32):
        lanes = [
            counts[base + i] if base + i < num_experts else garbage[i]
            for i in range(32)
        ]
        running += sum(lanes)
    assert running != sum(counts), (
        "expected the unmasked model to over-count; test is not exercising the bug"
    )


@pytest.mark.parametrize("path", _HEADERS, ids=lambda p: p.parents[3].name)
def test_header_masks_the_tail_load(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    start = src.index("struct MoE_Scatter_Prefix_Kernel")
    end = src.index("struct", start + 10)
    body = src[start:end]
    t = tokens(body)
    assert "block_load<int32_t,32>(" not in t
    # Polarity: `lane >= valid` selects exactly the out-of-range lanes.
    assert "simd_mask<32>m=lane<(uint32_t)valid" in t, (
        "the prefix load must enable lanes below the valid count"
    )
    assert "simd_mask<32>m=lane>=(uint32_t)valid" not in t, (
        "inverted mask: this loads the lanes past num_experts"
    )


@pytest.mark.parametrize("path", _HEADERS, ids=lambda p: p.parents[3].name)
def test_prefix_commits_each_chunk_as_one_scatter(path):
    """The scan is serial; its stores are not.

    A dword store per expert is one message each, and the kernel runs as a
    single work-item between two full-width kernels on every MoE layer, so the
    whole grid waits on it. num_experts reaches 512 here. Accumulating the
    offsets in a register and committing 32 lanes at a time keeps the scan
    order identical while cutting the message count by 32x.
    """
    if not path.exists():
        pytest.skip(f"{path} not present")
    src = path.read_text()
    start = src.index("struct MoE_Scatter_Prefix_Kernel")
    end = src.index("struct", start + 10)
    t = tokens(src[start:end])
    assert "block_store<uint32_t,1>(expert_start+base+i," not in t, (
        "per-expert dword store is back; the chunk must commit as one scatter"
    )
    assert "scatter<uint32_t,32>(expert_start+base," in t, (
        "the chunk's offsets must be committed with a masked 32-lane scatter"
    )
