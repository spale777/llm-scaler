"""The DeepSeek V4.1 FP4 dequant and noaux_tc router arithmetic.

Both are pure arithmetic, so they are checked exactly on CPU against the
reference in deepseek-ai/DeepSeek-V4.1-Flash
inference/{model.py,kernel.py,convert.py}.
"""

import math
import re
import struct
from pathlib import Path

import pytest

from srctext import code, tokens

_DS = Path(__file__).resolve().parents[1] / "csrc/deepseek_v41"
_DEQUANT = Path(__file__).resolve().parents[1] / "csrc/deepseek_v41/fp4_dequant.h"
_LUT = _DS / "fp4_dequant.h"
_TOPK = _DS / "topk_noaux_tc.h"
_KERNELS = Path(__file__).resolve().parents[1] / "csrc/xpu/deepseek_kernels.sycl"

# E2M1: 1 sign, 2 exponent, 1 mantissa. Reference clamps FP4 to +-6.0.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _f16_from_bits(b: int) -> float:
    return struct.unpack("e", struct.pack("H", b & 0xFFFF))[0]


def test_lut_matches_e2m1():
    src = _LUT.read_text()
    m = re.search(r"fp4_e2m1_lut\[16\]\s*=\s*\{(.*?)\};", src, re.S)
    assert m, "LUT not found"
    nums = [float(x) for x in re.findall(r"(-?\d+\.?\d*)f", m.group(1))]
    assert len(nums) == 16, f"expected 16 entries, got {len(nums)}"
    assert nums[:8] == E2M1, f"positive half wrong: {nums[:8]}"
    assert [abs(v) for v in nums[8:]] == E2M1, f"negative half wrong: {nums[8:]}"


def test_lut_is_not_eighth_scaled():
    """E2M1 / 8 would make every GEMM output 8x too small."""
    src = _LUT.read_text()
    assert "0.0625f" not in src, "LUT carries 1/8-scaled magnitudes"


def test_bit_trick_reproduces_the_lut():
    """The bit trick THE KERNEL SHIPS must reproduce the E2M1 table.

    The constants are read out of fp4_dequant.h: recomputing them in Python and
    comparing to a Python constant is a tautology that never opens the source.
    """
    src = code(_DEQUANT.read_text())
    m = re.search(r"simd<uint16_t, 8> res = (0x[0-9A-Fa-f]+) \+ \(\(m - (\d+)\) << (\d+)\)",
                  src)
    assert m, "the E2M1 bit trick is no longer recognisable in fp4_dequant.h"
    base, sub, shift = int(m.group(1), 16), int(m.group(2)), int(m.group(3))

    got = []
    for mm in range(8):
        if mm == 0:
            bits = 0x0000
        elif mm == 1:
            bits = 0x3800
        else:
            bits = base + ((mm - sub) << shift)
        got.append(_f16_from_bits(bits))
    assert got == E2M1, (
        f"the kernel's bit trick (0x{base:04X} + ((m - {sub}) << {shift})) "
        f"yields {got}, not the E2M1 table {E2M1}"
    )


def test_unpack_uses_the_e2m1_base():
    """0x2C00 encodes 0.0625 and yields the 1/8-scaled magnitudes."""
    # Case-insensitive both ways: hex literal case is not semantic.
    src = code(_LUT.read_text())
    assert re.search(r"0x3c00", src, re.I), "expected the 0x3C00 base for m >= 2"
    assert not re.search(r"0x2c00", src, re.I), (
        "0x2C00 encodes 0.0625, not 1.0: every magnitude comes out 8x small"
    )


def _sqrtsoftplus(x):
    import math

    return math.sqrt(math.log1p(math.exp(x)))


def route_reference(logits, bias, top_k, route_scale=1.5, eps=1e-20):
    """DeepSeek Gate.forward: transform ALL, select on +bias, gather unbiased."""
    scores = [_sqrtsoftplus(x) for x in logits]
    sel = [s + b for s, b in zip(scores, bias)]
    idx = sorted(range(len(sel)), key=lambda i: -sel[i])[:top_k]
    w = [scores[i] for i in idx]
    tot = sum(w) + eps
    return idx, [x / tot * route_scale for x in w]


def route_raw_selection(logits, bias, top_k, route_scale=1.5, eps=1e-5):
    """Selecting on the RAW logit + bias, transforming only the winner."""
    lg, bs = list(logits), list(bias)
    idx, w = [], []
    for _ in range(top_k):
        best = max(range(len(lg)), key=lambda i: lg[i] + bs[i])
        idx.append(best)
        w.append(_sqrtsoftplus(lg[best]))
        lg[best] = -65500.0
        bs[best] = -65500.0
    tot = sum(w) + eps
    return idx, [x / tot * route_scale for x in w]


def test_selection_diverges_when_bias_is_nonzero():
    """With bias != 0, raw-logit selection picks a different expert set."""
    import random

    random.seed(7)
    diff = 0
    trials = 400
    for _ in range(trials):
        logits = [random.gauss(0, 1.5) for _ in range(32)]
        bias = [random.gauss(0, 0.3) for _ in range(32)]
        a, _ = route_reference(logits, bias, 6)
        b, _ = route_raw_selection(logits, bias, 6)
        if set(a) != set(b):
            diff += 1
    assert diff > trials * 0.1, (
        f"only {diff}/{trials} differed; the test is not discriminating"
    )


def test_orders_agree_when_bias_is_zero():
    """sqrtsoftplus is monotonic, so with no bias both orders must match."""
    import random

    random.seed(11)
    for _ in range(50):
        logits = [random.gauss(0, 1.5) for _ in range(32)]
        zero = [0.0] * 32
        a, _ = route_reference(logits, zero, 6)
        b, _ = route_raw_selection(logits, zero, 6)
        assert set(a) == set(b)


def test_kernel_transforms_before_selecting():
    src = _TOPK.read_text()
    body = src[src.index("compute_noaux_tc_routing") :]
    tr = body.index("sqrt_softplus")
    # The property, not one spelling of the cast.
    m = re.search(r"scores\[i\]\s*\+\s*[^;]*bias\[i\]", body)
    assert m, "selection must be on scores + bias"
    assert tr < m.start(), (
        "sqrtsoftplus must be applied to all experts before selection"
    )
    assert "1e-20f" in body, "reference normalises by sum + 1e-20, not 1e-5"
    assert "float scores[NUM_EXPERTS]" in body, "accumulate in float, not fp16"


def test_expert_count_matches_the_config():
    if not _KERNELS.exists():
        pytest.skip("deepseek kernels not present")
    src = _KERNELS.read_text()
    assert "DeepSeekTopKKernel<384," in src, (
        "n_routed_experts is 384 in config.json; 256 matches no layer"
    )
    assert "case 6:" in src, "num_experts_per_tok is 6"


# --- the implemented kernels' arithmetic, mirrored and checked exactly ---

_GEMM = _DS / "fp4_gemm.h"
NEG_INF = -3.0e38
EPS = 1e-20
ROUTED_SCALING = 1.5


def _nib_val(n):
    return E2M1[n & 7] * (-1.0 if n & 8 else 1.0)


def _scale_val(raw):
    """UE8M0 -> float, mirroring decode_ue8m0_scales."""
    return 0.0 if raw <= 112 else 2.0 ** (min(raw, 142) - 112 - 15)


def test_ue8m0_decode_is_exact_over_every_byte():
    """raw=127 must be 1.0, and the Inf encoding must be unreachable.

    Exponent field 31 is fp16 Inf/NaN; an Inf scale against an E2M1 zero gives
    NaN and poisons the whole output tile, so the raw value saturates at 142.
    """
    src = code(_DEQUANT.read_text())
    m = re.search(r"min\(shifted,\s*\(uint16_t\)(\d+)\)", src)
    assert m, "the UE8M0 saturation bound is no longer recognisable"
    assert int(m.group(1)) == 142, (
        f"scale saturates at {m.group(1)}; 143 or above encodes fp16 Inf"
    )
    m = re.search(r"shifted\s*=\s*shifted\s*-\s*(\d+)", src)
    assert m and int(m.group(1)) == 112, "the 127->15 exponent rebias is wrong"
    assert _scale_val(127) == 1.0
    assert _scale_val(112) == 0.0
    assert _scale_val(142) == 32768.0


def _kernel_gemm(A, B, S, M, N, K, n_tile=16, group=32):
    """Mirror of FP4_GEMM_Kernel including its VNNI index arithmetic."""
    k_packed, k_groups = K // 2, K // group
    C = [[0.0] * N for _ in range(M)]
    for n0 in range(0, N, n_tile):
        for m0 in range(M):
            acc = [0.0] * n_tile
            for k in range(0, K, 32):
                kg = k // group
                scl = [_scale_val(S[(n0 + n) * k_groups + kg])
                       if n0 + n < N else 0.0 for n in range(n_tile)]
                b_rows = [0.0] * (n_tile * 32)
                for n in range(n_tile):
                    if n0 + n >= N:
                        continue
                    base = (n0 + n) * k_packed + k // 2
                    for byte in range(16):
                        p = B[base + byte]
                        b_rows[n * 32 + byte * 2] = _nib_val(p & 0xF) * scl[n]
                        b_rows[n * 32 + byte * 2 + 1] = _nib_val((p >> 4) & 0xF) * scl[n]
                for ks in range(0, 32, 16):
                    b_vnni = [0.0] * (n_tile * 16)
                    for n in range(n_tile):
                        for kk in range(16):
                            b_vnni[(kk // 2) * (n_tile * 2) + n * 2 + (kk & 1)] = \
                                b_rows[n * 32 + ks + kk]
                    a_tile = [A[m0 * K + k + ks + t] for t in range(16)]
                    for n in range(n_tile):
                        acc[n] += sum(
                            a_tile[kk] * b_vnni[(kk // 2) * (n_tile * 2) + n * 2 + (kk & 1)]
                            for kk in range(16))
            for n in range(n_tile):
                if n0 + n < N:
                    C[m0][n0 + n] = acc[n]
    return C


def _reference_gemm(A, B, S, M, N, K, group=32):
    k_packed, k_groups = K // 2, K // group
    C = [[0.0] * N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            s = 0.0
            for k in range(K):
                byte = B[n * k_packed + k // 2]
                nib = (byte & 0xF) if k % 2 == 0 else ((byte >> 4) & 0xF)
                s += A[m * K + k] * _nib_val(nib) * _scale_val(S[n * k_groups + k // group])
            C[m][n] = s
    return C


@pytest.mark.parametrize("M,N,K", [(1, 16, 32), (1, 16, 64), (2, 32, 64),
                                   (3, 16, 128), (1, 64, 256)])
def test_fp4_gemm_vnni_layout_matches_a_plain_dot_product(M, N, K):
    """The VNNI interleave is where a hand-derived DPAS layout usually breaks.

    b_vnni[(kk/2)*(N*2) + n*2 + (kk&1)] places k pairs for channel n; getting
    the stride or the parity wrong still produces a full output tile, just the
    wrong one, so this compares against a direct sum over k.
    """
    import random
    random.seed(1234 + M * 1000 + N * 10 + K)
    A = [random.uniform(-2, 2) for _ in range(M * K)]
    B = [random.randrange(256) for _ in range(N * (K // 2))]
    S = [random.randrange(113, 143) for _ in range(N * (K // 32))]

    got = _kernel_gemm(A, B, S, M, N, K)
    want = _reference_gemm(A, B, S, M, N, K)
    worst = max(abs(got[m][n] - want[m][n]) / max(1e-9, abs(want[m][n]))
                for m in range(M) for n in range(N))
    assert worst < 1e-9, f"VNNI layout diverges: worst relative error {worst:.3e}"


def _kernel_router(logits, bias, E, TK, NG, TKG):
    """Mirror of compute_noaux_tc_routing, including the group-limited stage."""
    scores = [math.sqrt(math.log(1.0 + math.exp(x))) for x in logits]
    sel = [scores[i] + bias[i] for i in range(E)]
    if NG > 1 and TKG < NG:
        GS = E // NG
        key = []
        for g in range(NG):
            b0 = b1 = NEG_INF
            for i in range(GS):
                v = sel[g * GS + i]
                if v > b0:
                    b1, b0 = b0, v
                elif v > b1:
                    b1 = v
            key.append(b0 + b1)
        live = [False] * NG
        for _ in range(TKG):
            best, bg = NEG_INF, 0
            for g in range(NG):
                if not live[g] and key[g] > best:
                    best, bg = key[g], g
            live[bg] = True
            key[bg] = NEG_INF
        for g in range(NG):
            if not live[g]:
                for i in range(GS):
                    sel[g * GS + i] = NEG_INF
    idx, w, tot = [], [], 0.0
    for _ in range(TK):
        mv, mi = NEG_INF, 0
        for i in range(E):
            if sel[i] > mv:
                mv, mi = sel[i], i
        idx.append(mi)
        w.append(scores[mi])
        tot += scores[mi]
        sel[mi] = NEG_INF
    inv = 1.0 / (tot + EPS)
    return idx, [x * inv * ROUTED_SCALING for x in w]


def test_group_limited_routing_matches_the_reference():
    """Only the best TOPK_GROUP groups may serve a token.

    A group ranks by the sum of its two best keys, so one strong expert cannot
    carry a group alone. Skipping this stage still returns TOP_K experts, just
    from anywhere, which no shape check would catch.
    """
    import random
    random.seed(11)
    E, TK, NG, TKG = 384, 6, 8, 4
    for _ in range(50):
        lg = [random.uniform(-6, 6) for _ in range(E)]
        bs = [random.uniform(-1, 1) for _ in range(E)]
        got_i, got_w = _kernel_router(lg, bs, E, TK, NG, TKG)

        scores = [math.sqrt(math.log1p(math.exp(x))) for x in lg]
        keys = [scores[i] + bs[i] for i in range(E)]
        GS = E // NG
        gscore = [sum(sorted(keys[g * GS:(g + 1) * GS], reverse=True)[:2])
                  for g in range(NG)]
        keep = set(sorted(range(NG), key=lambda g: -gscore[g])[:TKG])
        masked = [keys[i] if (i // GS) in keep else NEG_INF for i in range(E)]
        want_i = sorted(range(E), key=lambda i: (-masked[i], i))[:TK]

        assert got_i == want_i, "group-limited selection diverges"
        assert all((i // GS) in keep for i in got_i), (
            "an expert outside the surviving groups was selected"
        )
        tot = sum(scores[i] for i in want_i)
        want_w = [scores[i] / (tot + EPS) * ROUTED_SCALING for i in want_i]
        assert max(abs(a - b) for a, b in zip(got_w, want_w)) < 1e-12


def test_routing_weight_is_the_unbiased_score():
    """The bias steers selection only; it must not reach the returned weight.

    Folding the bias into the weight is the natural mistake and changes every
    expert's contribution while still selecting the right experts.
    """
    import random
    random.seed(5)
    E, TK = 384, 6
    lg = [random.uniform(-4, 4) for _ in range(E)]
    bs = [random.uniform(0.5, 2.0) for _ in range(E)]
    idx, w = _kernel_router(lg, bs, E, TK, 8, 4)
    scores = [math.sqrt(math.log1p(math.exp(x))) for x in lg]
    tot = sum(scores[i] for i in idx)
    for k, i in enumerate(idx):
        assert abs(w[k] - scores[i] / (tot + EPS) * ROUTED_SCALING) < 1e-12, (
            "the returned weight carries the selection bias"
        )


# --- the shipped kernels against the published config ----------------------

import dsv41_config as cfg  # noqa: E402

_KERNELS_SYCL = Path(__file__).resolve().parents[1] / "csrc/xpu/deepseek_kernels.sycl"


def test_router_expert_count_matches_config():
    t = tokens(_KERNELS_SYCL.read_text())
    assert f"DeepSeekTopKKernel<{cfg.N_ROUTED_EXPERTS}," in t, (
        f"the router must be instantiated for n_routed_experts="
        f"{cfg.N_ROUTED_EXPERTS}"
    )


def test_router_is_not_group_limited():
    """config.json carries no n_group and no topk_group.

    Their absence is load-bearing: noaux_tc here selects over every expert.
    A group count carried over from the V3-style configs masks experts this
    model never masks, and the router still returns num_experts_per_tok of
    them, so the output is plausible and wrong.
    """
    assert cfg.N_GROUP is None and cfg.TOPK_GROUP is None
    t = tokens(_KERNELS_SYCL.read_text())
    m = re.search(r"DeepSeekTopKKernel<(\d+),TK,(\d+),(\d+)>", t)
    assert m, "the router instantiation is no longer recognisable"
    n_group, topk_group = int(m.group(2)), int(m.group(3))
    assert n_group == topk_group, (
        f"the router keeps {topk_group} of {n_group} expert groups, but this "
        "config has no grouping; every expert must stay eligible"
    )


def test_topk_arms_cover_the_configured_experts_per_tok():
    """num_experts_per_tok is 6; an uninstantiated arm throws from the host."""
    src = _KERNELS_SYCL.read_text()
    arms = {int(x) for x in re.findall(r"case (\d+): submit_kernel", src)}
    assert cfg.NUM_EXPERTS_PER_TOK in arms, (
        f"top_k={cfg.NUM_EXPERTS_PER_TOK} has no instantiated arm; the model's "
        f"own setting would be refused. Arms present: {sorted(arms)}"
    )


def test_routing_constants_match_config():
    src = code(_TOPK.read_text())
    m = re.search(r"ROUTED_SCALING_FACTOR\s*=\s*([0-9.]+)f", src)
    assert m, "the routed scaling factor is no longer a named constant"
    assert float(m.group(1)) == cfg.ROUTED_SCALING_FACTOR, (
        f"routed_scaling_factor is {m.group(1)}, config says "
        f"{cfg.ROUTED_SCALING_FACTOR}"
    )
    assert cfg.NORM_TOPK_PROB, "config sets norm_topk_prob"
    assert "weight_sum" in src, (
        "norm_topk_prob is true, so the weights must be normalised by their sum"
    )


def test_expert_weight_block_size_is_32():
    """quantization_config.weight_block_size is [32, 32], not [128, 128].

    The MoE block-scale kernels were written for 128x128. A 32-wide block
    scaled as though it were 128 pairs every weight past the first block with
    the wrong scale.
    """
    assert cfg.WEIGHT_BLOCK_SIZE == [32, 32]
    host = code((Path(__file__).resolve().parents[1]
                 / "csrc/xpu/esimd_kernel_moe.sycl").read_text())
    assert "block_k == 32" in host, (
        "the MoE host refuses a 32-wide K block, which is what this model uses"
    )
    assert "block_n == 32" in host, (
        "the MoE host refuses a 32-wide N block, which is what this model uses"
    )


def test_expert_dtype_is_fp4_and_scales_are_ue8m0():
    assert cfg.EXPERT_DTYPE == "fp4"
    assert cfg.SCALE_FMT == "ue8m0"
    c = code(_GEMM.read_text())
    assert "unpack_fp4_row" in c, "the expert GEMM must unpack E2M1"
    d = code(_DEQUANT.read_text())
    assert "decode_ue8m0_scales" in d, "UE8M0 scale decode missing"


# --- lightning indexer ------------------------------------------------------

_INDEXER = _DS / "lightning_indexer.h"
NEG_INF_F = float("-inf")


def _kernel_index_scores(q, k, w, S, T, H, D, reach):
    """Mirror of LightningIndexerKernel: relu per head, then weighted sum."""
    out = [[0.0] * T for _ in range(S)]
    for s in range(S):
        for t in range(T):
            if t >= reach[s]:
                out[s][t] = NEG_INF_F
                continue
            acc = 0.0
            for h in range(H):
                dot = sum(q[s][h][d] * k[t][d] for d in range(D))
                acc += (dot if dot > 0 else 0.0) * w[s][h]
            out[s][t] = acc
    return out


def test_indexer_matches_the_reference_score():
    """einsum(bshd,btd->bsht), relu, weight per head, sum over heads.

    model.py::Indexer computes `(index_score.relu_() * weights).sum(dim=2)`.
    One shared key per position: this is MQA, so every index head reads the
    same key row.
    """
    import random
    random.seed(2)
    S, T, H, D = 3, 40, 4, 8
    q = [[[random.uniform(-1, 1) for _ in range(D)] for _ in range(H)]
         for _ in range(S)]
    k = [[random.uniform(-1, 1) for _ in range(D)] for _ in range(T)]
    w = [[random.uniform(0, 1) for _ in range(H)] for _ in range(S)]
    reach = [10, 25, 40]

    got = _kernel_index_scores(q, k, w, S, T, H, D, reach)
    checked = 0
    seen = 0
    for s in range(S):
        for t in range(T):
            if t >= reach[s]:
                seen += 1
                assert got[s][t] == NEG_INF_F, (
                    "an unreachable position must be -inf, not small: the "
                    "candidate stage reads a block max and treats -inf as "
                    "unreachable"
                )
                continue
            per_head = [sum(q[s][h][d] * k[t][d] for d in range(D))
                        for h in range(H)]
            per_head = [x if x > 0 else 0.0 for x in per_head]
            want = sum(per_head[h] * w[s][h] for h in range(H))
            checked += 1
            assert abs(got[s][t] - want) < 1e-12
    assert checked > 0 and seen > 0, (
        f"examined {checked} reachable and {seen} masked positions; the shapes "
        "no longer exercise both paths"
    )


def test_indexer_rectifies_before_weighting():
    """Summing first and rectifying after changes the ranking.

    A head that scores a position negatively must contribute zero, not a
    negative another head has to overcome.
    """
    import random
    random.seed(2)
    S, T, H, D = 3, 40, 4, 8
    q = [[[random.uniform(-1, 1) for _ in range(D)] for _ in range(H)]
         for _ in range(S)]
    k = [[random.uniform(-1, 1) for _ in range(D)] for _ in range(T)]
    w = [[random.uniform(0, 1) for _ in range(H)] for _ in range(S)]
    reach = [10, 25, 40]
    got = _kernel_index_scores(q, k, w, S, T, H, D, reach)

    differs = 0
    total = 0
    for s in range(S):
        for t in range(min(reach[s], T)):
            total += 1
            ph = [sum(q[s][h][d] * k[t][d] for d in range(D)) for h in range(H)]
            after = sum(ph[h] * w[s][h] for h in range(H))
            after = after if after > 0 else 0.0
            if abs(after - got[s][t]) > 1e-9:
                differs += 1
    assert differs > total // 2, (
        "rectifying after the sum is indistinguishable here, so this test is "
        "not exercising the ordering"
    )
    src = code(_INDEXER.read_text())
    assert "dot > 0.0f ? dot : 0.0f" in src, (
        "the per-head rectification is gone; the ranking changes"
    )


def test_indexer_applies_no_scale_of_its_own():
    """weights already carry softmax_scale * n_heads^-0.5 from the host."""
    src = code(_INDEXER.read_text())
    i = src.find("struct LightningIndexerKernel")
    body = src[i:src.find("launch_lightning_indexer", i)]
    assert "softmax_scale" not in body and "rsqrt" not in body, (
        "the kernel rescales scores the host already scaled"
    )


def _kernel_keep(scores, S, T, bs, topk, reach):
    """Mirror of CandidateBlockKernel."""
    nb = (T + bs - 1) // bs
    keep = [[0] * nb for _ in range(S)]
    for s in range(S):
        last = (reach[s] - 1) // bs
        for _ in range(min(topk, nb)):
            best, bb = NEG_INF_F, -1
            for b in range(nb):
                if keep[s][b]:
                    continue
                sc = (float("inf") if b == last
                      else max([scores[s][i]
                                for i in range(b * bs, min(b * bs + bs, T))]
                               + [NEG_INF_F]))
                if sc > best:
                    best, bb = sc, b
            if bb < 0 or best == NEG_INF_F:
                break
            keep[s][bb] = 1
    return keep


@pytest.mark.parametrize("topk", [1, 2, 3, 5])
def test_candidate_blocks_match_the_reference(topk):
    """Block score is its best position; the newest block is pinned in.

    select_candidate_blocks pads with -inf, takes an amax per block, forces the
    block holding the query's newest position to +inf, then keeps only picks
    that scored above -inf -- so a context with fewer reachable blocks than
    topk_blocks does not admit positions the query cannot see.
    """
    import random
    random.seed(2)
    S, T, H, D, bs = 3, 40, 4, 8, 8
    q = [[[random.uniform(-1, 1) for _ in range(D)] for _ in range(H)]
         for _ in range(S)]
    k = [[random.uniform(-1, 1) for _ in range(D)] for _ in range(T)]
    w = [[random.uniform(0, 1) for _ in range(H)] for _ in range(S)]
    reach = [10, 25, 40]
    scores = _kernel_index_scores(q, k, w, S, T, H, D, reach)

    got = _kernel_keep(scores, S, T, bs, topk, reach)

    nb = (T + bs - 1) // bs
    want = []
    for s in range(S):
        padded = scores[s] + [NEG_INF_F] * ((-T) % bs)
        bl = [max(padded[b * bs:(b + 1) * bs]) for b in range(nb)]
        last = (reach[s] - 1) // bs
        bl = [float("inf") if b == last else bl[b] for b in range(nb)]
        order = sorted(range(nb), key=lambda b: -bl[b])[:min(topk, nb)]
        keep = [0] * nb
        for b in order:
            if bl[b] > NEG_INF_F:
                keep[b] = 1
        want.append(keep)

    assert got == want, f"candidate selection diverges at topk={topk}"
    for s in range(S):
        assert got[s][(reach[s] - 1) // bs] == 1, (
            "the block holding the newest position must be kept"
        )


# --- sparse attention -------------------------------------------------------

_SPARSE = _DS / "sparse_attn.h"
SCORE_FLOOR = -1e30


def _kernel_sparse(q, kv, sink, idxs, S, H, N, TOPK, D, scale):
    """Mirror of SparseAttnKernel."""
    out = [[[0.0] * D for _ in range(H)] for _ in range(S)]
    for s in range(S):
        for h in range(H):
            qv = q[s][h]
            acc = [0.0] * D
            m = SCORE_FLOOR
            se = 0.0
            for c in range(TOPK):
                j = idxs[s][c]
                if j < 0 or j >= N:
                    continue
                kvv = kv[j]
                score = sum(qv[d] * kvv[d] for d in range(D)) * scale
                mN = max(m, score)
                corr = math.exp(m - mN)
                p = math.exp(score - mN)
                se = se * corr + p
                acc = [acc[d] * corr + kvv[d] * p for d in range(D)]
                m = mN
            if sink is not None:
                se += math.exp(sink[h] - m) if sink[h] - m > -700 else 0.0
            out[s][h] = [a / se for a in acc] if se > 0 else [0.0] * D
    return out


def test_sparse_attn_matches_the_blocked_reference():
    """kernel.py processes the index list in blocks of 64; this walks it one
    position at a time. The online-softmax recurrence must agree either way."""
    import random
    random.seed(9)
    S, H, N, TOPK, D = 3, 4, 50, 80, 8
    scale = (1.0 / D) ** 0.5
    q = [[[random.uniform(-1, 1) for _ in range(D)] for _ in range(H)]
         for _ in range(S)]
    kv = [[random.uniform(-1, 1) for _ in range(D)] for _ in range(N)]
    sink = [random.uniform(-2, 2) for _ in range(H)]
    idxs = []
    for _ in range(S):
        row = ([random.randrange(N) for _ in range(TOPK // 2)]
               + [-1] * (TOPK - TOPK // 2))
        random.shuffle(row)
        idxs.append(row)

    got = _kernel_sparse(q, kv, sink, idxs, S, H, N, TOPK, D, scale)

    BLOCK = 64
    for s in range(S):
        for h in range(H):
            qv = q[s][h]
            acc = [0.0] * D
            m = SCORE_FLOOR
            se = 0.0
            for t in range(0, TOPK, BLOCK):
                blk = idxs[s][t:t + BLOCK]
                rows = [(kv[j] if 0 <= j < N else [0.0] * D) for j in blk]
                sc = [(sum(qv[d] * rows[i][d] for d in range(D)) * scale
                       if 0 <= blk[i] < N else -math.inf)
                      for i in range(len(blk))]
                mprev = m
                live = [x for x in sc if x != -math.inf]
                m = max([m] + live)
                ss = math.exp(mprev - m)
                ex = [(math.exp(x - m) if x != -math.inf else 0.0) for x in sc]
                se = se * ss + sum(ex)
                acc = [acc[d] * ss for d in range(D)]
                for i in range(len(blk)):
                    for d in range(D):
                        acc[d] += ex[i] * rows[i][d]
            se += math.exp(sink[h] - m)
            want = [a / se for a in acc] if se > 0 else [0.0] * D
            for d in range(D):
                assert abs(got[s][h][d] - want[d]) < 1e-12


def test_sparse_attn_all_invalid_row_is_zero_not_nan():
    """The running max starts finite so an empty index list cannot make NaN.

    -inf would give exp(-inf - -inf) on the first valid position and poison the
    row; the reference uses -1e30 for exactly this.
    """
    import random
    random.seed(3)
    S, H, N, TOPK, D = 2, 3, 20, 16, 8
    q = [[[random.uniform(-1, 1) for _ in range(D)] for _ in range(H)]
         for _ in range(S)]
    kv = [[random.uniform(-1, 1) for _ in range(D)] for _ in range(N)]
    idxs = [[-1] * TOPK for _ in range(S)]
    out = _kernel_sparse(q, kv, None, idxs, S, H, N, TOPK, D, (1.0 / D) ** 0.5)
    for s in range(S):
        for h in range(H):
            for d in range(D):
                v = out[s][h][d]
                assert v == 0.0 and v == v, "an all-masked row must be zeros"

    src = code(_SPARSE.read_text())
    m = re.search(r"DS_SCORE_FLOOR\s*\(?(-?[0-9.e+]+)f?\)?", src)
    assert m, "the score floor is no longer a named constant"
    assert float(m.group(1)) <= -1e29, (
        f"the floor is {m.group(1)}; it must be finite and very negative"
    )
    assert "-INFINITY" not in src and "-std::numeric_limits<float>::infinity" \
        not in src.replace("DSI_NEG_INF", ""), (
        "an infinite floor makes an all-masked row NaN"
    )


def test_sparse_attn_sink_enters_only_the_denominator():
    """attn_sink has no value vector.

    kernel.py adds exp(attn_sink[h] - scores_max[h]) to sum_exp once, after the
    loop. The accumulator is not touched. Folding the sink in as an extra score
    is algebraically the same ratio, but it is not the reference's form and the
    two part company at the floor, where exp(sink - floor) overflows.
    """
    src = code(_SPARSE.read_text())
    i = src.find("attn_sink != nullptr")
    assert i >= 0, "the sink fold is no longer recognisable"
    # Exactly the statement, not the surrounding lines: a window wide enough to
    # catch neighbouring code also catches the epilogue that legitimately
    # scales acc by 1/sum_exp.
    stmt = src[i:src.index(";", i) + 1]
    assert "sum_exp +=" in stmt, "the sink must be added to the denominator"
    assert "acc" not in stmt, (
        "the sink touches the accumulator; the reference leaves it alone"
    )


# --- engram gate ------------------------------------------------------------

_ENGRAM = _DS / "engram_gate.h"


def _kernel_engram(h, key, w, val, mask, T, HC, D, eps, clamp):
    """Mirror of EngramGateKernel."""
    out = [[[0.0] * D for _ in range(HC)] for _ in range(T)]
    for t in range(T):
        for c in range(HC):
            hv, kv, wv = h[t][c], key[t][c], w[c]
            h_ms = sum(x * x for x in hv) / D
            k_ms = sum(x * x for x in kv) / D
            rstd = (1 / math.sqrt(h_ms + eps)) * (1 / math.sqrt(k_ms + eps))
            dot = (sum(hv[i] * wv[i] * kv[i] for i in range(D))
                   * rstd * (1 / math.sqrt(D)))
            a = max(abs(dot), clamp)
            g = math.sqrt(a)
            if dot < 0:
                g = -g
            gate = 1 / (1 + math.exp(-g))
            if mask is not None and mask[t] == 0:
                gate = 0.0
            out[t][c] = [hv[i] + val[t][i] * gate for i in range(D)]
    return out


def _engram_fixture(seed=4, T=4, HC=4, D=16):
    import random
    random.seed(seed)
    h = [[[random.uniform(-2, 2) for _ in range(D)] for _ in range(HC)]
         for _ in range(T)]
    key = [[[random.uniform(-2, 2) for _ in range(D)] for _ in range(HC)]
           for _ in range(T)]
    w = [[random.uniform(-1, 1) for _ in range(D)] for _ in range(HC)]
    val = [[random.uniform(-1, 1) for _ in range(D)] for _ in range(T)]
    return h, key, w, val, T, HC, D


def test_engram_gate_matches_the_reference():
    h, key, w, val, T, HC, D = _engram_fixture()
    eps, clamp = 1e-6, 1e-6
    mask = [1, 0, 1, 1]
    got = _kernel_engram(h, key, w, val, mask, T, HC, D, eps, clamp)
    for t in range(T):
        for c in range(HC):
            hv, kv, wv = h[t][c], key[t][c], w[c]
            rstd = ((sum(x * x for x in hv) / D + eps) ** -0.5
                    * (sum(x * x for x in kv) / D + eps) ** -0.5)
            dot = sum(hv[i] * wv[i] * kv[i] for i in range(D)) * rstd * D ** -0.5
            gate = 1 / (1 + math.exp(
                -math.copysign(math.sqrt(max(abs(dot), clamp)), dot)))
            if not mask[t]:
                gate = 0.0
            for i in range(D):
                want = hv[i] + gate * val[t][i]
                assert abs(got[t][c][i] - want) < 1e-12


def test_engram_gate_restores_the_sign_after_the_root():
    """copysign(sqrt(|dot|), dot), not sqrt(dot) and not sqrt(|dot|).

    sqrt of a raw negative dot is NaN, and dropping the sign turns every
    negative gate into its positive twin.
    """
    h, key, w, val, T, HC, D = _engram_fixture()
    eps = clamp = 1e-6
    negatives = 0
    for t in range(T):
        for c in range(HC):
            hv, kv, wv = h[t][c], key[t][c], w[c]
            rstd = ((sum(x * x for x in hv) / D + eps) ** -0.5
                    * (sum(x * x for x in kv) / D + eps) ** -0.5)
            dot = sum(hv[i] * wv[i] * kv[i] for i in range(D)) * rstd * D ** -0.5
            if dot < 0:
                negatives += 1
    assert negatives > 0, "fixture has no negative dots; not exercising the sign"

    src = code(_ENGRAM.read_text())
    assert "if (dot < 0.0f) g = -g;" in src, (
        "the sign is not restored after the square root"
    )
    i = src.find("sycl::sqrt(a)")
    assert i >= 0, "the root is no longer applied to the clamped magnitude"
    assert "a < clamp_value" in src, (
        "the clamp must floor the magnitude before the root"
    )


def test_engram_gate_normalises_per_copy_not_jointly():
    """rstd is per (token, hc copy) over dim.

    Reducing across the copies couples them and changes every gate.
    """
    h, key, w, val, T, HC, D = _engram_fixture()
    differs = 0
    for t in range(T):
        joint = sum(sum(x * x for x in h[t][c]) for c in range(HC)) / (HC * D)
        for c in range(HC):
            per = sum(x * x for x in h[t][c]) / D
            if abs(joint - per) > 1e-9:
                differs += 1
    assert differs == T * HC, (
        "per-copy and joint normalisation agree here, so this test is not "
        "exercising the distinction"
    )
    src = code(_ENGRAM.read_text())
    assert "/ (float)D" in src, "the mean must be over dim alone"
    assert "HC *" not in src.split("void operator()")[1].split("gate")[0], (
        "the reduction spans the hc copies"
    )


def test_engram_masked_token_passes_through():
    h, key, w, val, T, HC, D = _engram_fixture()
    mask = [1, 0, 1, 1]
    got = _kernel_engram(h, key, w, val, mask, T, HC, D, 1e-6, 1e-6)
    for c in range(HC):
        for i in range(D):
            assert abs(got[1][c][i] - h[1][c][i]) < 1e-15, (
                "a masked token must pass through untouched"
            )
