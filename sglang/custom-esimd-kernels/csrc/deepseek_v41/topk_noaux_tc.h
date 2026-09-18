// DeepSeek V4.1 no-auxiliary-loss TopK router (noaux_tc), for Intel Arc
// Battlemage (B70).

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

// noaux_tc scores experts independently, so there is no softmax and no global
// reduction across experts.
//
// Group-limited routing: the experts are partitioned into N_GROUP contiguous
// groups and only TOPK_GROUP of them may contribute. A group's rank is the sum
// of its two best selection keys, matching the reference; experts outside the
// surviving groups are masked out before the per-expert top-k. With
// TOPK_GROUP == N_GROUP the mask admits everything and this reduces to plain
// top-k, which is what the non-grouped callers want.
template <int NUM_EXPERTS, int TOP_K, int N_GROUP = 1, int TOPK_GROUP = 1>
inline void compute_noaux_tc_routing(
    simd<fp16, NUM_EXPERTS>& logits,
    simd<fp16, NUM_EXPERTS>& bias,
    simd<int, TOP_K>& out_indices,
    simd<fp16, TOP_K>& out_weights) SYCL_ESIMD_FUNCTION
{
    static_assert(NUM_EXPERTS % N_GROUP == 0,
                  "expert groups must partition the experts evenly");
    static_assert(TOPK_GROUP >= 1 && TOPK_GROUP <= N_GROUP,
                  "TOPK_GROUP must select between one and all groups");
    static_assert(TOP_K <= NUM_EXPERTS, "cannot select more experts than exist");
    // Required order:
    //   scores  = softplus(logits).sqrt()   -- all experts, before selection
    //   indices = topk(scores + bias)       -- bias steers selection only
    //   weights = scores.gather(indices)    -- transformed score, unbiased
    //   weights = weights / (sum + 1e-20) * route_scale
    //
    // The transform must precede selection: sqrtsoftplus is strongly
    // compressive, so a fixed bias carries far more rank distance in the
    // transformed domain and raw-logit selection picks a different expert set
    // whenever bias != 0. Accumulation is in float: fp16 cannot hold a 384-term
    // sum accurately.
    constexpr float ROUTED_SCALING_FACTOR = 1.5f;
    constexpr float NEG_INF = -3.0e38f;
    constexpr float EPS = 1e-20f;  // matches training; NOT norm_eps
    // The logit temperature divides before the transform. It is 1.0 for this
    // model, so it folds away, but sqrtsoftplus is not scale-invariant and a
    // temperature applied after the transform would change the selection.
    constexpr float GATE_TEMP = 1.0f;

    float scores[NUM_EXPERTS];
    #pragma unroll
    for (int i = 0; i < NUM_EXPERTS; ++i) {
        simd<float, 1> x_simd =
            static_cast<float>(logits[i].read()) / GATE_TEMP;
        simd<float, 1> exp_x = sycl::ext::intel::esimd::exp(x_simd);
        simd<float, 1> softplus = sycl::ext::intel::esimd::log(1.0f + exp_x);
        simd<float, 1> sqrt_softplus = sycl::ext::intel::esimd::sqrt(softplus);
        scores[i] = sqrt_softplus[0];
    }

    float sel[NUM_EXPERTS];
    #pragma unroll
    for (int i = 0; i < NUM_EXPERTS; ++i) {
        sel[i] = scores[i] + static_cast<float>(bias[i].read());
    }

    // Group-limited stage. A group scores as the sum of its two best keys, so
    // one very strong expert does not carry a group on its own. Only the best
    // TOPK_GROUP groups stay eligible; the rest are masked to -inf and can no
    // longer win a top-k round.
    if constexpr (N_GROUP > 1 && TOPK_GROUP < N_GROUP) {
        constexpr int GROUP_SIZE = NUM_EXPERTS / N_GROUP;
        float group_key[N_GROUP];
        #pragma unroll
        for (int g = 0; g < N_GROUP; ++g) {
            float b0 = NEG_INF, b1 = NEG_INF;
            #pragma unroll
            for (int i = 0; i < GROUP_SIZE; ++i) {
                const float v = sel[g * GROUP_SIZE + i];
                if (v > b0) { b1 = b0; b0 = v; }
                else if (v > b1) { b1 = v; }
            }
            group_key[g] = b0 + b1;
        }

        bool group_live[N_GROUP];
        #pragma unroll
        for (int g = 0; g < N_GROUP; ++g) group_live[g] = false;
        #pragma unroll
        for (int r = 0; r < TOPK_GROUP; ++r) {
            float best = NEG_INF;
            int best_g = 0;
            #pragma unroll
            for (int g = 0; g < N_GROUP; ++g) {
                if (!group_live[g] && group_key[g] > best) {
                    best = group_key[g];
                    best_g = g;
                }
            }
            group_live[best_g] = true;
            group_key[best_g] = NEG_INF;
        }

        #pragma unroll
        for (int g = 0; g < N_GROUP; ++g) {
            if (!group_live[g]) {
                #pragma unroll
                for (int i = 0; i < GROUP_SIZE; ++i) sel[g * GROUP_SIZE + i] = NEG_INF;
            }
        }
    }

    float weight_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        float max_val = NEG_INF;
        int max_idx = 0;
        #pragma unroll
        for (int i = 0; i < NUM_EXPERTS; ++i) {
            if (sel[i] > max_val) {
                max_val = sel[i];
                max_idx = i;
            }
        }
        // The weight is the unbiased transformed score, not the selection key.
        const float w = scores[max_idx];
        out_indices[k] = max_idx;
        out_weights[k] = (fp16)w;
        weight_sum += w;
        sel[max_idx] = NEG_INF;
    }

    // norm_topk_prob = true, and normalising a single selected expert would
    // make every weight exactly 1 regardless of its score.
    const float inv =
        (TOP_K > 1) ? (1.0f / (weight_sum + EPS)) : 1.0f;
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        out_weights[k] = (fp16)((static_cast<float>(out_weights[k].read()) * inv) * ROUTED_SCALING_FACTOR);
    }
}
