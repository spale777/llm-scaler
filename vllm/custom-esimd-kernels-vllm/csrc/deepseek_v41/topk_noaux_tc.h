// DeepSeek V4.1 no-auxiliary-loss TopK router (noaux_tc), for Intel Arc
// Battlemage (B70).

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

// noaux_tc scores experts independently, so there is no softmax and no global
// reduction across experts.
template <int NUM_EXPERTS, int TOP_K>
inline void compute_noaux_tc_routing(
    simd<fp16, NUM_EXPERTS>& logits,
    simd<fp16, NUM_EXPERTS>& bias,
    simd<int, TOP_K>& out_indices,
    simd<fp16, TOP_K>& out_weights) SYCL_ESIMD_FUNCTION
{
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

    float scores[NUM_EXPERTS];
    #pragma unroll
    for (int i = 0; i < NUM_EXPERTS; ++i) {
        simd<float, 1> x_simd = static_cast<float>(logits[i].read());
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

    // norm_topk_prob = true.
    const float inv = 1.0f / (weight_sum + EPS);
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        out_weights[k] = (fp16)((static_cast<float>(out_weights[k].read()) * inv) * ROUTED_SCALING_FACTOR);
    }
}
