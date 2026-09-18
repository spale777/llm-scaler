#include <c10/util/Exception.h>  // TORCH_CHECK
/* resadd_norm_gemv_fused.h — Fused ResidualAdd + RMSNorm + FP8 GEMV.
 *
 * Combines three operations into a single kernel:
 *   1. Residual add: residual = hidden_states + residual  (in-place)
 *   2. RMSNorm (Gemma-style): normed = residual / rms(residual) * weight
 *      where weight is pre-adjusted (w+1.0 already applied by caller)
 *   3. GEMV: output = normed @ dequant(gemv_weight^T) * scale
 *
 * Designed for Qwen3-Next post_attention_layernorm + MoE router:
 *   hidden_states: [1, K] fp16   (K=2048)
 *   residual:      [1, K] fp16   (updated in-place)
 *   norm_weight:   [K] fp16      (Gemma _gemma_w = original_w + 1.0)
 *   gemv_weight:   [N, K] FP8    (N=512 for router)
 *   gemv_scale:    [1] float32
 *   output:        [1, N] fp16
 *
 * Grid: N work-groups, 1 thread each.
 * Each WG redundantly computes residual_add + norm (data in L3 cache).
 * Only WG 0 writes the updated residual back to global memory.
 *
 * For K=2048 with VL=512: 4 loop iterations for norm, then 4 for GEMV.
 * Interleaved approach: compute norm chunk + GEMV chunk per iteration.
 */

#pragma once
#include "utils.h"

template<int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> fp8_dequant_rng(
    simd<uint8_t, VL> raw, int fp8_mode) {
    simd<uint16_t, VL> u16 = convert<uint16_t>(raw);
    simd<uint16_t, VL> fp8_sign = (u16 >> 7) & 1;
    simd<uint16_t, VL> fp16_bits;

    if (fp8_mode == 0) {
        simd<uint16_t, VL> fp8_exp  = (u16 >> 3) & 0xF;
        simd<uint16_t, VL> fp8_mant = u16 & 0x7;
        fp16_bits = (fp8_sign << 15) | ((fp8_exp + 8) << 10) | (fp8_mant << 7);
        fp16_bits.merge(fp8_sign << 15, fp8_exp == 0);
    } else {
        simd<uint16_t, VL> fp8_exp  = (u16 >> 2) & 0x1F;
        simd<uint16_t, VL> fp8_mant = u16 & 0x3;
        fp16_bits = (fp8_sign << 15) | (fp8_exp << 10) | (fp8_mant << 8);
        fp16_bits.merge(fp8_sign << 15, fp8_exp == 0);
    }

    simd<fp16, VL> wh = fp16_bits.template bit_cast_view<fp16>().read();
    return simd<float, VL>(wh);
}

/* ================================================================
 * Kernel: Fused ResidualAdd + RMSNorm + FP8 GEMV (per-tensor scale)
 *
 * Two-pass approach:
 *   Pass 1: Load hidden+residual, compute residual_add, accumulate
 *           sum-of-squares for RMS, store normed chunks to registers.
 *   Pass 2 (fused with pass 1 second half): GEMV dot product.
 *
 * Since we need the full RMS before normalizing, we do:
 *   Loop 1 (K/VL iters): load h+r, add, compute partial sum_sq
 *   Reduce sum_sq → inv_rms
 *   Loop 2 (K/VL iters): normalize stored residual, load weight, FMA
 * ================================================================ */
/* Residual-only pre-pass: residual[k] = fp16(hidden[k] + residual[k]).
 * A single work-item, run before the GEMV grid, so residual_ptr is settled by
 * the time N work-groups start reading it. */
struct ResAddResidualOnly_kernel {
    const fp16* hidden_ptr;
    fp16*       residual_ptr;
    int K;

    void operator()(sycl::nd_item<1>) const SYCL_ESIMD_KERNEL {
        constexpr int VL = 512;
        int k = 0;
        for (; k + VL <= K; k += VL) {
            simd<fp16, VL> h = block_load<fp16, VL>(hidden_ptr + k);
            simd<fp16, VL> r = block_load<fp16, VL>(residual_ptr + k);
            block_store<fp16, VL>(residual_ptr + k, h + r);
        }
        // Scalar remainder: K is not guaranteed to be a multiple of VL.
        for (; k < K; ++k) {
            residual_ptr[k] = (fp16)(hidden_ptr[k] + residual_ptr[k]);
        }
    }
};

struct ResAddNormGEMV_fp8_pert_kernel {
    fp16*          hidden_ptr;   // [1, K] — input (read-only for this kernel)
    fp16*          residual_ptr; // [1, K] — pre-updated by ResAddResidualOnly_kernel
    const fp16*    norm_w_ptr;   // [K] — Gemma norm weight (w+1.0)
    const uint8_t* gemv_weight;  // [N, K] FP8
    const float*   gemv_scale;   // [1]
    fp16*          output;       // [1, N] — router logits
    fp16*          normed_out;   // [1, K] — normed hidden_states (for MoE experts)
    int N, K;
    float eps;
    int fp8_mode;

    template<int MAX_CHUNKS>
    void run_impl(int n) const SYCL_ESIMD_FUNCTION {
        constexpr int VL = 512;
        int n_chunks = K / VL;

        // No register cache: a simd<float,VL>[MAX_CHUNKS] array is 16-32 KB of
        // GRF against a ~8 KB per-thread budget. Pass 2 re-loads residual from
        // L3, which the pre-pass has already settled.
        float sum_sq = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> added = block_load<fp16, VL>(residual_ptr + offset);

            simd<float, VL> sq = added * added;
            sq.select<256,1>(0) += sq.select<256,1>(256);
            sq.select<128,1>(0) += sq.select<128,1>(128);
            sq.select<64,1>(0) += sq.select<64,1>(64);
            sq.select<32,1>(0) += sq.select<32,1>(32);
            sq.select<16,1>(0) += sq.select<16,1>(16);
            sq.select<8,1>(0) += sq.select<8,1>(8);
            sq.select<4,1>(0) += sq.select<4,1>(4);
            sq.select<2,1>(0) += sq.select<2,1>(2);
            sum_sq += (float)sq[0] + (float)sq[1];
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        simd<float, VL> acc = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;

            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + offset);
            simd<float, VL> res = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> normed = res * inv_rms * nw;

            if (n == 0) {
                block_store<fp16, VL>(normed_out + offset, simd<fp16, VL>(normed));
            }

            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n * K + offset);
            simd<float, VL> w_f = fp8_dequant_rng<VL>(w_raw, fp8_mode);
            acc += normed * w_f;
        }

        acc.select<256,1>(0) += acc.select<256,1>(256);
        acc.select<128,1>(0) += acc.select<128,1>(128);
        acc.select<64,1>(0) += acc.select<64,1>(64);
        acc.select<32,1>(0) += acc.select<32,1>(32);
        acc.select<16,1>(0) += acc.select<16,1>(16);
        acc.select<8,1>(0) += acc.select<8,1>(8);
        acc.select<4,1>(0) += acc.select<4,1>(4);
        acc.select<2,1>(0) += acc.select<2,1>(2);
        float dot = ((float)acc[0] + (float)acc[1]) * *gemv_scale;
        output[n] = fp16(dot);
    }

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_group(0);
        if (n >= N) return;

        // The kernel streams, so MAX_CHUNKS sizes nothing; it only keeps the
        // two arms as distinct instantiations. K is bounded by the host.
        if (K <= 4096) {
            run_impl<8>(n);
        } else {
            run_impl<16>(n);
        }
    }
};

/* Host dispatcher */
inline void resadd_norm_gemv_fp8_pert_host(
    fp16* hidden_ptr,
    fp16* residual_ptr,
    const fp16* norm_w_ptr,
    const uint8_t* gemv_weight,
    const float* gemv_scale,
    fp16* output,
    fp16* normed_out,
    int N, int K,
    float eps,
    int fp8_mode,
    sycl::queue& q)
{
    // The kernel walks K in whole VL=512 chunks with no tail path. The 8192
    // bound limits the redundant per-work-group re-read; it is not a capacity.
    TORCH_CHECK(K % 512 == 0,
                "resadd_norm_gemv_fp8_pert: K must be a multiple of 512, got K=", K);
    TORCH_CHECK(K <= 8192,
                "resadd_norm_gemv_fp8_pert: K must be <= 8192, got K=", K);

    // All N work-groups read residual_ptr, so the residual update runs as its
    // own pass first; the in-order queue settles it before the grid starts.
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(1, 1),
            ResAddResidualOnly_kernel{hidden_ptr, residual_ptr, K});
    });

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(N, 1),
            ResAddNormGEMV_fp8_pert_kernel{
                hidden_ptr, residual_ptr, norm_w_ptr,
                gemv_weight, gemv_scale, output, normed_out,
                N, K, eps, fp8_mode});
    });
}

// ============================================================================
// V2: Templated VL for non-512-aligned K (e.g. gemma4 K=2816, VL=256)
// ============================================================================
template<int VL, int MAX_CHUNKS>
struct ResAddNormGEMV_fp8_pert_v2_kernel {
    fp16*          hidden_ptr;
    fp16*          residual_ptr;
    const fp16*    norm_w_ptr;
    const uint8_t* gemv_weight;
    const float*   gemv_scale;
    fp16*          output;
    fp16*          normed_out;
    int N, K;
    float eps;
    int fp8_mode;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        int n = item.get_group(0);
        if (n >= N) return;

        int n_chunks = K / VL;

        // No register cache: a simd<float,VL>[MAX_CHUNKS] array is VL*4*MC
        // bytes of GRF against a ~8 KB per-thread budget. Pass 2 re-loads
        // residual from L3, which the pre-pass has already warmed.
        float sum_sq = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> added = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> sq = added * added;
            sum_sq += reduce<float>(sq, std::plus<>());
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        simd<float, VL> acc = 0.0f;

        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> nw = block_load<fp16, VL>(norm_w_ptr + offset);
            simd<float, VL> res = block_load<fp16, VL>(residual_ptr + offset);
            simd<float, VL> normed = res * inv_rms * nw;

            if (n == 0) {
                block_store<fp16, VL>(normed_out + offset, simd<fp16, VL>(normed));
            }

            simd<uint8_t, VL> w_raw = block_load<uint8_t, VL>(
                gemv_weight + (size_t)n * K + offset);
            simd<float, VL> wf = fp8_dequant_rng<VL>(w_raw, fp8_mode);

            acc += normed * wf;
        }

        float dot = reduce<float>(acc, std::plus<>()) * (*gemv_scale);
        output[n] = fp16(dot);
    }
};

inline void resadd_norm_gemv_fp8_pert_v2_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* norm_w_ptr,
    const uint8_t* gemv_weight, const float* gemv_scale,
    fp16* output, fp16* normed_out,
    int N, int K, float eps, int fp8_mode, sycl::queue& q)
{
    // Route to original kernel if K%512==0 (no overhead)
    if (K % 512 == 0) {
        resadd_norm_gemv_fp8_pert_host(
            hidden_ptr, residual_ptr, norm_w_ptr,
            gemv_weight, gemv_scale, output, normed_out,
            N, K, eps, fp8_mode, q);
        return;
    }

    TORCH_CHECK(K <= 8192,
                "resadd_norm_gemv_fp8_pert_v2: K must be <= 8192, got K=", K);

    // The only fallback is the VL=512 no-tail kernel.
    TORCH_CHECK(K % 256 == 0 || K % 128 == 0,
                "resadd_norm_gemv_fp8_pert_v2: K must be a multiple of 128, "
                "got K=", K);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(1, 1),
            ResAddResidualOnly_kernel{hidden_ptr, residual_ptr, K});
    });

    #define LAUNCH_RNGV2(V, MC)        q.submit([&](sycl::handler& cgh) {             cgh.parallel_for(sycl::nd_range<1>(N, 1),                 ResAddNormGEMV_fp8_pert_v2_kernel<V, MC>{                     hidden_ptr, residual_ptr, norm_w_ptr,                     gemv_weight, gemv_scale, output, normed_out,                     N, K, eps, fp8_mode});         });

    if (K % 256 == 0) {
        int mc = K / 256;
        if      (mc <= 8)  { LAUNCH_RNGV2(256, 8)  }
        else if (mc <= 16) { LAUNCH_RNGV2(256, 16) }
        else               { LAUNCH_RNGV2(256, 32) }
    } else {
        int mc = K / 128;
        if      (mc <= 16) { LAUNCH_RNGV2(128, 16) }
        else               { LAUNCH_RNGV2(128, 32) }
    }

    #undef LAUNCH_RNGV2
}
