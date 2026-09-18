// DeepSeek V4.1 FP4 GEMM -- SKELETON, NOT FUNCTIONAL. Operands are declared
// and never loaded, the accumulator is never stored, and the DPAS call is
// commented out; the deepseek_v41_fp4_gemm binding refuses rather than return
// the uninitialised output.
//
// A native C = A(FP8) @ B(FP4).T is not reachable on Xe2: XMX supports FP16,
// BF16, INT8, INT4 and INT2 only, the bf8/hf8/e2m1 dpas enumerators target
// PVC-class silicon (they compile for BMG and fail at runtime), and
// sycl_ext_oneapi_fp8 offers conversions, not arithmetic. The viable shape is
// to unpack FP4 to FP16 and use the FP16 DPAS.

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include "fp4_dequant.h"

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;
namespace xmx_ns = sycl::ext::intel::esimd::xmx;

// Native Intel XMX DPAS Kernel
// Natively processes 8x8 blocks using Systolic Arrays
template <int BLOCK_M = 16, int BLOCK_N = 32, int BLOCK_K = 16>
struct DPAS_FP4_GEMM_Kernel {
    const uint8_t* A;          // FP8 activations
    const uint8_t* B_fp4;      // FP4 weights (2 values per byte)
    const uint8_t* scales_b;   // UE8M0 weight scales
    sycl::ext::oneapi::bfloat16* C; // BF16 output
    int M, N, K;
    
    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        const int by = item.get_group(0);
        const int bx = item.get_group(1);
        
        // 8x8 systolic blocks, FP32 accumulation, 16x32 tile to stay inside
        // the Battlemage GRF budget.
        simd<float, BLOCK_M * BLOCK_N> accumulator = 0.0f;
        
        const int K_iters = K / BLOCK_K;
        
        for (int k = 0; k < K_iters; ++k) {
            // Load FP8 activations, cast to FP16 (block_load once functional).
            simd<uint8_t, BLOCK_M * BLOCK_K> a_bytes; 
            
            // Packed FP4 weights.
            simd<uint32_t, (BLOCK_N * BLOCK_K) / 8> b_packed;
            
            // UE8M0 scales.
            simd<uint8_t, BLOCK_N> b_scales;
            
            // Unpacked just-in-time to save registers; left unrolled to keep
            // the instruction cache footprint small.
            simd<fp16, BLOCK_N * BLOCK_K> b_fp16;
            for (int i = 0; i < (BLOCK_N * BLOCK_K) / 8; ++i) {
                b_fp16.template select<8, 1>(i * 8) = unpack_fp4_to_fp16(b_packed[i]);
            }
            
            simd<fp16, BLOCK_N> decoded_scales = decode_ue8m0_scales<BLOCK_N>(b_scales);
            
            // XMX dpas expects B in [K, N] VNNI layout, but b_fp16 is
            // row-major [N, K], so it needs a register-level transpose and VNNI
            // interleave. The 16x32 accumulator is then addressed through 2D
            // strided .select<8, 1>() views to place each 8x8 tile:
            // auto c_tile = accumulator.template select<64, 1>(tile_offset);
            // c_tile = xmx_ns::dpas<8, 8, float, fp16, fp16>(c_tile, b_vnni_tile, a_tile);
        }
    }
};
