// DeepSeek V4.1 FP4 GEMM: C[M, N] = A[M, K] @ dequant(B_fp4[N, K]).T
//
// Xe2 XMX takes FP16, BF16, INT8, INT4 and INT2 only. The bf8/hf8/e2m1 dpas
// enumerators in the ESIMD headers target PVC-class silicon: they compile for
// Battlemage and fail at runtime. So FP4 is a storage format here -- the
// weights are unpacked to FP16 in registers and fed to the FP16 DPAS, which is
// the only shape that runs on this hardware.
//
// Layouts (row-major, contiguous):
//   A        [M, K]            fp16
//   B_fp4    [N, K/2]          uint8, two E2M1 nibbles per byte, low nibble
//                              is the even k
//   scales   [N, K/GROUP]      uint8, UE8M0, one per GROUP contiguous k
//   C        [M, N]            fp16
//
// One work-item owns an N_TILE of 16 output channels and walks all of K,
// accumulating a 16-wide row per M row of its M_TILE. The weight nibbles are
// unpacked once per k step and reused across every M row in the tile, so the
// weight stream -- which dominates at decode -- is read once.

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include "fp4_dequant.h"

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;
namespace xmx_ns = sycl::ext::intel::esimd::xmx;
namespace xesimd = sycl::ext::intel::experimental::esimd;

// Unpack 16 packed bytes (32 FP4 values) spanning one k run of a single N row.
// Nibble order matches convert.py: the low nibble is the lower k index.
template <int NBYTES>
inline simd<fp16, NBYTES * 2> unpack_fp4_row(simd<uint8_t, NBYTES> packed)
    SYCL_ESIMD_FUNCTION {
    simd<uint16_t, NBYTES> lo = packed & 0xF;
    simd<uint16_t, NBYTES> hi = (packed >> 4) & 0xF;

    simd<uint16_t, NBYTES * 2> nib;
    nib.template select<NBYTES, 2>(0) = lo;
    nib.template select<NBYTES, 2>(1) = hi;

    simd<uint16_t, NBYTES * 2> m = nib & 0x7;
    simd<uint16_t, NBYTES * 2> sign = (nib & 0x8) << 12;

    // Magnitudes 2..7 are a uniform 0x0200 apart from 0x3C00, so one affine
    // expression covers them; 0 and 1 are patched in.
    simd<uint16_t, NBYTES * 2> res = 0x3C00 + ((m - 2) << 9);
    res.merge(0x3800, m == 1);
    res.merge(0x0000, m == 0);
    res |= sign;
    return res.template bit_cast_view<fp16>();
}

// GROUP is the number of k elements sharing one UE8M0 scale (32 in V4.1).
template <int M_TILE, int N_TILE, int GROUP>
struct FP4_GEMM_Kernel {
    const fp16*    A;        // [M, K]
    const uint8_t* B_fp4;    // [N, K/2]
    const uint8_t* scales;   // [N, K/GROUP]
    fp16*          C;        // [M, N]
    int M, N, K;

    void operator()(sycl::nd_item<2> item) const SYCL_ESIMD_KERNEL {
        static_assert(N_TILE == 16, "the DPAS B operand is 16 channels wide");
        static_assert(GROUP % 32 == 0, "a scale group must cover whole k steps");

        const int m0 = (int)item.get_global_id(0) * M_TILE;
        const int n0 = (int)item.get_global_id(1) * N_TILE;
        if (m0 >= M || n0 >= N) return;

        const int K_packed = K / 2;
        const int K_groups = K / GROUP;

        // Accumulate in fp32: a 7168-term dot product in fp16 loses too much.
        simd<float, M_TILE * N_TILE> acc = 0.0f;

        // Scales for these N_TILE channels are K_groups apart, so one strided
        // gather issues the run rather than N_TILE dependent scalar loads.
        const simd<uint32_t, N_TILE> scl_off =
            simd<uint32_t, N_TILE>(0u, 1u) * (uint32_t)K_groups;

        for (int k = 0; k < K; k += 32) {
            const int kg = k / GROUP;
            simd<uint8_t, N_TILE> raw_scl =
                gather<uint8_t, N_TILE>(scales + (size_t)n0 * K_groups + kg, scl_off);
            simd<fp16, N_TILE> scl = decode_ue8m0_scales<N_TILE>(raw_scl);

            // B rows for this k run: 16 bytes per channel covers 32 k values.
            simd<fp16, N_TILE * 32> b_rows;
            #pragma unroll
            for (int n = 0; n < N_TILE; ++n) {
                const uint8_t* brow = B_fp4 + (size_t)(n0 + n) * K_packed + k / 2;
                if (k + 32 < K) {
                    xesimd::lsc_prefetch<uint8_t, 16,
                                         xesimd::lsc_data_size::default_size,
                                         xesimd::cache_hint::cached,
                                         xesimd::cache_hint::cached>(brow + 16);
                }
                simd<fp16, 32> vals = unpack_fp4_row<16>(block_load<uint8_t, 16>(brow));
                b_rows.template select<32, 1>(n * 32) = vals * (fp16)scl[n];
            }

            // Two DPAS steps of depth 16: the systolic depth is 8 and fp16
            // carries two elements per channel, so one dpas consumes k=16.
            #pragma unroll
            for (int ks = 0; ks < 32; ks += 16) {
                // B operand in VNNI2 [k/2][n][2] order.
                simd<fp16, N_TILE * 16> b_vnni;
                #pragma unroll
                for (int n = 0; n < N_TILE; ++n) {
                    #pragma unroll
                    for (int kk = 0; kk < 16; ++kk) {
                        b_vnni[(kk / 2) * (N_TILE * 2) + n * 2 + (kk & 1)] =
                            b_rows[n * 32 + ks + kk];
                    }
                }

                #pragma unroll
                for (int mt = 0; mt < M_TILE; ++mt) {
                    const int mrow = m0 + mt;
                    simd<fp16, 16> a_tile = 0;
                    if (mrow < M) {
                        a_tile = block_load<fp16, 16>(A + (size_t)mrow * K + k + ks);
                    }
                    simd<float, N_TILE> c = acc.template select<N_TILE, 1>(mt * N_TILE);
                    c = xmx_ns::dpas<8, 1, float, float, fp16, fp16>(c, b_vnni, a_tile);
                    acc.template select<N_TILE, 1>(mt * N_TILE) = c;
                }
            }
        }

        #pragma unroll
        for (int mt = 0; mt < M_TILE; ++mt) {
            const int mrow = m0 + mt;
            if (mrow >= M) continue;
            simd<float, N_TILE> c = acc.template select<N_TILE, 1>(mt * N_TILE);
            block_store<fp16, N_TILE>(C + (size_t)mrow * N + n0, convert<fp16>(c));
        }
    }
};
