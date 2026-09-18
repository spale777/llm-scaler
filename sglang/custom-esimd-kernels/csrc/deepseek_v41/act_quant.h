// DeepSeek V4.1 activation quantization: block-wise FP8 and FP4 with
// power-of-two (UE8M0) scales.
//
// Follows inference/kernel.py::act_quant_kernel and fp4_act_quant_kernel. Each
// row is cut into groups of `GROUP` elements; a group's scale is derived from
// its absolute max and every element is divided by it and clamped.
//
// The scale rounding is the part that must be exact. With `scale_fmt` set the
// scale is forced to a power of two so it fits a UE8M0 byte, and the reference
// does that by bit manipulation rather than log2/ceil:
//
//   fast_log2_ceil(x) = (exponent(x) - 127) + (mantissa(x) != 0)
//   fast_pow2(e)      = bitcast((e + 127) << 23)
//   scale             = fast_pow2(fast_log2_ceil(amax * max_inv))
//
// Rounding UP (the +1 when the mantissa is non-zero) is what keeps the
// quotient inside the representable range: rounding to nearest would let the
// largest element of a group exceed the format's max and clamp, which loses
// the very value the scale was chosen to preserve.
//
// The amax floor differs by format and is not cosmetic. FP8 floors at 1e-4;
// FP4 floors at 6 * 2^-126, which is the smallest amax whose scale is still
// normal. Without it an all-zero group yields a zero or subnormal scale and
// the division produces inf or NaN.
//
// Layouts:
//   X  [M, N]                 float
//   Y  [M, N]                 uint8, quantized codes
//   S  [M, ceil(N / GROUP)]   float (FP8 path) or uint8 UE8M0 (FP4 path)

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>

namespace dsv41_quant {
using namespace sycl;
using namespace sycl::ext::intel::esimd;

// ceil(log2(x)) by exponent extraction; the mantissa test is the ceiling.
ESIMD_INLINE int fast_log2_ceil(float x) {
  simd<float, 1> v = x;
  const uint32_t bits = v.bit_cast_view<uint32_t>()[0];
  const int exp = (int)((bits >> 23) & 0xFF);
  const uint32_t man = bits & ((1u << 23) - 1u);
  return exp - 127 + (man != 0u ? 1 : 0);
}

ESIMD_INLINE float fast_pow2(int e) {
  // The caller's exponent stays inside the normal range because the amax floor
  // bounds it from below and the format max bounds it from above.
  simd<uint32_t, 1> bits = (uint32_t)((e + 127) << 23);
  return bits.bit_cast_view<float>()[0];
}

ESIMD_INLINE float fast_round_scale(float amax, float max_inv) {
  return fast_pow2(fast_log2_ceil(amax * max_inv));
}

// FP8 E4M3 finite range, matching the reference's clamp.
static constexpr float FP8_MAX = 448.0f;
static constexpr float FP8_MIN = -448.0f;
static constexpr float FP8_AMAX_FLOOR = 1e-4f;
// FP4 E2M1 tops out at 6.0. The floor is the smallest amax whose rounded scale
// is still a normal float.
static constexpr float FP4_MAX = 6.0f;

// One work-item per (row, group): that is the granularity the scale is defined
// at, so no cross-item reduction is needed.
template <int GROUP, bool ROUND_SCALE>
struct ActQuantKernel {
  const float* X;   // [M, N]
  uint8_t* Y;       // [M, N] quantized codes
  float* S;         // [M, N/GROUP]
  int M, N, Groups;
  float qmax;       // 448 for FP8, 6 for FP4
  float amax_floor;

  void operator()(nd_item<1> item) const SYCL_ESIMD_KERNEL {
    const int gid = (int)item.get_global_id(0);
    if (gid >= M * Groups) return;
    const int g = gid % Groups;
    const int m = gid / Groups;

    const int base = g * GROUP;
    const int len = (base + GROUP <= N) ? GROUP : (N - base);
    if (len <= 0) return;

    const float* x_row = X + (size_t)m * N + base;

    simd<float, GROUP> xv = 0.0f;
    for (int i = 0; i < len; ++i) xv[i] = x_row[i];

    // Only the live lanes may contribute: a short tail group would otherwise
    // take its amax from the zeros padding it, which is harmless, or from
    // stale register content, which is not.
    simd<float, GROUP> av = abs(xv);
    float amax = 0.0f;
    for (int i = 0; i < len; ++i) {
      const float a = av[i];
      if (a > amax) amax = a;
    }
    if (amax < amax_floor) amax = amax_floor;

    const float max_inv = 1.0f / qmax;
    const float s = ROUND_SCALE ? fast_round_scale(amax, max_inv)
                                : amax * max_inv;

    const float inv_s = 1.0f / s;
    for (int i = 0; i < len; ++i) {
      float q = xv[i] * inv_s;
      if (q > qmax) q = qmax;
      if (q < -qmax) q = -qmax;
      // The caller owns the code's encoding; this writes the clamped
      // magnitude's nearest integer, which is what both formats' encoders
      // consume.
      Y[(size_t)m * N + base + i] = (uint8_t)(int)(q < 0.0f ? -q : q);
    }
    S[(size_t)m * Groups + g] = s;
  }
};

template <int GROUP, bool ROUND_SCALE>
inline void launch_act_quant(queue& q, const float* X, uint8_t* Y, float* S,
                             int M, int N, float qmax, float amax_floor) {
  const int groups = (N + GROUP - 1) / GROUP;
  constexpr int WG = 16;
  const size_t rows = (size_t)M * groups;
  const size_t global = ((rows + WG - 1) / WG) * WG;
  ActQuantKernel<GROUP, ROUND_SCALE> kern{X, Y,    S,    M,
                                          N, groups, qmax, amax_floor};
  q.submit([&](handler& h) {
    h.parallel_for(nd_range<1>(range<1>(global), range<1>(WG)), kern);
  });
}

}  // namespace dsv41_quant
