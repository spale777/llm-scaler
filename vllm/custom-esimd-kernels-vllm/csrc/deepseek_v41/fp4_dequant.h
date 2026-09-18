// DeepSeek V4.1 FP4 dequantization (E2M1 values, UE8M0 scales), for Intel Arc
// Battlemage (B70).

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

// E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit; magnitudes clamp at 6.0.
static const fp16 fp4_e2m1_lut[16] = {
    0.0f,  0.5f,  // 0000, 0001
    1.0f,  1.5f,  // 0010, 0011
    2.0f,  3.0f,  // 0100, 0101
    4.0f,  6.0f,  // 0110, 0111
   -0.0f, -0.5f,  // 1000, 1001
   -1.0f, -1.5f,  // 1010, 1011
   -2.0f, -3.0f,  // 1100, 1101
   -4.0f, -6.0f   // 1110, 1111
};

// UE8M0 scale: an 8-bit unsigned exponent denoting 2^(val - 127).
template <int N>
inline simd<fp16, N> decode_ue8m0_scales(simd<uint8_t, N> raw_scales) SYCL_ESIMD_FUNCTION {
    // The 8-bit exponent goes straight into the fp16 exponent field (1 sign,
    // 5 exp, 10 mantissa); rebiasing from 127 to 15 is the -112 below.
    simd<uint16_t, N> fp16_bits = 0;
    simd<uint16_t, N> shifted = raw_scales;
    // Exponent field 31 is the fp16 Inf/NaN encoding, so the raw scale
    // saturates at 142 (field 30, 32768.0): Inf against an E2M1 zero gives NaN
    // and poisons the whole output tile.
    shifted = sycl::ext::intel::esimd::min(shifted, (uint16_t)142);
    shifted = shifted - 112;
    fp16_bits.merge(shifted, raw_scales > 112);
    
    fp16_bits = fp16_bits << 10; // Shift into exponent position
    return fp16_bits.template bit_cast_view<fp16>();
}

// Eight FP4 values per packed 32-bit word.
inline simd<fp16, 8> unpack_fp4_to_fp16(uint32_t packed) SYCL_ESIMD_FUNCTION {
    // bit_cast_view needs an lvalue: it returns a view into the object.
    simd<uint32_t, 1> packed_v(packed);
    simd<uint8_t, 4> bytes = packed_v.bit_cast_view<uint8_t>();
    simd<uint16_t, 8> nibbles;
    nibbles.select<4, 2>(0) = bytes & 0xF;
    nibbles.select<4, 2>(1) = bytes >> 4;

    simd<uint16_t, 8> m = nibbles & 0x7;
    simd<uint16_t, 8> sign = (nibbles & 0x8) << 12; // sign to bit 15

    // fp16 bit patterns for the E2M1 magnitudes:
    //   m: 0      1      2      3      4      5      6      7
    //      0.0    0.5    1.0    1.5    2.0    3.0    4.0    6.0
    //      0x0000 0x3800 0x3C00 0x3E00 0x4000 0x4200 0x4400 0x4600
    // From m=2 the stride is a uniform 0x0200, so the expression covers
    // m >= 2 exactly; m = 0 and m = 1 are patched in.
    simd<uint16_t, 8> res = 0x3C00 + ((m - 2) << 9);

    res.merge(0x3800, m == 1);
    res.merge(0x0000, m == 0);

    res |= sign;
    return res.bit_cast_view<fp16>();
}
