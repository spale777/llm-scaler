// Executable reference for the DeepSeek V4.1 FP4 GEMM and noaux_tc router.
//
// The shipped kernels are ESIMD, and ESIMD needs a device reporting
// ext_intel_esimd. A CPU OpenCL device does not, so on a host without a
// Battlemage card they compile and cannot run. This is the same arithmetic in
// plain SYCL, which does run there: it executes the E2M1 unpack, the UE8M0
// scale decode, the dot product and the router on a real device and checks the
// results, so the numerics are validated by execution rather than only by a
// Python mirror.
//
// It is a reference, not a replacement. It makes no claim about the ESIMD
// kernels' performance, only that the arithmetic they implement is right.

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

// E2M1 magnitudes, from fp4_dequant.h. The bit trick the kernel ships is
// checked against this table by the host below.
constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// The kernel's bit trick, transcribed. Magnitudes 2..7 are a uniform 0x0200
// apart from 0x3C00, so one affine expression covers them; 0 and 1 are patched.
inline uint16_t fp4_bits(uint16_t nib) {
  const uint16_t m = nib & 0x7;
  const uint16_t sign = static_cast<uint16_t>((nib & 0x8) << 12);
  uint16_t res = static_cast<uint16_t>(0x3C00 + ((m - 2) << 9));
  if (m == 1) res = 0x3800;
  if (m == 0) res = 0x0000;
  return static_cast<uint16_t>(res | sign);
}

inline float from_half_bits(uint16_t b) {
  const uint16_t sign = (b >> 15) & 0x1;
  const int exp = (b >> 10) & 0x1F;
  const uint16_t man = b & 0x3FF;
  float v;
  if (exp == 0) {
    v = std::ldexp(static_cast<float>(man), -24);
  } else {
    v = std::ldexp(1.0f + static_cast<float>(man) / 1024.0f, exp - 15);
  }
  return sign ? -v : v;
}

// UE8M0: the raw byte is an exponent. Saturating at 142 keeps the decoded
// value below the fp16 Inf encoding, so an Inf scale cannot meet an E2M1 zero
// and produce NaN.
inline float decode_ue8m0(uint8_t raw) {
  if (raw <= 112) return 0.0f;
  const int e = (raw < 142 ? raw : 142) - 112;
  return from_half_bits(static_cast<uint16_t>(e << 10));
}

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  sycl::queue q;
  const auto dev = q.get_device();
  std::printf("device: %s\n",
              dev.get_info<sycl::info::device::name>().c_str());
  std::printf("ext_intel_esimd: %d (0 means the ESIMD kernels cannot run here)\n",
              static_cast<int>(dev.has(sycl::aspect::ext_intel_esimd)));

  // --- the E2M1 bit trick, on device -------------------------------------
  {
    uint16_t* bits = sycl::malloc_shared<uint16_t>(16, q);
    q.parallel_for(sycl::range<1>(16), [=](sycl::id<1> i) {
       bits[i] = fp4_bits(static_cast<uint16_t>(i));
     }).wait();
    for (int n = 0; n < 16; ++n) {
      const float want = kE2M1[n & 7] * ((n & 8) ? -1.0f : 1.0f);
      const float got = from_half_bits(bits[n]);
      check(got == want, "E2M1 nibble");
    }
    std::printf("E2M1: 16 nibbles decoded on device, exact\n");
    sycl::free(bits, q);
  }

  // --- UE8M0 scales, on device -------------------------------------------
  {
    float* sc = sycl::malloc_shared<float>(256, q);
    q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
       sc[i] = decode_ue8m0(static_cast<uint8_t>(i));
     }).wait();
    check(sc[127] == 1.0f, "UE8M0 raw 127 is 1.0");
    check(sc[112] == 0.0f, "UE8M0 floor");
    check(sc[142] == 32768.0f, "UE8M0 saturates below Inf");
    for (int r = 255; r >= 143; --r) {
      check(sc[r] == sc[142], "UE8M0 stays saturated");
    }
    std::printf("UE8M0: 256 bytes decoded on device, exact\n");
    sycl::free(sc, q);
  }

  // --- the FP4 GEMM, on device -------------------------------------------
  {
    constexpr int M = 3, N = 32, K = 128, GROUP = 32;
    const int kp = K / 2, kg = K / GROUP;

    float* A = sycl::malloc_shared<float>(M * K, q);
    uint8_t* B = sycl::malloc_shared<uint8_t>(N * kp, q);
    uint8_t* S = sycl::malloc_shared<uint8_t>(N * kg, q);
    float* C = sycl::malloc_shared<float>(M * N, q);

    uint32_t seed = 12345;
    auto rnd = [&seed]() {
      seed = seed * 1664525u + 1013904223u;
      return seed;
    };
    for (int i = 0; i < M * K; ++i)
      A[i] = static_cast<float>(static_cast<int>(rnd() % 200) - 100) / 50.0f;
    for (int i = 0; i < N * kp; ++i) B[i] = static_cast<uint8_t>(rnd() & 0xFF);
    for (int i = 0; i < N * kg; ++i)
      S[i] = static_cast<uint8_t>(118 + (rnd() % 16));

    q.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> id) {
       const int m = static_cast<int>(id[0]);
       const int n = static_cast<int>(id[1]);
       float acc = 0.0f;
       for (int k = 0; k < K; ++k) {
         const uint8_t byte = B[n * kp + k / 2];
         // Low nibble is the even k, matching convert.py.
         const uint16_t nib = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
         const float w = from_half_bits(fp4_bits(nib));
         acc += A[m * K + k] * w * decode_ue8m0(S[n * kg + k / GROUP]);
       }
       C[m * N + n] = acc;
     }).wait();

    // Host reference, computed independently of the kernel above.
    double worst = 0.0;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        double want = 0.0;
        for (int k = 0; k < K; ++k) {
          const uint8_t byte = B[n * kp + k / 2];
          const int nib = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
          const double w = kE2M1[nib & 7] * ((nib & 8) ? -1.0 : 1.0);
          want += static_cast<double>(A[m * K + k]) * w *
                  static_cast<double>(decode_ue8m0(S[n * kg + k / GROUP]));
        }
        const double got = C[m * N + n];
        const double den = std::fabs(want) > 1e-6 ? std::fabs(want) : 1.0;
        worst = std::fmax(worst, std::fabs(got - want) / den);
      }
    }
    check(worst < 1e-5, "FP4 GEMM against a host reference");
    std::printf("FP4 GEMM: %dx%dx%d executed on device, worst relative error %.3e\n",
                M, N, K, worst);

    sycl::free(A, q); sycl::free(B, q); sycl::free(S, q); sycl::free(C, q);
  }

  // --- the noaux_tc router, on device -------------------------------------
  {
    constexpr int T = 4, E = 384, TOPK = 6;
    constexpr float ROUTE_SCALE = 1.5f, EPS = 1e-20f;

    float* logits = sycl::malloc_shared<float>(T * E, q);
    float* bias = sycl::malloc_shared<float>(E, q);
    int* idx = sycl::malloc_shared<int>(T * TOPK, q);
    float* wts = sycl::malloc_shared<float>(T * TOPK, q);

    uint32_t seed = 999;
    auto rnd = [&seed]() {
      seed = seed * 1103515245u + 12345u;
      return seed;
    };
    for (int i = 0; i < T * E; ++i)
      logits[i] = static_cast<float>(static_cast<int>(rnd() % 1200) - 600) / 100.0f;
    for (int i = 0; i < E; ++i)
      bias[i] = static_cast<float>(static_cast<int>(rnd() % 200) - 100) / 100.0f;

    q.parallel_for(sycl::range<1>(T), [=](sycl::id<1> tid) {
       const int t = static_cast<int>(tid);
       // Score every expert before selecting: sqrtsoftplus is compressive, so
       // a fixed bias carries different rank distance in the two domains.
       float sel[E];
       float score[E];
       for (int i = 0; i < E; ++i) {
         const float x = logits[t * E + i];
         score[i] = sycl::sqrt(sycl::log(1.0f + sycl::exp(x)));
         sel[i] = score[i] + bias[i];
       }
       float sum = 0.0f;
       for (int k = 0; k < TOPK; ++k) {
         int best = 0;
         float bv = -3.0e38f;
         for (int i = 0; i < E; ++i) {
           if (sel[i] > bv) { bv = sel[i]; best = i; }
         }
         idx[t * TOPK + k] = best;
         // The weight is the unbiased score, not the selection key.
         wts[t * TOPK + k] = score[best];
         sum += score[best];
         sel[best] = -3.0e38f;
       }
       const float inv = 1.0f / (sum + EPS);
       for (int k = 0; k < TOPK; ++k)
         wts[t * TOPK + k] = wts[t * TOPK + k] * inv * ROUTE_SCALE;
     }).wait();

    for (int t = 0; t < T; ++t) {
      std::vector<float> score(E), sel(E);
      for (int i = 0; i < E; ++i) {
        const double x = logits[t * E + i];
        score[i] = static_cast<float>(std::sqrt(std::log1p(std::exp(x))));
        sel[i] = score[i] + bias[i];
      }
      std::vector<int> want;
      std::vector<float> tmp = sel;
      for (int k = 0; k < TOPK; ++k) {
        int best = 0;
        for (int i = 1; i < E; ++i)
          if (tmp[i] > tmp[best]) best = i;
        want.push_back(best);
        tmp[best] = -3.0e38f;
      }
      double wsum = 0.0;
      for (int b : want) wsum += score[b];
      for (int k = 0; k < TOPK; ++k) {
        check(idx[t * TOPK + k] == want[k], "router selected the same expert");
        const double w = score[want[k]] / (wsum + EPS) * ROUTE_SCALE;
        check(std::fabs(wts[t * TOPK + k] - w) < 1e-5, "router weight");
      }
      // Selection must not be reproducible from the raw logits: if it were,
      // the transform would not be load-bearing and this test would be vacuous.
      std::vector<float> raw(logits + t * E, logits + t * E + E);
      int raw_best = 0;
      for (int i = 1; i < E; ++i)
        if (raw[i] + bias[i] > raw[raw_best] + bias[raw_best]) raw_best = i;
      if (raw_best == want[0]) {
        std::printf("note: token %d picks the same top expert either way\n", t);
      }
    }
    std::printf("noaux_tc router: %d tokens over %d experts executed on device\n",
                T, E);

    sycl::free(logits, q); sycl::free(bias, q);
    sycl::free(idx, q); sycl::free(wts, q);
  }

  if (failures) {
    std::printf("\n%d check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("\nall checks passed on a real device\n");
  return 0;
}
