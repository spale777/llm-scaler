// DeepSeek V4.1 KV compressor and rotary embedding.
//
// Follows inference/model.py::Compressor.forward and apply_rotary_emb.
//
// The compressor pools `RATIO` consecutive tokens into one KV latent with a
// learned softmax gate:
//
//   latent[g] = sum_i softmax(score[g, :], over i)[i] * kv[g, i]
//
// The softmax is over the RATIO positions inside a group, per channel: each
// channel of the latent is its own convex combination of that channel across
// the group. Reducing over channels instead would mix dimensions that have no
// reason to compete.
//
// Two further details from the reference:
//
//   - Pooling above ratio 1 runs in fp32 and its weights are promoted to
//     match; ratio 1 is a plain projection with no gate at all. The caller
//     picks the path, but a ratio-1 kernel that still applied a softmax would
//     divide by the single score and return kv unchanged only by accident.
//
//   - The latent is produced BEFORE RoPE. The indexer needs the unrotated
//     form, so the rotation happens afterwards on the caller's side.
//
// The rotary embedding takes adjacent element pairs as complex numbers and
// multiplies by a per-position unit complex. `inverse` conjugates it, which is
// how the attention output has the query's rotation removed so the cache can
// stay in one shared rotated form. V4.1 uses two bases -- rope_theta 10000 for
// the main path and compress_rope_theta 160000 for the compressed one -- so
// the frequency table is an input rather than something this kernel derives.
//
// Layouts:
//   kv      [G, RATIO, D]   float, pre-pool per group
//   score   [G, RATIO, D]   float, gate logits
//   latent  [G, D]          float
//   x       [S, D]          float, rotated in place; D must be even
//   cos/sin [S, D/2]        float, one row per position

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>
#include <limits>

namespace dsv41_compress {
using namespace sycl;
using namespace sycl::ext::intel::esimd;

// One work-item per (group, channel): the softmax is over RATIO values at one
// channel, so that is the exact granularity and no cross-item reduction is
// needed.
template <int RATIO>
struct CompressPoolKernel {
  const float* kv;     // [G, RATIO, D]
  const float* score;  // [G, RATIO, D]
  float* latent;       // [G, D]
  int G, D;

  void operator()(nd_item<1> item) const SYCL_ESIMD_KERNEL {
    const int gid = (int)item.get_global_id(0);
    if (gid >= G * D) return;
    const int d = gid % D;
    const int g = gid / D;

    const size_t base = (size_t)g * RATIO * D + d;

    // Softmax over the group's RATIO entries at this channel.
    float m = -std::numeric_limits<float>::infinity();
#pragma unroll
    for (int i = 0; i < RATIO; ++i) {
      const float v = score[base + (size_t)i * D];
      if (v > m) m = v;
    }
    float denom = 0.0f;
    float w[RATIO];
#pragma unroll
    for (int i = 0; i < RATIO; ++i) {
      simd<float, 1> e = score[base + (size_t)i * D] - m;
      w[i] = exp(e)[0];
      denom += w[i];
    }

    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < RATIO; ++i)
      acc += (w[i] / denom) * kv[base + (size_t)i * D];

    latent[(size_t)g * D + d] = acc;
  }
};

template <int RATIO>
inline void launch_compress_pool(queue& q, const float* kv, const float* score,
                                 float* latent, int G, int D) {
  constexpr int WG = 16;
  const size_t rows = (size_t)G * D;
  const size_t global = ((rows + WG - 1) / WG) * WG;
  CompressPoolKernel<RATIO> kern{kv, score, latent, G, D};
  q.submit([&](handler& h) {
    h.parallel_for(nd_range<1>(range<1>(global), range<1>(WG)), kern);
  });
}

// Rotate adjacent pairs as complex numbers. One work-item per (position,
// pair): pairs are independent, so this is the natural width.
//
// The pair is (x[2i], x[2i+1]) -- adjacent elements, not a split-half layout.
// Rotating halves against each other instead is the classic RoPE transcription
// error and still produces a full, plausible output.
struct RotaryKernel {
  float* x;          // [S, D], rotated in place
  const float* cosv; // [S, D/2]
  const float* sinv; // [S, D/2]
  int S, D;
  int inverse;       // conjugate the rotation

  void operator()(nd_item<1> item) const SYCL_ESIMD_KERNEL {
    const int gid = (int)item.get_global_id(0);
    const int half = D / 2;
    if (gid >= S * half) return;
    const int i = gid % half;
    const int s = gid / half;

    const size_t off = (size_t)s * D + (size_t)i * 2;
    const float re = x[off];
    const float im = x[off + 1];
    const float c = cosv[(size_t)s * half + i];
    float sn = sinv[(size_t)s * half + i];
    if (inverse) sn = -sn;

    x[off] = re * c - im * sn;
    x[off + 1] = re * sn + im * c;
  }
};

inline void launch_rotary(queue& q, float* x, const float* cosv,
                          const float* sinv, int S, int D, bool inverse) {
  constexpr int WG = 16;
  const size_t rows = (size_t)S * (D / 2);
  const size_t global = ((rows + WG - 1) / WG) * WG;
  RotaryKernel kern{x, cosv, sinv, S, D, inverse ? 1 : 0};
  q.submit([&](handler& h) {
    h.parallel_for(nd_range<1>(range<1>(global), range<1>(WG)), kern);
  });
}

}  // namespace dsv41_compress
