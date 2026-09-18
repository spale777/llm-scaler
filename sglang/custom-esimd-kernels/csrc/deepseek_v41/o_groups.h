// DeepSeek V4.1 block-diagonal output projection.
//
// Follows inference/model.py::Attention.forward:
//
//   o    = o.view(bsz, seqlen, n_groups, -1)
//   wo_a = self.wo_a.weight.view(n_groups, o_lora_rank, -1)
//   o    = einsum("bsgd,grd->bsgr", o, wo_a)
//
// wo_a is block diagonal over the groups: group g projects only its own
// heads, so this is n_groups independent [D_IN -> R] matrices rather than one
// [n_groups*D_IN -> n_groups*R]. Running it as a dense Linear would multiply
// every group by every other group's block, which is both n_groups times the
// work and numerically wrong unless the off-diagonal blocks are zero.
//
// convert.py dequantizes wo_a to bf16, so the weight arrives dense per group
// and no dequant happens here.
//
// Layouts:
//   x   [T, G, D_IN]   float, attention output reshaped per group
//   w   [G, R, D_IN]   float, one block per group
//   out [T, G, R]      float

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>

namespace dsv41_ogroups {
using namespace sycl;
using namespace sycl::ext::intel::esimd;

// One work-item per (token, group, output rank). The group's input row is
// reloaded per output rank, but it is the small operand: the weight block is
// the long stream and is read once per work-item.
template <int D_IN>
struct OGroupProjKernel {
  const float* x;    // [T, G, D_IN]
  const float* w;    // [G, R, D_IN]
  float* out;        // [T, G, R]
  int T, G, R;

  void operator()(nd_item<1> item) const SYCL_ESIMD_KERNEL {
    const int gid = (int)item.get_global_id(0);
    if (gid >= T * G * R) return;
    const int r = gid % R;
    const int g = (gid / R) % G;
    const int t = gid / (R * G);

    const float* x_row = x + ((size_t)t * G + g) * D_IN;
    // Group g indexes its own block only: this is what block diagonal means,
    // and using a single [G*D_IN, G*R] weight here would mix the groups.
    const float* w_row = w + ((size_t)g * R + r) * D_IN;

    simd<float, D_IN> xv, wv;
#pragma unroll
    for (int i = 0; i < D_IN; i += 64) {
      xv.template select<64, 1>(i) = block_load<float, 64>(x_row + i);
      wv.template select<64, 1>(i) = block_load<float, 64>(w_row + i);
    }

    out[((size_t)t * G + g) * R + r] =
        reduce<float>(xv * wv, std::plus<>());
  }
};

template <int D_IN>
inline void launch_o_group_proj(queue& q, const float* x, const float* w,
                                float* out, int T, int G, int R) {
  constexpr int WG = 16;
  const size_t rows = (size_t)T * G * R;
  const size_t global = ((rows + WG - 1) / WG) * WG;
  OGroupProjKernel<D_IN> kern{x, w, out, T, G, R};
  q.submit([&](handler& h) {
    h.parallel_for(nd_range<1>(range<1>(global), range<1>(WG)), kern);
  });
}

}  // namespace dsv41_ogroups
