/* Custom IPC all-reduce binding for Intel XPU.
 *
 * The Level Zero handle exchange is unimplemented: register_buffer rejects the
 * call, so the op is registered but not usable. Implementing it requires a
 * {pid, ze_ipc_mem_handle_t} per rank -- the handle is char[64] whose first
 * four bytes are a process-local dma-buf fd, so the receiver re-materialises it
 * (pidfd_open + pidfd_getfd, or SCM_RIGHTS) before zeMemOpenIpcHandle, which is
 * why `handles` carries opaque bytes as strings -- peer memory opened against
 * the queue's own ze_context_handle_t, and can_access_peer(peer,
 * atomics_supported) over every ordered pair, since the flag handshake uses
 * atomics on peer memory.
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <Python.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include "esimd_kernels/ipc_allreduce.h"

namespace {

constexpr int kMaxPeers = 8;

struct CustomAllreduceContext {
    int rank = 0;
    int world_size = 0;
    bool ready = false;          // true only once every peer pointer is mapped
    uint32_t seq = 0;            // monotonically increasing collective counter
    void* local_buf = nullptr;   // staging buffer, world_size slots
    size_t slot_stride = 0;      // elements per slot
    uint32_t* local_flags = nullptr;
    void* remote_slots[kMaxPeers] = {};
    void* remote_flags[kMaxPeers] = {};
};

int dtype_code(const torch::Tensor& t) {
    switch (t.scalar_type()) {
        case at::ScalarType::Float: return 0;
        case at::ScalarType::Half: return 1;
        case at::ScalarType::BFloat16: return 2;
        default: return -1;
    }
}

}  // namespace

int64_t meta_size() { return sizeof(CustomAllreduceContext); }

void init_custom_ar(torch::Tensor& meta, torch::Tensor& rank_data,
                    const std::vector<std::string>& handles,
                    const std::vector<int64_t>& offsets, int64_t rank,
                    bool full_nvlink) {
    TORCH_CHECK(meta.numel() * meta.element_size() >=
                    static_cast<int64_t>(sizeof(CustomAllreduceContext)),
                "init_custom_ar: meta tensor is smaller than meta_size()");
    TORCH_CHECK(static_cast<int64_t>(handles.size()) <= kMaxPeers,
                "init_custom_ar: world_size ", handles.size(),
                " exceeds the compiled peer limit of ", kMaxPeers);
    TORCH_CHECK(rank >= 0 && rank < static_cast<int64_t>(handles.size()),
                "init_custom_ar: rank ", rank, " outside world_size ",
                handles.size());

    auto* ctx = new (meta.data_ptr()) CustomAllreduceContext();
    ctx->rank = static_cast<int>(rank);
    ctx->world_size = static_cast<int>(handles.size());
    ctx->local_flags = reinterpret_cast<uint32_t*>(rank_data.data_ptr());
    ctx->ready = false;  // set by register_buffer once peers are mapped
}

void dispose() {
    // Must call zeMemCloseIpcHandle for every mapped peer once the exchange is
    // implemented: L0 IPC handles are file descriptors and leak otherwise.
}

void register_buffer(torch::Tensor& t, const std::vector<std::string>& handles,
                     const std::vector<int64_t>& offsets) {
    TORCH_CHECK(false,
                "custom_ar: Level Zero IPC handle exchange is not implemented. "
                "The op is registered but unusable; leave "
                "VLLM_XPU_USE_CUSTOM_ALLREDUCE off so the communicator selects "
                "oneCCL.");
}

void all_reduce_reg(int64_t fa, torch::Tensor& inp, torch::Tensor& out) {
    auto* ctx = reinterpret_cast<CustomAllreduceContext*>(fa);
    TORCH_CHECK(ctx != nullptr, "custom_ar: null context");

    TORCH_CHECK(ctx->ready,
                "custom_ar: peer buffers are not registered. Refusing to return "
                "an unreduced tensor.");

    TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(),
                "custom_ar: inputs must be contiguous");
    TORCH_CHECK(inp.sizes() == out.sizes(),
                "custom_ar: input and output shapes differ");
    TORCH_CHECK(inp.scalar_type() == out.scalar_type(),
                "custom_ar: input and output dtypes differ");

    const int dt = dtype_code(inp);
    TORCH_CHECK(dt >= 0, "custom_ar: unsupported dtype ", inp.scalar_type(),
                "; expected float32, float16 or bfloat16");

    const size_t elements = static_cast<size_t>(inp.numel());
    TORCH_CHECK(elements <= ctx->slot_stride,
                "custom_ar: tensor of ", elements,
                " elements exceeds the registered slot capacity of ",
                ctx->slot_stride);

    // The tensor's own stream orders this against the producing matmul.
    auto& q = c10::xpu::getCurrentXPUStream(inp.device().index()).queue();

    ctx->seq += 1;
    execute_ipc_allreduce_push_wrapper(
        &q, inp.data_ptr(), ctx->local_buf, out.data_ptr(), ctx->remote_slots,
        ctx->local_flags, ctx->remote_flags, elements, ctx->slot_stride,
        ctx->rank, ctx->world_size, dt, ctx->seq);
}

void all_reduce_unreg(int64_t fa, torch::Tensor& inp, torch::Tensor& reg_buffer,
                      torch::Tensor& out) {
    // `inp` is not IPC-mapped here, so it is staged through reg_buffer.
    TORCH_CHECK(reg_buffer.numel() >= inp.numel(),
                "custom_ar: staging buffer smaller than the input");
    reg_buffer.narrow(0, 0, inp.numel()).copy_(inp.view(-1));
    auto staged = reg_buffer.narrow(0, 0, inp.numel()).view_as(inp);
    all_reduce_reg(fa, staged, out);
}

TORCH_LIBRARY_FRAGMENT(_C_custom_ar, m) {
  m.def("meta_size() -> int");
  m.impl("meta_size", torch::kXPU, &meta_size);
  // Both tensors are mutated: a CustomAllreduceContext is placement-new'd into
  // meta's storage, and rank_data's storage is captured as the writable
  // uint32_t* the flag protocol writes through.
  m.def("init_custom_ar(Tensor(a!) meta, Tensor(b!) rank_data, str[] handles, int[] offsets, int rank, bool full_nvlink) -> ()");
  m.impl("init_custom_ar", torch::kXPU, &init_custom_ar);
  m.def("dispose() -> ()");
  m.impl("dispose", torch::kXPU, &dispose);
  m.def("register_buffer(Tensor t, str[] handles, int[] offsets) -> ()");
  m.impl("register_buffer", torch::kXPU, &register_buffer);
  m.def("all_reduce_reg(int fa, Tensor(b!) inp, Tensor(a!) out) -> ()");
  m.impl("all_reduce_reg", torch::kXPU, &all_reduce_reg);
  m.def("all_reduce_unreg(int fa, Tensor(b!) inp, Tensor(c!) reg_buffer, Tensor(a!) out) -> ()");
  m.impl("all_reduce_unreg", torch::kXPU, &all_reduce_unreg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
