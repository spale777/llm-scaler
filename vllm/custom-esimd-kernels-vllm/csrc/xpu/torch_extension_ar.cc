/* Custom IPC all-reduce binding for Intel XPU.
 *
 * Peer buffers are shared with sycl_ext_oneapi_inter_process_communication:
 * ipc_memory::get() yields a portable byte payload, so `handles` carries
 * opaque bytes as strings and the receiver does not re-materialise a
 * process-local file descriptor itself.
 *
 * Two constraints shape the mapping. Peer memory must be opened against the
 * same context the queue runs on, or the pointer faults inside a kernel, so
 * every rank maps through the PyTorch XPU runtime's context. And the flag
 * handshake performs atomics on peer memory, which is undefined unless the
 * device reports atomics_supported for that ordered pair -- peer access is
 * one-directional, so a world of N needs N*(N-1) checks, not half that.
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <Python.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>

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

// Unmap every peer buffer this rank opened. IPC handles are file descriptors,
// so skipping this leaks one per peer per process.
void dispose_ctx(int64_t fa) {
    namespace ipc = sycl::ext::oneapi::experimental::ipc_memory;
    auto* ctx = reinterpret_cast<CustomAllreduceContext*>(fa);
    if (ctx == nullptr || !ctx->ready) return;
    auto stream = c10::xpu::getCurrentXPUStream();
    const sycl::context sctx = stream.queue().get_context();
    for (int p = 0; p < ctx->world_size; ++p) {
        if (p == ctx->rank || ctx->remote_slots[p] == nullptr) continue;
        ipc::close(ctx->remote_slots[p], sctx);
        ctx->remote_slots[p] = nullptr;
        ctx->remote_flags[p] = nullptr;
    }
    ctx->ready = false;
}

void dispose() {
    // Kept for the upstream schema; it carries no context handle, so the
    // mapped peers cannot be reached from here. Callers use dispose_ctx.
}

// The upstream schema passes no context handle, so this entry point cannot
// reach the CustomAllreduceContext it would have to fill. Callers use
// register_buffer_ctx, which takes the handle returned by init_custom_ar.
// Refusing here keeps a no-op from leaving ctx->ready false while the caller
// believes registration succeeded.
void register_buffer(torch::Tensor& t, const std::vector<std::string>& handles,
                     const std::vector<int64_t>& offsets) {
    TORCH_CHECK(false,
                "custom_ar: use register_buffer_ctx(fa, buffer, handles, "
                "offsets); this overload cannot address the context.");
}

// Map every peer's staging buffer into this rank's address space.
//
// `handles` carries one opaque payload per rank, produced by
// ipc_memory::get(). Rank r's own entry is its local pointer, so it is not
// reopened: a process cannot import its own handle.
void register_buffer_ctx(int64_t fa, torch::Tensor& local_buf,
                         const std::vector<std::string>& handles,
                         const std::vector<int64_t>& offsets) {
    namespace ipc = sycl::ext::oneapi::experimental::ipc_memory;

    auto* ctx = reinterpret_cast<CustomAllreduceContext*>(fa);
    TORCH_CHECK(ctx != nullptr, "custom_ar: null context");
    TORCH_CHECK(local_buf.is_xpu() && local_buf.is_contiguous(),
                "custom_ar: staging buffer must be a contiguous XPU tensor");
    TORCH_CHECK(static_cast<int>(handles.size()) == ctx->world_size,
                "custom_ar: ", handles.size(), " handles for world_size ",
                ctx->world_size);
    TORCH_CHECK(handles.size() == offsets.size(),
                "custom_ar: ", handles.size(), " handles but ", offsets.size(),
                " offsets");

    auto stream = c10::xpu::getCurrentXPUStream(local_buf.device().index());
    sycl::queue& q = stream.queue();
    const sycl::device dev = q.get_device();
    const sycl::context sctx = q.get_context();

    TORCH_CHECK(dev.has(sycl::aspect::ext_oneapi_ipc_memory),
                "custom_ar: device does not support inter-process memory "
                "sharing; leave VLLM_XPU_USE_CUSTOM_ALLREDUCE off");

    // The staging buffer holds one slot per rank plus the flag array.
    const int64_t total = local_buf.numel();
    TORCH_CHECK(total % ctx->world_size == 0,
                "custom_ar: staging buffer of ", total,
                " elements does not divide across ", ctx->world_size, " ranks");
    ctx->local_buf = local_buf.data_ptr();
    ctx->slot_stride = static_cast<size_t>(total / ctx->world_size);

    for (int p = 0; p < ctx->world_size; ++p) {
        if (p == ctx->rank) {
            ctx->remote_slots[p] = ctx->local_buf;
            ctx->remote_flags[p] = ctx->local_flags;
            continue;
        }
        TORCH_CHECK(!handles[p].empty(),
                    "custom_ar: empty IPC handle from rank ", p);
        ipc::handle_data_t hd(
            reinterpret_cast<const std::byte*>(handles[p].data()),
            reinterpret_cast<const std::byte*>(handles[p].data()) +
                handles[p].size());
        void* peer = ipc::open(hd, sctx, dev);
        TORCH_CHECK(peer != nullptr,
                    "custom_ar: could not map the buffer exported by rank ", p);
        // The exporter may have handed out a base pointer with the payload at
        // an offset inside the same allocation.
        auto* base = static_cast<std::byte*>(peer) + offsets[p];
        ctx->remote_slots[p] = base;
        ctx->remote_flags[p] =
            base + (size_t)total * local_buf.element_size();
    }
    ctx->ready = true;
}

// Export this rank's staging buffer so peers can map it.
std::string export_buffer_handle(torch::Tensor& t) {
    namespace ipc = sycl::ext::oneapi::experimental::ipc_memory;
    TORCH_CHECK(t.is_xpu() && t.is_contiguous(),
                "custom_ar: buffer must be a contiguous XPU tensor");
    auto stream = c10::xpu::getCurrentXPUStream(t.device().index());
    sycl::queue& q = stream.queue();
    TORCH_CHECK(q.get_device().has(sycl::aspect::ext_oneapi_ipc_memory),
                "custom_ar: device does not support inter-process memory "
                "sharing; leave VLLM_XPU_USE_CUSTOM_ALLREDUCE off");
    ipc::handle h = ipc::get(t.data_ptr(), q.get_context());
    auto bytes = h.data();
    return std::string(reinterpret_cast<const char*>(bytes.data()),
                       bytes.size());
}

// True when every ordered pair can reach the other's memory with atomics.
// enable_peer_access is one-directional, so a world of N needs N*(N-1) checks,
// and atomics on peer memory are undefined when the device denies them -- the
// flag handshake is exactly that.
bool peer_access_supported() {
    auto stream = c10::xpu::getCurrentXPUStream();
    sycl::queue& q = stream.queue();
    const sycl::context sctx = q.get_context();
    const auto devs = sctx.get_devices();
    for (auto a : devs) {
        if (!a.has(sycl::aspect::ext_oneapi_ipc_memory)) return false;
        for (auto b : devs) {
            if (a == b) continue;
            if (!a.ext_oneapi_can_access_peer(
                    b, sycl::ext::oneapi::peer_access::atomics_supported)) {
                return false;
            }
        }
    }
    return true;
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
  m.def("dispose_ctx(int fa) -> ()");
  m.impl("dispose_ctx", torch::kXPU, &dispose_ctx);
  m.def("register_buffer(Tensor t, str[] handles, int[] offsets) -> ()");
  m.impl("register_buffer", torch::kXPU, &register_buffer);
  // The staging buffer is mutated: its storage becomes the slot array the
  // peers write into.
  m.def("register_buffer_ctx(int fa, Tensor(a!) local_buf, str[] handles, int[] offsets) -> ()");
  m.impl("register_buffer_ctx", torch::kXPU, &register_buffer_ctx);
  m.def("export_buffer_handle(Tensor t) -> str");
  m.impl("export_buffer_handle", torch::kXPU, &export_buffer_handle);
  m.def("peer_access_supported() -> bool");
  m.impl("peer_access_supported", torch::kXPU, &peer_access_supported);
  m.def("all_reduce_reg(int fa, Tensor(b!) inp, Tensor(a!) out) -> ()");
  m.impl("all_reduce_reg", torch::kXPU, &all_reduce_reg);
  m.def("all_reduce_unreg(int fa, Tensor(b!) inp, Tensor(c!) reg_buffer, Tensor(a!) out) -> ()");
  m.impl("all_reduce_unreg", torch::kXPU, &all_reduce_unreg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
