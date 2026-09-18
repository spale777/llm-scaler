/* bmg_occupancy.h — hardware thread count of the device a queue runs on.
 *
 * K-split dispatchers size their grid against the number of hardware threads
 * the GPU can keep resident. B60 and B70 are both Battlemage but carry
 * different Xe core counts, so a constant sized for one under-splits K on the
 * other.
 */
#pragma once

#include <sycl/sycl.hpp>

// B70 (BMG-G31): 32 Xe cores x 8 vector engines x 8 hardware threads per engine
// in small-GRF mode. Used when the driver will not report the core count.
static constexpr int BMG_HW_THREADS = 2048;

// Per-core geometry, which does not vary across the Battlemage parts. The
// register file is a fixed 64 KB per engine: 128 registers per thread gives 8
// threads, and -doubleGRF halves that.
static constexpr int BMG_THREADS_PER_XVE = 8;
static constexpr int BMG_XVE_PER_CORE = 8;

// max_compute_units reports the Xe core count, which is the figure that differs
// between the parts. Cached per device: the query reaches the driver and this
// sits on the decode dispatch path.
inline int bmg_hw_threads(sycl::queue& q) {
    static thread_local sycl::device cached_dev;
    static thread_local int cached = 0;
    const sycl::device dev = q.get_device();
    if (cached != 0 && dev == cached_dev) return cached;
    int t = BMG_HW_THREADS;
    try {
        const uint32_t cores =
            dev.get_info<sycl::info::device::max_compute_units>();
        if (cores > 0) {
            t = (int)cores * BMG_XVE_PER_CORE * BMG_THREADS_PER_XVE;
        }
    } catch (const sycl::exception&) {
        // Keep the B70 figure when the driver will not report it.
    }
    cached_dev = dev;
    cached = t;
    return t;
}
