#pragma once
#include <cstddef>
#include <cstdint>

/* Push-model IPC all-reduce; ipc_allreduce.sycl carries the layout and
 * synchronisation contract.
 *
 *   local_buf    this rank's staging buffer, world_size slots of slot_stride
 *   remote_slots [world_size] base of THIS rank's slot inside each peer
 *   local_flags  [world_size] flags in this rank's buffer, written by peers
 *   remote_flags [world_size] this rank's flag word inside each peer
 *   dtype        0 = float32, 1 = float16, 2 = bfloat16
 *   seq          monotonically increasing sequence for this collective
 */
void execute_ipc_allreduce_push_wrapper(void* q_ptr, const void* local_in,
                                        void* local_buf, void* out,
                                        void** remote_slots,
                                        uint32_t* local_flags,
                                        void** remote_flags, size_t elements,
                                        size_t slot_stride, int rank,
                                        int world_size, int dtype,
                                        uint32_t seq);
