/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Jan Ciesko (jciesko@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef RACERLIB_INTERFACE_H
#define RACERLIB_INTERFACE_H

#include <Kokkos_Core.hpp>
#include <RDMA_Engine.hpp>

namespace RACERlib {

/*
Kokkos::MemoryTraits<Kokkos::Cached>
Kokkos::MemoryTraits<Kokkos::Aggregated>
*/

template <typename Type, typename Traits>
void put(void *comm_id, void *allocation, Type &value, int PE, int offset);
template <typename Type, typename Traits>
Type get(void *comm_id, void *allocation, int PE, int offset);

void start(void *comm_id, void *allocation_id) {
  // Call this at View memory allocation (allocation record)
}
void stop(void *comm_id, void *allocation_id);
{
  // Call this at View memory deallocation (~allocation record);
}

void flush(void *comm_id, void *allocation) {
  // Call this on fence. We need to make sure that at sychronization points,
  // caches are empty
}

int init(void *comm_id); // set communicator reference, return RACERLIB_STATUS
{
  // Call this on Kokkos initialize.
}
int finalize(
    void *comm_id); // finalize communicator instance, return RECERLIB_STATUS
{
  // Call this on kokkos finalize
}

// Todo: template this on Feature for generic Engine feature support
struct Engine {
  RdmaScatterGatherEngine *sge;
  RdmaScatterGatherWorker *sgw;

  static std::set<RdmaScatterGatherEngine *> sges;

  Engine() {}

  void allocate_device_component(void *p, MPI_Comm comm) {
    // Create here a persistent kernel with functor (polling)
    // Call into Feature::Worker();
    size_t header_size = 0x1;
    sge = new RdmaScatterGatherEngine(comm, buf, elem_size, header_size);
    sges.insert(sge);

    RdmaScatterGatherWorker dev_worker;

    dev_worker.tx_element_request_ctrs =
        sge->tx_element_request_ctrs; // already a device buffer
    dev_worker.rx_element_reply_queue =
        sge->rx_element_reply_queue_mr->addr; // already a device buffer
    dev_worker.tx_element_reply_queue = sge->tx_element_reply_queue_mr->addr;
    dev_worker.tx_element_request_trip_counts =
        sge->tx_element_request_trip_counts; // already a device buffer
    dev_worker.cache = sge->cache;
    dev_worker.direct_ptrs = sge->direct_ptrs_d; // already a device buffer
    dev_worker.tx_element_request_queue =
        (uint32_t *)sge->tx_element_request_queue_mr->addr;
    dev_worker.ack_ctrs_d = sge->ack_ctrs_d;
    dev_worker.tx_element_reply_ctrs = sge->tx_element_reply_ctrs;
    dev_worker.rx_element_request_queue =
        (uint32_t *)sge->rx_element_request_queue_mr->addr;
    dev_worker.num_pes = sge->num_pes;
    dev_worker.tx_element_aggregate_ctrs = sge->tx_element_aggregate_ctrs;
    dev_worker.tx_block_request_cmd_queue = sge->tx_block_request_cmd_queue;
    dev_worker.tx_block_reply_cmd_queue = sge->tx_block_reply_cmd_queue;
    dev_worker.tx_block_request_ctr = sge->tx_block_request_ctr;
    dev_worker.rx_block_request_cmd_queue = sge->rx_block_request_cmd_queue;
    dev_worker.rx_block_request_ctr = sge->rx_block_request_ctr;
    dev_worker.tx_element_request_ctrs = sge->tx_element_request_ctrs;
    cuda_safe(cuMemHostGetDevicePointer((CUdeviceptr *)&dev_worker.ack_ctrs_h,
                                        sge->ack_ctrs_h, 0));
    dev_worker.rank = sge->rank;
    dev_worker.request_done_flag = sge->request_done_flag;
    dev_worker.response_done_flag = sge->response_done_flag;

    cuda_safe(cuMemHostGetDevicePointer(
        (CUdeviceptr *)&dev_worker.fence_done_flag, sge->fence_done_flag, 0));

    cudaMalloc(&sgw, sizeof(RdmaScatterGatherWorker));
    cudaMemcpyAsync(sgw, &dev_worker, sizeof(RdmaScatterGatherWorker),
                    cudaMemcpyHostToDevice);
  }

  void allocate_host_component() {

    // Create here a PThread ensemble
    // Call into Feature Init(); //Here the RDMAEngine initializes
    // Assign Call backs

    sgw = new RdmaScatterGatherWorker;
    sgw->tx_element_request_ctrs = sge->tx_element_request_ctrs;
    sgw->ack_ctrs_h = sge->ack_ctrs_h;
    sgw->tx_element_request_queue =
        (uint32_t *)sge->tx_element_request_queue_mr->addr;
    sgw->rx_element_reply_queue = sge->rx_element_reply_queue_mr->addr;
    sgw->rank = sge->rank;
  }

  // Dealloc all for now.
  void deallocate_device_component() {
    for (RdmaScatterGatherEngine *sge : sges) {
      delete sge;
    }
  }

  void deallocate_host_component() { delete sgw; }

  RdmaScatterGatherWorker *get_worker() const { return sgw; }

  RdmaScatterGatherEngine *get_engine() const { return sge; }

  ~Engine() {
    fence();
    deallocate_device_component();
    deallocate_host_component();
  }

  void fence() {
    for (RdmaScatterGatherEngine *sge : sges) {
      sge->fence();
    }
  }
};

} // namespace RACERlib

#endif