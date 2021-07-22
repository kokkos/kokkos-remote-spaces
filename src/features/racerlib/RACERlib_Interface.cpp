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

#include <RACERlib_Interface.hpp>

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

// ETI this
template class Engine<int>;
template class Engine<double>;
template class Engine<size_t>;
// etc.

extern Transport *request_tport;

void rdma_ibv_finalize();

#define RACERLIB_SUCCESS 1

template <typename T>
void Engine<T>::put(void *allocation, T &value, int PE, int offset,
                    MPI_Comm comm_id){
    // do nothin'
};

template <typename T>
T Engine<T>::get(void *allocation, int PE, int offset, MPI_Comm comm_id) {
  return 1;
};

template <typename T> int Engine<T>::start(void *target, MPI_Comm comm_id) {
  // Do nothing since we're triggering the device-side
  // kernels in the remote_paralle_for (for now)
  // The host side is lauched at init (for now)
  return RACERLIB_SUCCESS;
}

template <typename T> int Engine<T>::stop(void *target, MPI_Comm comm_id) {
  // Call this at View memory deallocation (~allocation record);
  return RACERLIB_SUCCESS;
}

template <typename T> int Engine<T>::flush(void *allocation, MPI_Comm comm_id) {
  // Call this on fence. We need to make sure that at sychronization points,
  // caches are empty
  return RACERLIB_SUCCESS;
}

template <typename T> int Engine<T>::init(void *device_data, MPI_Comm comm_id) {
  // Init components
  allocate_host_device_component(device_data, comm_id);
  return RACERLIB_SUCCESS;
}
template <typename T>
int Engine<T>::finalize() // finalize communicator instance, return
                          // RECERLIB_STATUS
{
  fence();
  deallocate_host_component();
  rdma_ibv_finalize();
  return RACERLIB_SUCCESS;
}

// Todo: template this on Feature for generic Engine feature support

template <typename T> Engine<T>::Engine(void *target, MPI_Comm comm_id) {}

template <typename T> Engine<T>::Engine() {}

template <typename T>
void Engine<T>::allocate_host_device_component(void *data, MPI_Comm comm) {
  // Init RDMA transport layer
  rdma_ibv_init();

  // Create the RdmaScatterGatherEngine (Host side)
  sge = new RdmaScatterGatherEngine(comm, data, sizeof(T));
  sges.insert(sge);

  // Create the device worker object and copy to device
  RdmaScatterGatherWorker<T> dev_worker;

  // Configure dev_worker
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
  dev_worker.num_ranks = sge->num_ranks;
  dev_worker.tx_element_aggregate_ctrs = sge->tx_element_aggregate_ctrs;
  dev_worker.tx_block_request_cmd_queue = sge->tx_block_request_cmd_queue;
  dev_worker.tx_block_reply_cmd_queue = sge->tx_block_reply_cmd_queue;
  dev_worker.tx_block_request_ctr = sge->tx_block_request_ctr;
  dev_worker.rx_block_request_cmd_queue = sge->rx_block_request_cmd_queue;
  dev_worker.rx_block_request_ctr = sge->rx_block_request_ctr;
  dev_worker.tx_element_request_ctrs = sge->tx_element_request_ctrs;
  dev_worker.my_rank = sge->my_rank;
  dev_worker.request_done_flag = sge->request_done_flag;
  dev_worker.response_done_flag = sge->response_done_flag;

  // Set host-pinned memory pointers
  cuda_safe(cuMemHostGetDevicePointer((CUdeviceptr *)&dev_worker.ack_ctrs_h,
                                      sge->ack_ctrs_h, 0));
  cuda_safe(cuMemHostGetDevicePointer(
      (CUdeviceptr *)&dev_worker.fence_done_flag, sge->fence_done_flag, 0));
  // Device malloc of worker and copy
  cudaMalloc(&sgw, sizeof(RdmaScatterGatherWorker<T>));
  cudaMemcpyAsync(sgw, &dev_worker, sizeof(RdmaScatterGatherWorker<T>),
                  cudaMemcpyHostToDevice);

  debug("Host engine allocated. %i\n", 0);
}

// Use for host-sided memory spaces (MPI, SHMEM)
// Not used for NVSHMEM
template <typename T> void Engine<T>::allocate_host_host_component() {

  // Create here a PThread ensemble
  // Call into Feature Init(); //Here the RDMAEngine initializes
  // Assign Call backs
  sgw = new RdmaScatterGatherWorker<T>;
  sgw->tx_element_request_ctrs = sge->tx_element_request_ctrs;
  sgw->ack_ctrs_h = sge->ack_ctrs_h;
  sgw->tx_element_request_queue =
      (uint32_t *)sge->tx_element_request_queue_mr->addr;
  sgw->rx_element_reply_queue = sge->rx_element_reply_queue_mr->addr;
  sgw->my_rank = sge->my_rank;

  debug("Host engine allocated. %i\n", 0);
}

// Dealloc all for now.
template <typename T> void Engine<T>::deallocate_host_component() {
  for (RdmaScatterGatherEngine *sge : sges) {
    delete sge;
  }
  debug_2("Host engine deallocated. \n");
}

template <typename T>
RdmaScatterGatherWorker<T> *Engine<T>::get_worker() const {
  return sgw;
}

template <typename T> RdmaScatterGatherEngine *Engine<T>::get_engine() const {
  return sge;
}

template <typename T> Engine<T>::~Engine() {}

template <typename T> void Engine<T>::fence() {
  
  
  for (RdmaScatterGatherEngine *sge : sges) {
    sge->fence();    
  }
}

} // namespace RACERlib
} // namespace Experimental
} // namespace Kokkos
