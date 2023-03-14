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

#include <RACERlib_HostInterface.hpp>

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

// ETI this
template class Engine<int>;
template class Engine<double>;
template class Engine<size_t>;
// etc.

template <typename T>
Engine<T>::Engine(void *target) {}

template <typename T>
Engine<T>::Engine() {}

template <typename T>
void Engine<T>::deallocate_device_component(void *data) {
  cuda_safe(cuMemFree((CUdeviceptr)data));
}

template <typename T>
void Engine<T>::allocate_device_component(void *data, MPI_Comm comm) {
  // Create the device worker object and copy to device
  RdmaScatterGatherWorker<T> dev_worker;

  MPI_Comm_size(comm, &num_ranks);
  MPI_Comm_rank(comm, &my_rank);

  // Configure dev_worker

  /*dev_worker.tx_element_request_ctrs = (uint64_t *)allocate_device(
      num_ranks * sizeof(uint64_t), ignore_actual_size);
  memset_device(dev_worker.tx_element_request_ctrs, 0, num_ranks *
  sizeof(uint64_t));*/

  /*
dev_worker.rx_element_reply_queue =
    sge->rx_element_reply_queue_mr->addr;  // already a device buffer

dev_worker.tx_element_reply_queue = sge->tx_element_reply_queue_mr->addr;
dev_worker.tx_element_request_trip_counts =
    sge->tx_element_request_trip_counts;
// already a device buffer
dev_worker.cache       = sge->cache;
dev_worker.direct_ptrs = sge->direct_ptrs_d;  // already a device buffer
dev_worker.tx_element_request_queue =
    (uint32_t *)sge->tx_element_request_queue_mr->addr;
dev_worker.ack_ctrs_d            = sge->ack_ctrs_d;
dev_worker.tx_element_reply_ctrs = sge->tx_element_reply_ctrs;
dev_worker.rx_element_request_queue =
    (uint32_t *)sge->rx_element_request_queue_mr->addr;
dev_worker.num_ranks                  = sge->num_ranks;
dev_worker.tx_element_aggregate_ctrs  = sge->tx_element_aggregate_ctrs;
dev_worker.tx_block_request_cmd_queue = sge->tx_block_request_cmd_queue;
dev_worker.tx_block_reply_cmd_queue   = sge->tx_block_reply_cmd_queue;
dev_worker.tx_block_request_ctr       = sge->tx_block_request_ctr;
dev_worker.rx_block_request_cmd_queue = sge->rx_block_request_cmd_queue;
dev_worker.rx_block_request_ctr       = sge->rx_block_request_ctr;
dev_worker.tx_element_request_ctrs    = sge->tx_element_request_ctrs;
dev_worker.my_rank                    = sge->my_rank;
dev_worker.request_done_flag          = sge->request_done_flag;
dev_worker.response_done_flag         = sge->response_done_flag;*/

  // Set host-pinned memory pointers

  // Device malloc of worker and copy
  cudaMalloc(&sgw, sizeof(RdmaScatterGatherWorker<T>));
  cudaMemcpyAsync(sgw, &dev_worker, sizeof(RdmaScatterGatherWorker<T>),
                  cudaMemcpyHostToDevice);
  debug("RACERlib2 device queues allocated. %i\n", 0);
}

template <typename T>
int Engine<T>::init(void *device_data, MPI_Comm comm) {
  // Init components
  allocate_device_component(device_data, comm);
  return RACERLIB_SUCCESS;
}
template <typename T>
int Engine<T>::finalize()  // finalize instance, return
                           // RECERLIB_STATUS
{
  fence();
  deallocate_device_component((void *)sgw);

  /*
    MPI_Barrier(comm);

    void *ignore;

    stop_running();
    pthread_join(request_thread, &ignore);
    pthread_join(ack_thread, &ignore);
    pthread_join(response_thread, &ignore);

    free_device_rdma_memory(rx_element_reply_queue_mr);
    free_device_rdma_memory(tx_element_reply_queue_mr);
    free_host_rdma_memory(rx_remote_windows_mr);
    free_host_rdma_memory(tx_remote_windows_mr);
    free_host_rdma_memory(all_request_mr);

    free_host_pinned(ack_ctrs_h, num_ranks * sizeof(uint64_t));
    free_device_rdma_memory(tx_element_request_queue_mr);
    free_device(tx_element_request_sent_ctrs, num_ranks * sizeof(uint64_t));
    free_device(tx_element_aggregate_ctrs, num_ranks * sizeof(uint64_t));

    free_device(ack_ctrs_d, num_ranks * sizeof(uint64_t));
    free_device(cache.flags, cache.cache_size);

    debug("Shutting down RdmaScatterGatherEngine: %i", 0);

    delete[] tx_element_request_acked_ctrs;
  */

  return RACERLIB_SUCCESS;
}

template <typename T>
RdmaScatterGatherWorker<T> *Engine<T>::get_worker() const {
  return sgw;
}

template <typename T>
Engine<T>::~Engine() {
  /*TBD*/
}

template <typename T>
void Engine<T>::fence() {
  /*TBD*/
}

}  // namespace RACERlib
}  // namespace Experimental
}  // namespace Kokkos
