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

//ETI this
Engine<int>    __en1;
Engine<double> __en2;
Engine<size_t> __en3;

#define RACERLIB_SUCCESS 1

template <typename data_type>
void Engine<data_type>::put(void *comm_id, void *allocation, data_type &value, int PE, int offset)
{
  //do nothin'
};

template <typename data_type>
data_type Engine<data_type>::get(void *comm_id, void *allocation, int PE, int offset){
  return 1;
};

template <typename data_type>
int Engine<data_type>::start(void *comm_id, void *allocation_id) {
  // Call this at View memory allocation (allocation record)
  return RACERLIB_SUCCESS;
}

template <typename data_type>
int Engine<data_type>::stop(void *comm_id, void *allocation_id)
{
  // Call this at View memory deallocation (~allocation record);
  return RACERLIB_SUCCESS;
}

template <typename data_type>
int Engine<data_type>::flush(void *comm_id, void *allocation) {
  // Call this on fence. We need to make sure that at sychronization points,
  // caches are empty
  return RACERLIB_SUCCESS;
}

template <typename data_type>
int Engine<data_type>::init(void *comm_id) // set communicator reference, return RACERLIB_STATUS
{
  // Call this on Kokkos initialize.
  return RACERLIB_SUCCESS;
}
template <typename data_type>
int Engine<data_type>::finalize(
    void *comm_id) // finalize communicator instance, return RECERLIB_STATUS
{
  // Call this on kokkos finalize
  return RACERLIB_SUCCESS;
}

// Todo: template this on Feature for generic Engine feature support

template <typename data_type>
Engine<data_type>::Engine() { 
 
}

template <typename data_type>
void Engine<data_type>::allocate_device_component(void *p, MPI_Comm comm) {
  // Create here a persistent kernel with functor (polling)
  // Call into Feature::Worker();
  size_t header_size = 0x1;
  sge = new RdmaScatterGatherEngine(comm, sizeof(data_type));
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

template <typename data_type>
void Engine<data_type>::allocate_host_component() {

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
  template <typename data_type>
void Engine<data_type>::deallocate_device_component() {
  for (RdmaScatterGatherEngine *sge : sges) {
    delete sge;
  }
}
template <typename data_type>
  void Engine<data_type>::deallocate_host_component() { delete sgw; }

template <typename data_type>
  RdmaScatterGatherWorker *Engine<data_type>::get_worker() const { return sgw; }

template <typename data_type>
  RdmaScatterGatherEngine *Engine<data_type>::get_engine() const { return sge; }

template <typename data_type>
  Engine<data_type>::~Engine() {
    fence();
    deallocate_device_component();
    deallocate_host_component();
  }


template <typename data_type>
  void Engine<data_type>::fence() {
    for (RdmaScatterGatherEngine *sge : sges) {
      sge->fence();
    }
  }

} // namespace RACERlib
} // namespace Experimental
} // namespace Kokkos
