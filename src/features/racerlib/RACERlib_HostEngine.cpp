//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Contact: Jan Ciesko (jciesko@sandia.gov)
//
//@HEADER

#include <RACERlib_HostEngine.hpp>

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

// ETI this
template class HostEngine<int>;
template class HostEngine<double>;
template class HostEngine<size_t>;
// etc.

extern Transport *request_tport;

void rdma_ibv_finalize();

#define RACERLIB_SUCCESS 1

template <typename T>
void HostEngine<T>::put(void *allocation, T &value, int PE, int offset,
                        MPI_Comm comm_id){
    // do nothin'
};

template <typename T>
T HostEngine<T>::get(void *allocation, int PE, int offset, MPI_Comm comm_id) {
  return 1;
};

template <typename T>
int HostEngine<T>::start(void *target, MPI_Comm comm_id) {
  // Do nothing since we're triggering the device-side
  // kernels in the remote_paralle_for (for now)
  // The host side is lauched at init (for now)
  return RACERLIB_SUCCESS;
}

template <typename T>
int HostEngine<T>::stop(void *target, MPI_Comm comm_id) {
  // Call this at View memory deallocation (~allocation record);
  return RACERLIB_SUCCESS;
}

template <typename T>
int HostEngine<T>::flush(void *allocation, MPI_Comm comm_id) {
  // Call this on fence. We need to make sure that at sychronization points,
  // caches are empty
  return RACERLIB_SUCCESS;
}

template <typename T>
int HostEngine<T>::init(void *device_data, MPI_Comm comm_id) {
  // Init components
  allocate_host_device_component(device_data, comm_id);
  return RACERLIB_SUCCESS;
}
template <typename T>
int HostEngine<T>::finalize()  // finalize communicator instance, return
                               // RECERLIB_STATUS
{
  fence();
  deallocate_host_component();
  rdma_ibv_finalize();
  return RACERLIB_SUCCESS;
}

// Todo: template this on Feature for generic HostEngine feature support

template <typename T>
HostEngine<T>::HostEngine(void *target, MPI_Comm comm_id) {}

template <typename T>
HostEngine<T>::HostEngine() {}

template <typename T>
void HostEngine<T>::allocate_host_device_component(void *data, MPI_Comm comm) {
  // Init RDMA transport layer
  rdma_ibv_init();

  // Create the RDMAEngine (Host side)
  rdma_engine = new RDMAEngine(comm, data, sizeof(T));
  rdma_engines.insert(rdma_engine);

  // Create the device worker object and copy to device
  DeviceWorker<T> dev_worker;

  // Configure dev_worker
  dev_worker.tx_element_request_ctrs =
      rdma_engine->tx_element_request_ctrs;  // already a device buffer
  dev_worker.rx_element_reply_queue =
      rdma_engine->rx_element_reply_queue_mr->addr;  // already a device buffer
  dev_worker.tx_element_reply_queue =
      rdma_engine->tx_element_reply_queue_mr->addr;
  dev_worker.tx_element_request_trip_counts =
      rdma_engine->tx_element_request_trip_counts;
  // already a device buffer
  dev_worker.cache = rdma_engine->cache;
  dev_worker.direct_ptrs =
      rdma_engine->direct_ptrs_d;  // already a device buffer
  dev_worker.tx_element_request_queue =
      (uint32_t *)rdma_engine->tx_element_request_queue_mr->addr;
  dev_worker.ack_ctrs_d            = rdma_engine->ack_ctrs_d;
  dev_worker.tx_element_reply_ctrs = rdma_engine->tx_element_reply_ctrs;
  dev_worker.rx_element_request_queue =
      (uint32_t *)rdma_engine->rx_element_request_queue_mr->addr;
  dev_worker.num_ranks                 = rdma_engine->num_ranks;
  dev_worker.tx_element_aggregate_ctrs = rdma_engine->tx_element_aggregate_ctrs;
  dev_worker.tx_block_request_cmd_queue =
      rdma_engine->tx_block_request_cmd_queue;
  dev_worker.tx_block_reply_cmd_queue = rdma_engine->tx_block_reply_cmd_queue;
  dev_worker.tx_block_request_ctr     = rdma_engine->tx_block_request_ctr;
  dev_worker.rx_block_request_cmd_queue =
      rdma_engine->rx_block_request_cmd_queue;
  dev_worker.rx_block_request_ctr    = rdma_engine->rx_block_request_ctr;
  dev_worker.tx_element_request_ctrs = rdma_engine->tx_element_request_ctrs;
  dev_worker.my_rank                 = rdma_engine->my_rank;
  dev_worker.request_done_flag       = rdma_engine->request_done_flag;
  dev_worker.response_done_flag      = rdma_engine->response_done_flag;

  // Set host-pinned memory pointers
  cuda_safe(cuMemHostGetDevicePointer((CUdeviceptr *)&dev_worker.ack_ctrs_h,
                                      rdma_engine->ack_ctrs_h, 0));
  cuda_safe( 
      cuMemHostGetDevicePointer((CUdeviceptr *)&dev_worker.fence_done_flag,
                                rdma_engine->fence_done_flag, 0));
  // Device malloc of worker and copy
  cudaMalloc(&worker, sizeof(DeviceWorker<T>));
  cudaMemcpyAsync(worker, &dev_worker, sizeof(DeviceWorker<T>),
                  cudaMemcpyHostToDevice);
  debug("Host engine allocated. %i\n", 0);
}

// Use for host-sided memory spaces (MPI, SHMEM)
// Not used for NVSHMEM
template <typename T>
void HostEngine<T>::allocate_host_host_component() {
  // Create here a PThread ensemble
  // Call into Feature Init(); //Here the RDMAEngine initializes
  // Assign Call backs
  worker                          = new DeviceWorker<T>;
  worker->tx_element_request_ctrs = rdma_engine->tx_element_request_ctrs;
  worker->ack_ctrs_h              = rdma_engine->ack_ctrs_h;
  worker->tx_element_request_queue =
      (uint32_t *)rdma_engine->tx_element_request_queue_mr->addr;
  worker->rx_element_reply_queue = rdma_engine->rx_element_reply_queue_mr->addr;
  worker->my_rank                = rdma_engine->my_rank;

  debug("Host engine allocated. %i\n", 0);
}

// Dealloc all for now.
template <typename T>
void HostEngine<T>::deallocate_host_component() {
  for (RDMAEngine *rdma_engine : rdma_engines) {
    delete rdma_engine;
  }
  debug_2("Host engine deallocated. \n");
}

template <typename T>
DeviceWorker<T> *HostEngine<T>::get_worker() const {
  return worker;
}

template <typename T>
RDMAEngine *HostEngine<T>::get_rdma_engine() const {
  return rdma_engine;
}

template <typename T>
HostEngine<T>::~HostEngine() {}

template <typename T>
void HostEngine<T>::fence() {
  for (RDMAEngine *rdma_engine : rdma_engines) {
    rdma_engine->fence();
  }
}

}  // namespace RACERlib
}  // namespace Experimental
}  // namespace Kokkos
