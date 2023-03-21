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
#include <unistd.h>

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

// ETI this
template class HostEngine<int>;
template class HostEngine<double>;
template class HostEngine<size_t>;
// etc.

template <typename T>
HostEngine<T>::HostEngine(void *target) {}

template <typename T>
HostEngine<T>::HostEngine() {}

template <typename T>
void HostEngine<T>::deallocate_device_component(void *data) {
  cuda_safe(cuMemFree((CUdeviceptr)data));
}

static void memset_device(void *buf, int value, size_t size) {
#ifdef KOKKOS_ENABLE_CUDA
  cudaMemsetAsync(buf, value, size);
#else
  ::memset(buf, value, size);
#endif
}

static size_t aligned_size(size_t alignment, size_t size) {
  size_t npages = size / alignment;
  if (size % alignment) npages++;
  return npages * alignment;
}

static void *allocate_host_pinned(size_t size, size_t &real_size) {
  size_t pagesize = (size_t)sysconf(_SC_PAGE_SIZE);
  real_size       = aligned_size(pagesize, size);
  void *addr;
#ifdef KOKKOS_ENABLE_CUDA
  cuda_safe(cuMemAllocHost(&addr, real_size));
#else
  posix_memalign(&addr, pagesize, real_size);
#endif
  return addr;
}

static void *allocate_device(size_t size, size_t &real_size) {
#ifdef KOKKOS_ENABLE_CUDA
  size_t pagesize = (size_t)sysconf(_SC_PAGE_SIZE);
  real_size       = aligned_size(pagesize, size);
  void *addr;
  cuda_safe(cuMemAlloc((CUdeviceptr *)&addr, real_size));
  return addr;
#else
  // if there is no CUDA, just send back host pinned
  return allocate_host_pinned(size, real_size);
#endif
}

static void free_device(void *buf, size_t size) {
  // cuda_safe(cuMemFree((CUdeviceptr)buf));
}

#define NUM_RECV_WRS 16

template <typename T>
void HostEngine<T>::allocate_device_component(void *data, MPI_Comm comm) {
  // Create the device worker object and copy to device
  DeviceWorker<T> dev_worker;
  size_t ignore_actual_size;
  uint32_t queue_size = QUEUE_SIZE;

  MPI_Comm_size(comm, &num_ranks);
  MPI_Comm_rank(comm, &my_rank);
  size_t elem_size             = sizeof(T);
  size_t request_counters_size = num_ranks * sizeof(uint64_t);
  size_t reply_size            = elem_size * num_ranks * queue_size;
  size_t tx_element_request_trip_counts_size =
      queue_size * num_ranks * sizeof(uint32_t);
  size_t ack_ctrs_d_size = num_ranks * sizeof(uint64_t);

  // Configure dev_worker
  dev_worker.tx_element_request_ctrs =
      (uint64_t *)allocate_device(request_counters_size, ignore_actual_size);
  dev_worker.rx_element_reply_queue =
      (uint64_t *)allocate_device(reply_size, ignore_actual_size);
  dev_worker.tx_element_reply_queue =
      (uint64_t *)allocate_device(reply_size, ignore_actual_size);
  dev_worker.tx_element_request_trip_counts = (uint32_t *)allocate_device(
      tx_element_request_trip_counts_size, ignore_actual_size);
  dev_worker.ack_ctrs_d =
      (uint64_t *)allocate_device(ack_ctrs_d_size, ignore_actual_size);
  dev_worker.tx_element_reply_ctrs =
      (uint64_t *)allocate_device(ack_ctrs_d_size, ignore_actual_size);
  dev_worker.rx_element_request_queue =
      NULL;  //(uint32_t *)sge->rx_element_request_queue_mr->addr;
  dev_worker.tx_element_aggregate_ctrs =
      (uint64_t *)allocate_device(ack_ctrs_d_size, ignore_actual_size);
  dev_worker.tx_element_request_ctrs =
      (uint64_t *)allocate_device(ack_ctrs_d_size, ignore_actual_size);
  dev_worker.request_done_flag =
      (unsigned *)allocate_device(sizeof(unsigned) * 2, ignore_actual_size);
  memset_device(dev_worker.tx_element_request_ctrs, 0, request_counters_size);
  memset_device(dev_worker.rx_element_reply_queue, 0, reply_size);
  memset_device(dev_worker.tx_element_reply_queue, 0, reply_size);
  memset_device(dev_worker.tx_element_request_trip_counts, 0,
                tx_element_request_trip_counts_size);
  memset_device(dev_worker.ack_ctrs_d, 0, ack_ctrs_d_size);
  memset_device(dev_worker.tx_element_reply_ctrs, 0, ack_ctrs_d_size);
  memset_device(dev_worker.tx_element_aggregate_ctrs, 0, ack_ctrs_d_size);
  memset_device(dev_worker.tx_element_request_ctrs, 0, ack_ctrs_d_size);
  memset_device(dev_worker.request_done_flag, 0, 2 * sizeof(unsigned));

  dev_worker.num_ranks            = num_ranks;
  dev_worker.my_rank              = my_rank;
  dev_worker.tx_block_request_ctr = 0;
  dev_worker.rx_block_request_ctr = 0;
  dev_worker.response_done_flag   = dev_worker.request_done_flag + 1;

  cudaMalloc(&worker, sizeof(DeviceWorker<T>));
  cudaMemcpyAsync(worker, &dev_worker, sizeof(DeviceWorker<T>),
                  cudaMemcpyHostToDevice);
  debug("RACERlib2 device queues allocated. %i\n", 0);
}

template <typename T>
int HostEngine<T>::init(void *device_data, MPI_Comm comm) {
  // Init components
  allocate_device_component(device_data, comm);
  return RACERLIB_SUCCESS;
}
template <typename T>
int HostEngine<T>::finalize()  // finalize instance, return
                               // RECERLIB_STATUS
{
  fence();
  deallocate_device_component((void *)worker);
  return RACERLIB_SUCCESS;
}

template <typename T>
DeviceWorker<T> *HostEngine<T>::get_worker() const {
  return worker;
}

template <typename T>
HostEngine<T>::~HostEngine() {
  /*TBD*/
}

template <typename T>
void HostEngine<T>::fence() {
  /*TBD*/
}

}  // namespace RACERlib
}  // namespace Experimental
}  // namespace Kokkos
