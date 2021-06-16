
#ifndef RACERLIB_RDMA_TRANSPORT
#define RACERLIB_RDMA_TRANSPORT

#include <Kokkos_Core.hpp>
#include <RDMA_Helpers.hpp>

#include <infiniband/verbs.h>
#include <mpi.h>

#include <vector>
#include <iostream>

namespace RACERlib {

struct Transport {

struct BootstrapPort {
  uint16_t lid;
  uint8_t port;
  uint32_t qp_num;
};

  ibv_pd* global_pd = nullptr;
  ibv_context* global_ctx = nullptr;
  int global_pd_ref_count = 0;

  int nproc;
  int rank;
  ibv_context* ctx;
  ibv_cq* cq;
  ibv_qp** qps;
  ibv_device* dev;
  ibv_pd* pd;
  ibv_srq* srq;

  Transport(MPI_Comm comm);
  ~Transport();
};

} // namespace RACERlib

#endif // RACERLIB_RDMA_TRANSPORT

