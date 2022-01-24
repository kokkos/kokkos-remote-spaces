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

#include <RDMA_Transport.hpp>

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

// Global protection domain
ibv_pd* global_pd = nullptr;
// Global device context
ibv_context* global_ctx = nullptr;
// Ref counter
int global_pd_ref_count = 0;
// Global Transport instances
Transport *request_tport = nullptr;
Transport *response_tport = nullptr;


void rdma_ibv_init() {
  if (!request_tport) {
    request_tport = new Transport(MPI_COMM_WORLD);
  }
  if (!response_tport) {
    response_tport = new Transport(MPI_COMM_WORLD);
  }

  debug_2("Transport allocated.\n");
}

void rdma_ibv_finalize() {
  
  if (request_tport) {
    delete request_tport;
    request_tport = nullptr;
  }

  if (response_tport) {
    delete response_tport;
    response_tport = nullptr;
  }
  debug_2("Transport deallocated.\n");
}

Transport::Transport(MPI_Comm comm)
{
  MPI_Comm_size(comm, &num_ranks);
  MPI_Comm_rank(comm, &my_rank);

  int num_devices;
  ibv_device** devices = ibv_get_device_list(&num_devices);
  assert_ibv(devices);
  dev = devices[0];

  if (global_pd_ref_count == 0){
    global_ctx = ibv_open_device(dev);

    // Allocate protection domain for device context
    global_pd = ibv_alloc_pd(global_ctx);
  }
  
  ++global_pd_ref_count;
  pd = global_pd;
  ctx = global_ctx;
  assert_ibv(pd);

  std::vector<BootstrapPort> global_ports;
  std::vector<BootstrapPort> local_ports;

  constexpr int num_cqe = 2048;

  // Create completion queue for device context with num_cque entries
  cq = ibv_create_cq(ctx, num_cqe, nullptr, nullptr, 0);
  assert_ibv(cq);

  struct ibv_srq_init_attr srq_init_attr;

  memset(&srq_init_attr, 0, sizeof(srq_init_attr));

  // Set max outstanding number of work requests 
  // and number scatter elements per work request
  srq_init_attr.attr.max_wr  = 1024;
  srq_init_attr.attr.max_sge = 2;

  // Create shared receive queue for protection domain 
  srq = ibv_create_srq(pd, &srq_init_attr);
  assert_ibv(srq);

  ibv_device_attr dev_attr;
  ibv_query_device(ctx, &dev_attr);

  ibv_port_attr port_attr;
  ibv_query_port(ctx, 1, &port_attr);
  if (port_attr.state != IBV_PORT_ACTIVE || port_attr.link_layer != IBV_LINK_LAYER_INFINIBAND){
    Kokkos::abort("RDMA_Engine: Queried ibv port is not active");
  }

  ibv_qp_init_attr qp_attr;
  memset(&qp_attr, 0, sizeof(struct ibv_qp_init_attr));

  qp_attr.srq = srq;
  qp_attr.send_cq = cq;
  qp_attr.recv_cq = cq;
  qp_attr.qp_type = IBV_QPT_RC;
  qp_attr.cap.max_send_wr = 1024;
  qp_attr.cap.max_recv_wr = 1024;
  qp_attr.cap.max_send_sge = 2;
  qp_attr.cap.max_recv_sge = 2;
  qp_attr.cap.max_inline_data = 0;

  qps = new ibv_qp*[num_ranks];

  for (int pe=0; pe < num_ranks; ++pe){

    // Create queue pairs
    qps[pe] = ibv_create_qp(pd, &qp_attr);
    assert_ibv(qps[pe]);

    uint8_t max_ports = 1; //dev_attr.phys_port_cnt

    for (uint8_t p=1; p <= max_ports; ++p){
      local_ports.push_back({port_attr.lid, p, qps[pe]->qp_num});
      ibv_qp_attr attr; 
      memset(&attr, 0, sizeof(struct ibv_qp_attr));
      attr.qp_state = IBV_QPS_INIT;
      attr.pkey_index = 0;
      attr.port_num = p;
      attr.qp_access_flags =
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
      int qp_flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

      // Upddate queue pair
      ibv_safe(ibv_modify_qp(qps[pe], &attr, qp_flags));
    }
  }

  // Exchange the data out-of-band using MPI for queue pairs
  global_ports.resize(local_ports.size());

  MPI_Alltoall(local_ports.data(), sizeof(BootstrapPort), MPI_BYTE,
               global_ports.data(), sizeof(BootstrapPort), MPI_BYTE,
               comm);

  // Put all the queue pairs into a ready state
  for (int pe=0; pe < num_ranks; ++pe){

    //if (pe == my_rank) continue;

    ibv_qp_attr attr;
    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.dest_qp_num = global_ports[pe].qp_num;
    attr.rq_psn = 0;
    attr.path_mtu = IBV_MTU_1024; //port_attr.active_mtu;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 0x12;
    attr.ah_attr.dlid = global_ports[pe].lid;
    attr.ah_attr.is_global = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = global_ports[pe].port;
    int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
            IBV_QP_MIN_RNR_TIMER | IBV_QP_PATH_MTU | IBV_QP_MAX_DEST_RD_ATOMIC;
    ibv_safe(ibv_modify_qp(qps[pe], &attr, mask));

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = 0x12;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
    ibv_safe(ibv_modify_qp(qps[pe], &attr, mask));
  }
}

Transport::~Transport()
{
  for (int pe=0; pe < num_ranks; ++pe){
    if (pe != my_rank){
     ibv_destroy_qp(qps[pe]);
    }
  }

  ibv_destroy_cq(cq);
  --global_pd_ref_count;
  if (global_pd_ref_count == 0){
    ibv_dealloc_pd(global_pd);
    ibv_close_device(global_ctx);
    global_pd = nullptr;
    global_ctx = nullptr;
  }

}

} // namespace RACERlib
} // namespace Experimental
} // namespace Kokkos