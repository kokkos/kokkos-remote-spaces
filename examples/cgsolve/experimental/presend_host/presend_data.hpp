// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER

/*
  Adapted from the Mantevo Miniapp Suite.
  https://mantevo.github.io/pdfs/MantevoOverview.pdf
*/

#ifndef CGSOLVE_PRESEND_HPP
#define CGSOLVE_PRESEND_HPP

#include <Kokkos_Core.hpp>
#include <mpi.h>

namespace Impl {

template <class SType, class DType, class DDType, class IType>
struct SendReceiveLists {
  SType sendList;
  SType recvList;
  SType sendListSize;
  SType recvListSize;
  SType sendListSizeScan;
  SType recvListSizeScan;
  IType sendListSizeTotal;
  IType recvListSizeTotal;
  DType sendData;
  DType recvData;
  DDType x_new;

  SendReceiveLists(SType sendList_, SType recvList_, SType sendListSize_,
                   SType recvListSize_, SType sendListSizeScan_,
                   SType recvListSizeScan_, IType sendListSizeTotal_,
                   IType recvListSizeTotal_, DType sendData_, DType recvData_,
                   DDType x_new_)
      : sendList(sendList_), recvList(recvList_), sendListSize(sendListSize_),
        recvListSize(recvListSize_), sendListSizeScan(sendListSizeScan_),
        recvListSizeScan(recvListSizeScan_),
        sendListSizeTotal(sendListSizeTotal_),
        recvListSizeTotal(recvListSizeTotal_), sendData(sendData_),
        recvData(recvData_), x_new(x_new_) {}
};

using SendRecvLists =
    Impl::SendReceiveLists<Kokkos::View<int64_t *, Kokkos::HostSpace>,
                           Kokkos::View<double *, Kokkos::HostSpace>,
                           Kokkos::View<double *>, int64_t>;

template <class AType, class XType>
Kokkos::View<double *> presend_exchange_data(AType &A, XType &x,
                                             SendRecvLists &srl) {
  int myRank, numRanks;
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  Kokkos::View<double *, Kokkos::HostSpace> x_host("X", x.size());
  Kokkos::View<int64_t *, Kokkos::HostSpace> counter("counter", numRanks);

  Kokkos::deep_copy(counter, 0);

  // Pack send-data
  Kokkos::deep_copy(x_host, x);
  for (int i = 0; i < numRanks; ++i) {
    if (i == myRank)
      continue;
    for (int j = 0; j < srl.recvListSize(i); ++j) {
      int64_t offset = srl.recvListSizeScan(i) + counter(i)++;
      assert(srl.recvList(offset) < x_host.size());
      srl.sendData(offset) = x_host(srl.recvList(offset));
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Exchange data
  MPI_Request requests[(numRanks - 1) * 2];
  int req_count = 0;

  for (int i = 0; i < numRanks; ++i) {
    if (i == myRank)
      continue;
    MPI_Irecv(&srl.recvData(srl.sendListSizeScan(i)), srl.sendListSize(i),
              MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &requests[req_count++]);
    MPI_Isend(&srl.sendData(srl.recvListSizeScan(i)), srl.recvListSize(i),
              MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &requests[req_count++]);
  }

  MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
  MPI_Barrier(MPI_COMM_WORLD);

  // Add local contribution (x) to packed data format
  for (int i = 0; i < srl.recvListSize(myRank); ++i) {
    int64_t offset = srl.recvListSizeScan(myRank) + counter(myRank)++;
    assert(offset < srl.recvList.size());
    assert(srl.recvList(offset) < x_host.size());
    srl.recvData(offset) = x_host(srl.recvList(offset));
  }

  Kokkos::deep_copy(srl.x_new, srl.recvData);
  return srl.x_new;
}

template <class AType>
SendRecvLists presend_assemble_indices(AType &h_A, int64_t ncols) {
  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  Kokkos::View<int64_t *, Kokkos::HostSpace> sendListSize("sendListSize",
                                                          numRanks);
  Kokkos::View<int64_t *, Kokkos::HostSpace> recvListSize("recvListSize",
                                                          numRanks);

  Kokkos::deep_copy(sendListSize, 0);

  // Compute contribution sizes
  for (int64_t i = 0; i < h_A.num_rows(); ++i) {
    for (int64_t elem = h_A.row_ptr(i); elem < h_A.row_ptr(i + 1); ++elem) {
      int64_t idx = h_A.col_idx(elem);
      int64_t pid = idx / ncols;
      assert(sendListSize(pid) < h_A.nnz());
      assert(pid < numRanks);
      sendListSize(pid)++;
    }
  }

  // Exchange sizes (Now, all PEs send- and receive-size information)
  // requestListSize(myRank) = 0;
  MPI_Alltoall(sendListSize.data(), 1, MPI_LONG_LONG, recvListSize.data(), 1,
               MPI_LONG_LONG, MPI_COMM_WORLD);

  // Reduce (count total to-send-contributions and to-receive-contributions)
  int64_t sendListSizeTotal = 0;
  int64_t recvListSizeTotal = 0;
  for (int64_t i = 0; i < numRanks; ++i) {
    sendListSizeTotal += sendListSize(i);
    recvListSizeTotal += recvListSize(i);
  }

  // Scan (find offsets, used later)
  Kokkos::View<int64_t *, Kokkos::HostSpace> sendListSizeScan(
      "sendListSizeScan", numRanks);
  Kokkos::View<int64_t *, Kokkos::HostSpace> recvListSizeScan(
      "recvListSizeScan", numRanks);
  Kokkos::deep_copy(sendListSizeScan, 0);
  Kokkos::deep_copy(recvListSizeScan, 0);

  for (int64_t i = 1; i < numRanks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      sendListSizeScan(i) += sendListSize(j);
      recvListSizeScan(i) += recvListSize(j);
    }
  }

  // Assemble send-list
  Kokkos::View<int64_t *, Kokkos::HostSpace> sendList("sendList",
                                                      sendListSizeTotal);
  Kokkos::View<int64_t *, Kokkos::HostSpace> recvList("recvList",
                                                      recvListSizeTotal);
  Kokkos::deep_copy(sendList, 0);
  Kokkos::deep_copy(recvList, 0);

  Kokkos::View<int64_t *, Kokkos::HostSpace> counter("counter", numRanks);
  Kokkos::deep_copy(counter, 0);

  // Assemble send list
  for (int64_t i = 0; i < h_A.num_rows(); ++i) {
    for (int64_t elem = h_A.row_ptr(i); elem < h_A.row_ptr(i + 1); ++elem) {
      int64_t idx = h_A.col_idx(elem);
      int64_t pid = idx / ncols;
      int64_t offset = idx % ncols;
      assert(counter(pid) < sendListSize(pid));
      assert(pid < numRanks);
      sendList(sendListSizeScan(pid) + counter(pid)++) = offset;
    }
  }

  // Exchange send lists
  MPI_Request requests[(numRanks - 1) * 2];
  int req_count = 0;

  for (int i = 0; i < numRanks; ++i) {
    if (i == myRank)
      continue;
    MPI_Irecv(&recvList(recvListSizeScan(i)), recvListSize(i), MPI_LONG_LONG, i,
              1, MPI_COMM_WORLD, &requests[req_count++]);
    MPI_Isend(&sendList(sendListSizeScan(i)), sendListSize(i), MPI_LONG_LONG, i,
              1, MPI_COMM_WORLD, &requests[req_count++]);
  }

  MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
  MPI_Barrier(MPI_COMM_WORLD);

  // Alocate a View for send- and receive-data
  Kokkos::View<double *, Kokkos::HostSpace> sendData("recvData",
                                                     sendListSizeTotal);
  Kokkos::View<double *, Kokkos::HostSpace> recvData("recvData",
                                                     recvListSizeTotal);
  Kokkos::View<double *> x_new("recvData", recvListSizeTotal);

  return SendRecvLists(sendList, recvList, sendListSize, recvListSize,
                       sendListSizeScan, recvListSizeScan, sendListSizeTotal,
                       recvListSizeTotal, sendData, recvData, x_new);
}

} // namespace Impl

#endif