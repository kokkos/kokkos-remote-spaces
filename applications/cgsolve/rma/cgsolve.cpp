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

/*
  Adapted from the Mantevo Miniapp Suite.
  https://mantevo.github.io/pdfs/MantevoOverview.pdf
*/

#define USE_GLOBAL_LAYOUT
//#define USE_EXPLICIT_INDEXING

#include <Kokkos_RemoteSpaces.hpp>
#include <generate_matrix.hpp>
#include <mpi.h>

#include <comm.hpp>

#include <iostream>

typedef Kokkos::Experimental::DefaultRemoteMemorySpace RemoteMemSpace_t;
#ifdef USE_GLOBAL_LAYOUT
typedef Kokkos::View<double *, RemoteMemSpace_t> RemoteView_t;
#else
typedef Kokkos::View<double **, Kokkos::PartitionedLayoutLeft, RemoteMemSpace_t>
    RemoteView_t;
#endif

template <class YType, class AType, class XType>
void spmv(YType y, AType A, XType x) {
  int numRanks = 1, rank = 0;
  int64_t nrows     = y.extent(0);
  int vector_length = 8;
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef KOKKOS_ENABLE_CUDA
  int rows_per_team = 16;
  int team_size     = 16;
#else
  int rows_per_team = 512;
  int team_size     = 1;
#endif

  auto policy =
      require(Kokkos::TeamPolicy<>((nrows + rows_per_team - 1) / rows_per_team,
                                   team_size, vector_length),
              Kokkos::Experimental::WorkItemProperty::HintHeavyWeight);
  Kokkos::parallel_for(
      "spmv", policy,
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {
        const int64_t first_row = team.league_rank() * rows_per_team;
        const int64_t last_row  = first_row + rows_per_team < nrows
                                     ? first_row + rows_per_team
                                     : nrows;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first_row, last_row),
                             [&](const int64_t row) {
                               const int64_t row_start = A.row_ptr(row);
                               const int64_t row_length =
                                   A.row_ptr(row + 1) - row_start;
                               double y_row = 0.0;
                               Kokkos::parallel_reduce(
                                   Kokkos::ThreadVectorRange(team, row_length),
                                   [=](const int64_t i, double &sum) {
                                     int64_t current_row = row_start + i;
                                     int64_t idx = A.col_idx(current_row);

#ifdef USE_GLOBAL_LAYOUT
                                     sum += A.values(current_row) * x(idx);
#else
#ifdef USE_EXPLICIT_INDEXING
    // Enable for faster pid and offset calculation. May result in unfair comparison
    int64_t pid = idx / MASK;
    int64_t offset = idx % MASK;
#else
    int64_t pid = idx / nrows;
    int64_t offset = idx % nrows;
#endif
    sum += A.values(current_row) * x(pid, offset);
#endif
                                   },
                                   y_row);
                               y(row) = y_row;
                             });
      });

  Kokkos::fence();
  RemoteMemSpace_t().fence();
}

template <class YType, class XType>
double dot(YType y, XType x) {
  double result = 0.0;
  int64_t n     = y.extent(0);
  Kokkos::parallel_reduce(
      "DOT", n,
      KOKKOS_LAMBDA(const int64_t &i, double &lsum) { lsum += y(i) * x(i); },
      result);
  return result;
}

template <class ZType, class YType, class XType>
void axpby(ZType z, double alpha, XType x, double beta, YType y) {
  int64_t n = z.extent(0);
  Kokkos::parallel_for(
      "AXPBY", n,
      KOKKOS_LAMBDA(const int &i) { z(i) = alpha * x(i) + beta * y(i); });
  Kokkos::fence();
}

template <class VType>
void print_vector(int label, VType v) {
  std::cout << "\n\nPRINT " << v.label() << std::endl << std::endl;

  int myRank = 0;
  Kokkos::parallel_for(
      v.extent(0), KOKKOS_LAMBDA(const int i) {
        printf("%i %i %i %lf\n", label, myRank, i, v(i));
      });
  Kokkos::fence();
  std::cout << "\n\nPRINT DONE " << v.label() << std::endl << std::endl;
}

template <class VType, class AType, class PType>
int cg_solve(VType y, AType A, VType b, PType p_global, int max_iter,
             double tolerance) {
  int myproc = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
  int num_iters = 0;

  double normr     = 0;
  double rtrans    = 0;
  double oldrtrans = 0;

  int64_t print_freq = max_iter / 10;
  if (print_freq > 50) print_freq = 50;
  if (print_freq < 1) print_freq = 1;
  VType x("x", b.extent(0));
  VType r("r", x.extent(0));
  VType p(p_global.data(), x.extent(0));  // Globally accessible data
  VType Ap("Ap", x.extent(0));

  double one  = 1.0;
  double zero = 0.0;

  axpby(p, one, x, zero, x);
  Kokkos::fence();
  RemoteMemSpace_t().fence();
  spmv(Ap, A, p_global);
  axpby(r, one, b, -one, Ap);

  rtrans = dot(r, r);
  MPI_Allreduce(MPI_IN_PLACE, &rtrans, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  normr = std::sqrt(rtrans);

#ifdef KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
  if (true) {
    if (myproc == 0) {
      std::cout << "Initial Residual = " << normr << std::endl;
    }
  }
#endif

  double brkdown_tol = std::numeric_limits<double>::epsilon();

  for (int64_t k = 1; k <= max_iter && normr > tolerance; ++k) {
    if (k == 1) {
      axpby(p, one, r, zero, r);
    } else {
      oldrtrans = rtrans;
      rtrans    = dot(r, r);
      MPI_Allreduce(MPI_IN_PLACE, &rtrans, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      double beta = rtrans / oldrtrans;
      axpby(p, one, r, beta, p);
    }

    normr = std::sqrt(rtrans);

#ifdef KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
    if (true) {
      if (myproc == 0 && (k % print_freq == 0 || k == max_iter)) {
        std::cout << "Iteration = " << k << "   Residual = " << normr
                  << std::endl;
      }
    }
#endif

    double alpha    = 0;
    double p_ap_dot = 0;
    spmv(Ap, A, p_global);
    p_ap_dot = dot(Ap, p);

    MPI_Allreduce(MPI_IN_PLACE, &p_ap_dot, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    if (p_ap_dot < brkdown_tol) {
      if (p_ap_dot < 0) {
        std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!"
                  << std::endl;
        num_iters = k;
        return num_iters;
      } else
        brkdown_tol = 0.1 * p_ap_dot;
    }
    alpha = rtrans / p_ap_dot;

    axpby(x, one, x, alpha, p);
    axpby(r, one, r, -alpha, Ap);
    num_iters = k;
  }
  return num_iters;
}

int main(int argc, char *argv[]) {
  comm_init(argc, argv);
  int myRank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  Kokkos::initialize(argc, argv);
  {
    int N                            = argc > 1 ? atoi(argv[1]) : 100;
    int max_iter                     = argc > 2 ? atoi(argv[2]) : 200;
    double tolerance                 = 1e-7;
    CrsMatrix<Kokkos::HostSpace> h_A = Impl::generate_miniFE_matrix(N);
    Kokkos::View<double *, Kokkos::HostSpace> h_x =
        Impl::generate_miniFE_vector(N);

    Kokkos::View<int64_t *> row_ptr("row_ptr", h_A.row_ptr.extent(0));
    Kokkos::View<int64_t *> col_idx("col_idx", h_A.col_idx.extent(0));
    Kokkos::View<double *> values("values", h_A.values.extent(0));
    CrsMatrix<Kokkos::DefaultExecutionSpace::memory_space> A(
        row_ptr, col_idx, values, h_A.num_cols());
    Kokkos::View<double *> x("X", h_x.extent(0));
    Kokkos::View<double *> y("Y", A.num_rows());

    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(A.row_ptr, h_A.row_ptr);
    Kokkos::deep_copy(A.col_idx, h_A.col_idx);
    Kokkos::deep_copy(A.values, h_A.values);

#ifdef USE_GLOBAL_LAYOUT
    // Allocate global size (runtime splits into chunks)
    RemoteView_t p = RemoteView_t("MyView", numRanks * h_x.extent(0));
#else
    RemoteView_t p = RemoteView_t("MyView", h_x.extent(0));
#endif
    Kokkos::Timer timer;
    int num_iters = cg_solve(y, A, x, p, max_iter, tolerance);
    double time   = timer.seconds();

    // Compute Bytes and Flops
    double spmv_bytes = A.num_rows() * sizeof(int64_t) +  // A.row_ptr
                        A.nnz() * sizeof(int64_t) +       // A.col_idx
                        A.nnz() * sizeof(double) +        // A.values
                        A.nnz() * sizeof(double) +        // input vector
                        A.num_rows() * sizeof(double);    // output vector
    double dot_bytes   = A.num_rows() * sizeof(double) * 2;
    double axpby_bytes = A.num_rows() * sizeof(double) * 3;

    double spmv_flops  = A.nnz() * 2;
    double dot_flops   = A.num_rows() * 2;
    double axpby_flops = A.num_rows() * 3;

    int spmv_calls  = 1 + num_iters;
    int dot_calls   = num_iters;
    int axpby_calls = 2 + num_iters * 3;

    double total_flops = spmv_flops * spmv_calls + dot_flops * dot_calls +
                         axpby_flops * axpby_calls;

    double total_bytes = spmv_bytes * spmv_calls + dot_bytes * dot_calls +
                         axpby_bytes * axpby_calls;

    MPI_Allreduce(MPI_IN_PLACE, &total_flops, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &total_bytes, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    double GFlops = 1e-9 * total_flops / time;
    double GBs    = (1.0 / 1024 / 1024 / 1024) * total_bytes / time;

    if (myRank == 0) {
#ifndef KOKKOS_REMOTE_SPACES_ENABLE_DEBUG
      printf("%i, %i, %.2e, %.6lf, %.6lf, %.6lf\n", N, num_iters, total_flops,
             time, GFlops, GBs);
#else
      printf(
          "N, num_iters, total_flops, time, GFlops, BW(GB/sec), %i, %i, %.2e, "
          "%.6lf, %.6lf, %.6lf\n",
          N, num_iters, total_flops, time, GFlops, GBs);
#endif
    }
  }

  Kokkos::finalize();
  comm_fini();
  return 0;
}