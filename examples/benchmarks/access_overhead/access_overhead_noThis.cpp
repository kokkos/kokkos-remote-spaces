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

#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <assert.h>
#include <typeinfo>
#include <type_traits>
#include <string>

#define CHECK_FOR_CORRECTNESS

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteView_t  = Kokkos::View<double*, RemoteSpace_t>;
using PlainView_t   = Kokkos::View<double*>;
using UnmanagedView_t =
    Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostView_t  = typename RemoteView_t::HostMirror;
using StreamIndex = size_t;
using policy_t    = Kokkos::RangePolicy<Kokkos::IndexType<StreamIndex>>;

#define default_N 134217728
#define default_iters 3

std::string modes[3] = {"Kokkos::View", "Kokkos::RemoteView",
                        "Kokkos::LocalProxyView"};

struct Args_t {
  int mode  = 0;
  size_t N  = default_N;
  int iters = default_iters;
};

void print_help() {
  printf("Options (default):\n");
  printf("  -N IARG: (%i) num elements in the vector\n", default_N);
  printf("  -I IARG: (%i) num repititions\n", default_iters);
  printf("  -M IARG: (%i) mode (view type)\n", 0);
  printf("     modes:\n");
  printf("       0: Kokkos (Normal)  View\n");
  printf("       1: Kokkos Remote    View\n");
  printf("       2: Kokkos Unmanaged View\n");
}

// read command line args
bool read_args(int argc, char* argv[], Args_t& args) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0) {
      print_help();
      return false;
    }
  }

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-N") == 0) args.N = atoi(argv[i + 1]);
    if (strcmp(argv[i], "-I") == 0) args.iters = atoi(argv[i + 1]);
    if (strcmp(argv[i], "-M") == 0) args.mode = atol(argv[i + 1]);
  }
  return true;
}

// run copy benchmark
void run_1(Args_t& args) {
  Kokkos::Timer timer;
  double time_a, time_b;
  time_a = time_b  = 0;
  double time      = 0;
  using ViewType_t = PlainView_t;
  ViewType_t v("PlainView_t", args.N);

  size_t N  = args.N;     /* size of vector */
  int iters = args.iters; /* number of iterations */
  int mode  = args.mode;  /* View type */

  Kokkos::parallel_for(
      "access_overhead-init", policy_t({0}, {N}),
      KOKKOS_LAMBDA(const size_t i) { v(i) = 0.0; });

  Kokkos::fence();
  nvshmem_barrier_all();  // Not sure why this impacts perf

  time_a = timer.seconds();
  for (int i = 0; i < iters; i++) {
    Kokkos::parallel_for(
        "access_overhead", policy_t({0}, {N}),
        KOKKOS_LAMBDA(const size_t i) { v(i) += 1; });
    RemoteSpace_t().fence();
  }
  time_b = timer.seconds();
  time += time_b - time_a;

#ifdef CHECK_FOR_CORRECTNESS
  Kokkos::parallel_for(
      "access_overhead-check", policy_t({0}, {N}),
      KOKKOS_LAMBDA(const size_t i) { assert(v(i) == iters * 1.0); });
  Kokkos::fence();
#endif

  double gups = 1e-9 * ((N * iters) / time);
  double size = N * sizeof(double) / 1024.0 / 1024.0;
  double bw   = gups * sizeof(double);
  printf("access_overhead-noThis,%s,%lu,%lf,%lu,%lf,%lf,%lf\n",
         modes[mode].c_str(), N, size, iters, time, gups, bw);
}

// run copy benchmark
void run_2(Args_t& args) {
  Kokkos::Timer timer;
  double time_a, time_b;
  time_a = time_b  = 0;
  double time      = 0;
  using ViewType_t = RemoteView_t;
  ViewType_t v("RemoteView_t", args.N);

  size_t N  = args.N;     /* size of vector */
  int iters = args.iters; /* number of iterations */
  int mode  = args.mode;  /* View type */

  Kokkos::parallel_for(
      "access_overhead-init", policy_t({0}, {N}),
      KOKKOS_LAMBDA(const size_t i) { v(i) = 0.0; });

  Kokkos::fence();
  nvshmem_barrier_all();  // Not sure why this impacts perf

  time_a = timer.seconds();
  for (int i = 0; i < iters; i++) {
    Kokkos::parallel_for(
        "access_overhead", policy_t({0}, {N}),
        KOKKOS_LAMBDA(const size_t i) { v(i) += 1; });
    RemoteSpace_t().fence();
  }
  time_b = timer.seconds();
  time += time_b - time_a;

#ifdef CHECK_FOR_CORRECTNESS
  Kokkos::parallel_for(
      "access_overhead-check", policy_t({0}, {N}),
      KOKKOS_LAMBDA(const size_t i) { assert(v(i) == iters * 1.0); });
  Kokkos::fence();
#endif

  double gups = 1e-9 * ((N * iters) / time);
  double size = N * sizeof(double) / 1024.0 / 1024.0;
  double bw   = gups * sizeof(double);
  printf("access_overhead-noThis,%s,%lu,%lf,%lu,%lf,%lf,%lf\n",
         modes[mode].c_str(), N, size, iters, time, gups, bw);
}

// run copy benchmark
void run_3(Args_t& args) {
  Kokkos::Timer timer;
  double time_a, time_b;
  time_a = time_b = 0;
  double time     = 0;

  size_t N  = args.N;     /* size of vector */
  int iters = args.iters; /* number of iterations */
  int mode  = args.mode;  /* View type */

  RemoteView_t rv("RemoteView_t", args.N);
  UnmanagedView_t v(rv.data(), N);

  Kokkos::parallel_for(
      "access_overhead-init", policy_t({0}, {N}),
      KOKKOS_LAMBDA(const size_t i) { v(i) = 0.0; });

  Kokkos::fence();
  nvshmem_barrier_all();  // Not sure why this impacts perf

  time_a = timer.seconds();
  for (int i = 0; i < iters; i++) {
    Kokkos::parallel_for(
        "access_overhead", policy_t({0}, {N}),
        KOKKOS_LAMBDA(const size_t i) { v(i) += 1; });
    RemoteSpace_t().fence();
  }
  time_b = timer.seconds();
  time += time_b - time_a;

#ifdef CHECK_FOR_CORRECTNESS
  Kokkos::parallel_for(
      "access_overhead-check", policy_t({0}, {N}),
      KOKKOS_LAMBDA(const size_t i) { assert(v(i) == iters * 1.0); });
  Kokkos::fence();
#endif

  double gups = 1e-9 * ((N * iters) / time);
  double size = N * sizeof(double) / 1024.0 / 1024.0;
  double bw   = gups * sizeof(double);
  printf("access_overhead-noThis,%s,%lu,%lf,%lu,%lf,%lf,%lf\n",
         modes[mode].c_str(), N, size, iters, time, gups, bw);
}

int main(int argc, char* argv[]) {
  int mpi_thread_level_available;
  int mpi_thread_level_required = MPI_THREAD_MULTIPLE;

#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL
  mpi_thread_level_required = MPI_THREAD_SINGLE;
#endif

  MPI_Init_thread(&argc, &argv, mpi_thread_level_required,
                  &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);

#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_init_thread(mpi_thread_level_required, &mpi_thread_level_available);
  assert(mpi_thread_level_available >= mpi_thread_level_required);
#endif

#ifdef KRS_ENABLE_NVSHMEMSPACE
  MPI_Comm mpi_comm;
  nvshmemx_init_attr_t attr;
  mpi_comm      = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif

  Kokkos::initialize(argc, argv);

  do {
    Args_t args;
    if (!read_args(argc, argv, args)) {
      break;
    };
    if (args.mode == 0) {
      run_1(args);
    } else if (args.mode == 1) {
      run_2(args);
    } else if (args.mode == 2) {
      run_3(args);
    } else {
      printf("invalid mode selected (%d)\n", args.mode);
    }
  } while (false);

  Kokkos::finalize();
#ifdef KRS_ENABLE_SHMEMSPACE
  shmem_finalize();
#endif
#ifdef KRS_ENABLE_NVSHMEMSPACE
  nvshmem_finalize();
#endif
  MPI_Finalize();
  return 0;
}

#undef CHECK_FOR_CORRECTNESS
