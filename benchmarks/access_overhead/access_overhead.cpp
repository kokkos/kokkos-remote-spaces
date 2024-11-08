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
using RemoteView_t  = Kokkos::View<double *, RemoteSpace_t>;
using PlainView_t   = Kokkos::View<double *, Kokkos::LayoutLeft>;
using UnmanagedView_t =
    Kokkos::View<double *, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using HostView_t = typename RemoteView_t::HostMirror;
struct InitTag {};
struct UpdateTag {};
struct CheckTag {};
using policy_init_t   = Kokkos::RangePolicy<InitTag, size_t>;
using policy_update_t = Kokkos::RangePolicy<UpdateTag, size_t>;
using policy_check_t  = Kokkos::RangePolicy<CheckTag, size_t>;
#define default_N 134217728
#define default_iters 3

std::string modes[3] = {"Kokkos::View", "Kokkos::RemoteView",
                        "Kokkos::LocalProxyView"};

struct Args_t {
  int mode  = 0;
  int N     = default_N;
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
bool read_args(int argc, char *argv[], Args_t &args) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0) {
      print_help();
      return false;
    }
  }

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-N") == 0) args.N = atoi(argv[i + 1]);
    if (strcmp(argv[i], "-I") == 0) args.iters = atoi(argv[i + 1]);
    if (strcmp(argv[i], "-M") == 0) args.mode = atoi(argv[i + 1]);
  }
  return true;
}

template <typename ViewType_t, typename Enable = void>
struct Access;

template <typename ViewType_t>
struct Access<ViewType_t, typename std::enable_if_t<!std::is_same<
                              ViewType_t, UnmanagedView_t>::value>> {
  size_t N;  /* size of vector */
  int iters; /* number of iterations */
  int mode;  /* View type */

  ViewType_t v;

  Access(Args_t args)
      : N(args.N),
        iters(args.iters),
        v(std::string(typeid(v).name()), args.N),
        mode(args.mode){};

  KOKKOS_FUNCTION
  void operator()(const InitTag &, const size_t i) const { v(i) = 0; }

  KOKKOS_FUNCTION
  void operator()(const UpdateTag &, const size_t i) const { v(i) += 1; }

  KOKKOS_FUNCTION
  void operator()(const CheckTag &, const size_t i) const {
    assert(v(i) == iters * 1.0);
  }

  // run copy benchmark
  void run() {
    Kokkos::Timer timer;
    double time_a, time_b;
    time_a = time_b = 0;
    double time     = 0;

    Kokkos::parallel_for("access_overhead-init", policy_init_t(0, N), *this);
    Kokkos::fence();
#ifdef KRS_ENABLE_NVSHMEMSPACE
    //nvshmem_barrier_all();  // Not sure why this impacts perf
#endif

    time_a = timer.seconds();
    for (int i = 0; i < iters; i++) {
      Kokkos::parallel_for("access_overhead", policy_update_t(0, N), *this);
      Kokkos::fence();
    }
    time_b = timer.seconds();
    time += time_b - time_a;

#ifdef CHECK_FOR_CORRECTNESS
    Kokkos::parallel_for("access_overhead-check", policy_check_t(0, N), *this);
    Kokkos::fence();
#endif

    double gups = 1e-9 * ((N * iters) / time);
    double size = N * sizeof(double) / 1024.0 / 1024.0;
    double bw   = gups * sizeof(double);
    printf("access_overhead,%s,%lu,%lf,%lu,%lf,%lf,%lf\n", modes[mode].c_str(),
           N, size, iters, time, gups, bw);
  }
};

template <typename ViewType_t>
struct Access<ViewType_t, typename std::enable_if_t<std::is_same<
                              ViewType_t, UnmanagedView_t>::value>> {
  size_t N;  /* size of vector */
  int iters; /* number of iterations */
  int mode;  /* View type */

  UnmanagedView_t v;
  RemoteView_t rv;

  Access(Args_t args)
      : N(args.N),
        iters(args.iters),
        rv(std::string(typeid(v).name()), args.N),
        mode(args.mode) {
    v = ViewType_t(rv.data(), N);
  };

  KOKKOS_FUNCTION
  void operator()(const InitTag &, const size_t i) const { v(i) = 0; }

  KOKKOS_FUNCTION
  void operator()(const UpdateTag &, const size_t i) const { v(i) += 1; }

  KOKKOS_FUNCTION
  void operator()(const CheckTag &, const size_t i) const {
    assert(v(i) == iters * 1.0);
  }

  // run copy benchmark
  void run() {
    Kokkos::Timer timer;
    double time_a, time_b;
    time_a = time_b = 0;
    double time     = 0;

    Kokkos::parallel_for("access_overhead-init", policy_init_t(0, N), *this);
    Kokkos::fence();

    time_a = timer.seconds();
    for (int i = 0; i < iters; i++) {
      Kokkos::parallel_for("access_overhead", policy_update_t(0, N), *this);
      Kokkos::fence();
    }
    time_b = timer.seconds();
    time += time_b - time_a;

#ifdef CHECK_FOR_CORRECTNESS
    Kokkos::parallel_for("access_overhead-check", policy_check_t(0, N), *this);
    Kokkos::fence();
#endif

    double gups = 1e-9 * ((N * iters) / time);
    double size = N * sizeof(double) / 1024.0 / 1024.0;
    double bw   = gups * sizeof(double);
    printf("access_overhead,%s,%lu,%lf,%lu,%lf,%lf,%lf\n", modes[mode].c_str(),
           N, size, iters, time, gups, bw);
  }
};

int main(int argc, char *argv[]) {
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
      Access<PlainView_t> s(args);
      s.run();
    } else if (args.mode == 1) {
      Access<RemoteView_t> s(args);
      s.run();
    } else if (args.mode == 2) {
      Access<UnmanagedView_t> s(args);
      s.run();
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
