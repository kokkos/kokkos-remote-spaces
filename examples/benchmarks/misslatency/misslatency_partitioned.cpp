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

#include <Kokkos_Random.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_RemoteSpaces.hpp>
#include <getopt.h>
#include <mpi.h>
#include <numeric>
#include <cmath>

using GenPool_t     = Kokkos::Random_XorShift64_Pool<>;
using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteView_t =
    Kokkos::View<double**, Kokkos::PartitionedLayoutRight, RemoteSpace_t>;

int main(int argc, char** argv) {
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

  MPI_Comm mpi_comm;

#ifdef KRS_ENABLE_NVSHMEMSPACE
  nvshmemx_init_attr_t attr;
  mpi_comm      = MPI_COMM_WORLD;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#endif

  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  uint64_t view_size          = 1e6;
  int league_size             = -1;
  int team_size               = -1;
  int repeats                 = 5;
  int remote_threads_per_warp = 1;

  option gopt[] = {
      {"help", no_argument, NULL, 'h'},
      {"team_size", required_argument, NULL, 'T'},
      {"size", required_argument, NULL, 'n'},
      {"league_size", required_argument, NULL, 'L'},
      {"repeat", required_argument, NULL, 'r'},
      {"remote_teammates", required_argument, NULL, 'R'},
  };

  int ch;
  bool help            = false;
  bool keepParsingOpts = true;
  optind               = 1;
  while ((ch = getopt_long(argc, argv, "hT:n:L:r:R:", gopt, NULL)) != -1 &&
         keepParsingOpts) {
    switch (ch) {
      case 0:
        // this set an input flag
        break;
      case 'L': league_size = std::atoi(optarg); break;
      case 'n': view_size = std::atol(optarg); break;
      case 'R': remote_threads_per_warp = std::atoi(optarg); break;
      case 'T': team_size = std::atoi(optarg); break;
      case 'r': repeats = std::atoi(optarg); break;
      case 'h': help = true; break;
    }
  }

  if (help) {
    std::cout << "latency_test <optional_args>"
                 "\n-n/--size:             The size of the array"
                 "\n-T/--team_size:        The team size (default: 32)"
                 "\n-L/--league_size:      The league size (default: "
                 "array_size/team_size)"
                 "\n-r/--repeat: 	  The number of iterations (default: 5)"
                 "\n-R/--remote_teammates: The number of threads in a warp "
                 "doing only remote accesses";
    return 0;
  }

  if (team_size == -1) {
    team_size = 32;
  }
  if (league_size == -1) {
    league_size = view_size / team_size;
  }

  int kokkos_argc    = argc - optind + 1;
  char** kokkos_argv = argv + optind - 1;
  if (kokkos_argv[0] != std::string("--")) {
    // there are no kokkos options
    kokkos_argv = argv;
    kokkos_argc = 1;
  }
  optind = 0;

  if ((uint64_t(team_size) * uint64_t(league_size)) != view_size) {
    std::cerr << "Total Size != League * Team" << std::endl;
    return 1;
  }

  if (kokkos_argc > 1) {
    std::cout << "Kokkos: Argc=" << kokkos_argc << "  Argv[]=" << kokkos_argv[1]
              << " ..." << std::endl;
  }

  Kokkos::initialize(kokkos_argc, kokkos_argv);

  if (nproc != 2) {
    std::cerr << "Benchmark should only be run with 2 procs" << std::endl;
    Kokkos::finalize();
    MPI_Finalize();
    return 1;
  }
  int partner_rank = rank ^ 1;

  using Team = Kokkos::TeamPolicy<>::member_type;
  {
    Kokkos::View<double*> target("target", view_size);
    Kokkos::TeamPolicy<> policy(league_size, team_size, 1);
    RemoteView_t remote("MyView", nproc, view_size);

    Kokkos::Timer init_timer;
    // initialize the list of indices to zero
    // randomly select gaps between misses in the index list
    Kokkos::parallel_for(
        "fill", policy, KOKKOS_LAMBDA(const Team& team) {
          int team_rank = team.league_rank();
          int offset    = team_rank * team_size;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team_size),
                               [&](int idx) {
                                 target[offset + idx]       = 0;
                                 remote(rank, offset + idx) = offset;
                               });
        });

    Kokkos::fence();

    Kokkos::Timer work_timer;
    for (int r = 0; r < repeats; ++r) {
      Kokkos::Timer timer;
      // stream through the array and essentially just copy it over
      // based on the logic above, some subset of the accesses will
      //"miss", causing a remote access
      Kokkos::parallel_for(
          "work", policy, KOKKOS_LAMBDA(const Team& team) {
            int offset = uint64_t(team.league_rank()) * team_size;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, team_size), [&](int team_idx) {
                  int dst_idx = offset + team_idx;
                  if (team_idx < remote_threads_per_warp) {
                    target[dst_idx] = 2.0 * remote(partner_rank, dst_idx);
                  }
                });
          });

      RemoteSpace_t().fence();
      double time = timer.seconds();
      if (rank == 0) {
        printf("Iteration %d: %12.8fs\n", r, time);
      }
      // let the first repeat be a warm-up
      if (r == 0) work_timer.reset();
    }
    Kokkos::fence();
    double work_time = work_timer.seconds();
    // we run league_size warp cycles
    double latency_per_warp = (work_time / league_size) / (repeats - 1) * 1e6;
    if (rank == 0) {
      printf("Latency per warp: %12.8f us\n", latency_per_warp);
    }
  }

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
