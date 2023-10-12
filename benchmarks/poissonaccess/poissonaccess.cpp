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
    Kokkos::View<double**, Kokkos::PartitionedLayoutLeft, RemoteSpace_t>;
using Team_t = Kokkos::TeamPolicy<>::member_type;

#define MASK (2 << 27)
constexpr uint64_t MISS_INDEX = std::numeric_limits<uint64_t>::max();

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

  int nx                   = 64;
  int league_size          = -1;
  int team_size            = -1;
  int repeats              = 5;
  double lambda            = 10;
  int remote_warp_fraction = 1;

  option gopt[] = {
      {"help", no_argument, NULL, 'h'},
      {"lambda", required_argument, NULL, 'l'},
      {"team_size", required_argument, NULL, 'T'},
      {"nx", required_argument, NULL, 'n'},
      {"league_size", required_argument, NULL, 'L'},
      {"repeat", required_argument, NULL, 'r'},
      {"fraction", required_argument, NULL, 'f'},
  };

  int ch;
  bool help            = false;
  bool keepParsingOpts = true;
  optind               = 1;
  while ((ch = getopt_long(argc, argv, "hl:T:n:L:r:f:", gopt, NULL)) != -1 &&
         keepParsingOpts) {
    switch (ch) {
      case 0:
        // this set an input flag
        break;
      case 'l': lambda = std::atof(optarg); break;
      case 'L': league_size = std::atoi(optarg); break;
      case 'n': nx = std::atol(optarg); break;
      case 'f': remote_warp_fraction = std::atoi(optarg); break;
      case 'T': team_size = std::atoi(optarg); break;
      case 'r': repeats = std::atoi(optarg); break;
      case 'h': help = true; break;
    }
  }

  if (help) {
    std::cout
        << "randomaccess-poisson <optional_args>"
           "\n-n/--nx:               The dimension of the problem (total array "
           "size is nXnX)"
           "\n-l/--lambda:           The possion parameter controlling the gap "
           "between misses."
           "\n                       Lambda is the average gap"
           "\n-T/--team_size:        The team size (default: 32)"
           "\n-L/--league_size:      The league size (default: "
           "array_size/team_size)"
           "\n-r/--repeat: 	        The number of iterations (default: 5)"
           "\n-f/--fraction:         The number of warps doing only local "
           "accesses for every"
           "\n			                  remote warp. The remote "
           "fraction is 1/f";
    return 0;
  }

  uint64_t view_size = uint64_t(nx) * nx * nx;
  if (team_size == -1) {
    team_size = 32;
  }
  if (league_size == -1) {
    league_size = view_size / team_size;
  }

  double sqrt_lambda = sqrt(lambda);

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

  {
    Kokkos::ScopeGuard guard(argc, argv);

    GenPool_t pool(5374857);
    Kokkos::View<int*> gaps("gaps", view_size);
    Kokkos::View<uint64_t*> misses("misses", view_size);
    Kokkos::View<uint64_t*> indices("indices", view_size);
    Kokkos::View<double*> target("target", view_size);
    Kokkos::View<double*> values("values", view_size);
    Kokkos::TeamPolicy<> policy(league_size, team_size, 1);
    RemoteView_t remote("MyView", nproc, view_size);

    Kokkos::Timer init_timer;
    // initialize the list of indices to zero
    // randomly select gaps between misses in the index list
    Kokkos::parallel_for(
        "fill", policy, KOKKOS_LAMBDA(const Team_t& team) {
          int rank   = team.league_rank();
          int offset = rank * team_size;
          auto gen   = pool.get_state();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team_size),
                               [&](int idx) {
                                 int k = 0;
                                 if (lambda < 30) {
                                   double L = exp(-lambda);
                                   double p = 1.;
                                   do {
                                     k++;
                                     p = p * gen.drand(1.0);
                                   } while (p > L);
                                 } else {
                                   // large lambda, approx with normal
                                   k = gen.normal(lambda, sqrt_lambda);
                                   if (k <= 0) k = 1;
                                 }
                                 gaps[offset + idx]    = k;
                                 indices[offset + idx] = 0;
                               });
          pool.free_state(gen);
        });

    // convert the list of gaps into the list of misses
    // by performing a sum prefix scan
    Kokkos::parallel_scan(
        "scan", view_size,
        KOKKOS_LAMBDA(int idx, uint64_t& sum, const bool last) {
          sum += gaps[idx];
          if (last) {
            misses[idx] = sum;
          }
        });

    // mark the remote indices with a sentinel value
    Kokkos::parallel_for(
        "mark", view_size, KOKKOS_LAMBDA(uint64_t idx) {
          uint64_t miss_idx = misses[idx];
          if (miss_idx < view_size) {
            indices[miss_idx] = MISS_INDEX;
          }
        });

    Kokkos::fence();

    // fill the index list
    // in the absence of misses, the index list is an "iota" fill
    // that performs a stream benchmark
    // for certain teams (warps) and indices, change the stream index
    // to a remote index on another proc
    Kokkos::parallel_for(
        "index", policy, KOKKOS_LAMBDA(const Team_t& team) {
          uint64_t offset    = uint64_t(team.league_rank()) * team_size;
          int warp_remainder = team.league_rank() % remote_warp_fraction;
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, team_size), [&](int team_idx) {
                uint64_t local_idx = offset + team_idx;
                // this warp has misses
                if (warp_remainder == 0 && indices[local_idx] == MISS_INDEX) {
                  int rank_stride = local_idx / nproc;
                  int dst_idx     = local_idx % view_size;
                  if (rank_stride == 0) rank_stride = 1;
                  int dst = (rank + rank_stride) % nproc;
                  uint64_t global_dst_idx =
                      uint64_t(dst) * MASK + uint64_t(dst_idx);
                  indices[local_idx] = global_dst_idx;
                } else {  // this warp does not have misses
                  indices[local_idx] = uint64_t(rank) * MASK + local_idx;
                }
                remote(rank, local_idx) = local_idx;
              });
        });

    RemoteSpace_t().fence();

    double init_time = init_timer.seconds();

    Kokkos::Timer work_timer;
    for (int r = 0; r < repeats; ++r) {
      Kokkos::Timer timer;
      // stream through the array and essentially just copy it over
      // based on the logic above, some subset of the accesses will
      // "miss", causing a remote access
      Kokkos::parallel_for(
          "work", policy, KOKKOS_LAMBDA(const Team_t& team) {
            uint64_t offset = uint64_t(team.league_rank()) * team_size;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, team_size), [&](int team_idx) {
                  uint64_t global_idx = indices[offset + team_idx];
                  target[offset + team_idx] =
                      2.0 * remote(global_idx / MASK, global_idx % MASK);
                });
          });

      RemoteSpace_t().fence();

      double time = timer.seconds();
      if (rank == 0) {
        printf("Iteration %d: %12.8fs\n", r, time);
      }
    }

    double work_time = work_timer.seconds();
    // each op reads/writes 2 doubles and 1 64-bit int
    double GB =
        repeats * view_size * (2 * sizeof(double) + sizeof(uint64_t)) / 1e9;
    double bw = GB / work_time;
    if (rank == 0) {
      printf("Observed BW: %18.8f GB/s\n", bw);
    }
  }

  MPI_Finalize();
  return 0;
}
