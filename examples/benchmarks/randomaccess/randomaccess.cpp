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
#include <Kokkos_RemoteSpaces.hpp>
#include <getopt.h>
#include <limits>
#include <mpi.h>
#include <stdlib.h>

// Default values
#define SIZE 1024
#define NUM_ITER 100
#define LEAGUE_SIZE 1
#define TEAM_SIZE 32
#define VEC_LEN 1
#define ORDINAL_T int64_t
#define SIGMA 1000

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using RemoteView_t =
    Kokkos::View<ORDINAL_T **, Kokkos::PartitionedLayoutLeft, RemoteSpace_t>;
using Generator_t = Kokkos::Random_XorShift64_Pool<>;
using TeamPolicy  = Kokkos::TeamPolicy<>;

/*
  Uncomment to select between random or linear access pattern.
  Random access pattern follow either a
  normal or random distribution.
*/

KOKKOS_INLINE_FUNCTION
ORDINAL_T get(const ORDINAL_T mean, const float variance,
              Generator_t::generator_type &g) {
  return g.normal(mean, variance);
  // return g.urand64();
  // return mean;
}

int main(int argc, char *argv[]) {
  ORDINAL_T num_elems = (SIZE << 10) / sizeof(ORDINAL_T);
  ORDINAL_T num_iters = NUM_ITER;
  int sigma           = SIGMA;
  int league_size     = LEAGUE_SIZE;
  int team_size       = TEAM_SIZE;
  int vec_len         = VEC_LEN;

  option gopt[] = {
      {"help", no_argument, NULL, 'h'},
      {"size", required_argument, NULL, 's'},
      {"sigma", required_argument, NULL, 'g'},
      {"leage_size", required_argument, NULL, 'l'},
      {"team_size", required_argument, NULL, 't'},
  };

  int ch;
  bool help = false;
  while ((ch = getopt_long(argc, argv, "hs:g:l:t:", gopt, NULL)) != -1) {
    switch (ch) {
      case 0:
        // this set an input flag
        break;
      case 'h': help = true; break;
      case 's':
        num_elems = (std::atoi(optarg) << 10) / sizeof(ORDINAL_T);
        break;
      case 'g': sigma = std::atoi(optarg); break;
      case 'l': league_size = std::atoi(optarg); break;
      case 't': team_size = std::atoi(optarg); break;
    }
  }

  if (help) {
    std::cout << "randomaccess <optional_args>"
                 "\n-s/--size:             The size of the problem in kB "
                 "(default: 1000)"
                 "\n-g/--sigma:            Sigma used to conpute variance "
                 "normalized to 1000 (default: 1000)"
                 "\n-t/--team_size:        The team size (default: 32)"
                 "\n-l/--league_size:      The league size (default: 1)"
                 "\n-h/--help:             Prints this help message"
                 "\n";
    return 0;
  }

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

  // Vars
  float time = 0;
  int my_rank;
  int num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  ORDINAL_T next_iters = num_iters;
  ORDINAL_T elems_per_rank;

  // Compute variance: Three Sigma rule
  // https://en.wikipedia.org/wiki/68–95–99.7_rule

  const float sigma_fixed = 0.20;
  float variance;
  if (sigma <= 0)
    variance = sigma_fixed;
  else {
    float max_spread = num_elems * sigma_fixed;
    variance         = max_spread / 1000.0 * sigma;
  }

  {
    Kokkos::ScopeGuard guard(argc, argv);
    TeamPolicy policy = TeamPolicy(league_size, team_size, vec_len);

    ORDINAL_T num_elems_per_rank;
    ORDINAL_T iters_per_team;
    num_elems_per_rank = ceil(1.0 * num_elems / num_ranks);

    RemoteView_t v("RemoteView", num_ranks, num_elems_per_rank);

    do {
      Generator_t gen_pool(5374857);

      // Update execution parameters
      num_iters      = next_iters;
      iters_per_team = ceil(num_iters / league_size);
      num_iters      = iters_per_team * league_size;

      Kokkos::Timer timer;

      Kokkos::parallel_for(
          "Outer", policy, KOKKOS_LAMBDA(const TeamPolicy::member_type &team) {
            Generator_t::generator_type g = gen_pool.get_state();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, my_rank * iters_per_team,
                                        (my_rank + 1) * iters_per_team),
                [&](const int i) {
                  int rank;
                  ORDINAL_T index;
                  index = abs(get(i, variance, g));
                  rank  = index / elems_per_rank;
                  index = index % num_elems_per_rank;
                  v(rank, index) ^= 0xC0FFEE;
                });
            gen_pool.free_state(g);
          });

      RemoteSpace_t().fence();
      time = timer.seconds();

      // Increase iteration space to reach a 2 sec. execution time.
      if (next_iters * 4 > std::numeric_limits<ORDINAL_T>::max() / 4) break;
      next_iters *= 4;
    } while (time <= 2.0);
  }

  if (my_rank == 0) {
    float GBs =
        (2.0 * num_iters * sizeof(ORDINAL_T)) / 1024.0 / 1024.0 / 1024.0 / time;
    float access_latency = time / num_iters * 1.0e6;

    printf("%i, %i, %i, %i, %ld, %.3f, %.2f, %.4f\n", num_ranks, league_size,
           team_size, vec_len, num_elems, access_latency, time, GBs);
  }

  MPI_Finalize();
}
