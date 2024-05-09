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

//#define USE_PARTITIONED_LAYOUT
#define USE_GLOBAL_LAYOUT
//#define USE_LOCAL_LAYOUT

// Default values
#define SIZE 1024
#define NUM_ITER 1000
#define LEAGUE_SIZE 1
#define TEAM_SIZE 32
#define VEC_LEN 1
#define SPREAD 1

#define ORDINAL_T int64_t

using RemoteSpace_t = Kokkos::Experimental::DefaultRemoteMemorySpace;
using Generator_t   = Kokkos::Random_XorShift64_Pool<>;
using TeamPolicy    = Kokkos::TeamPolicy<>;

#if defined(USE_PARTITIONED_LAYOUT)
using View_t =
    Kokkos::View<ORDINAL_T **, Kokkos::PartitionedLayoutLeft, RemoteSpace_t>;
#elif defined(USE_GLOBAL_LAYOUT)
using View_t = Kokkos::View<ORDINAL_T *, RemoteSpace_t>;
#elif defined(USE_LOCAL_LAYOUT)
using View_t = Kokkos::View<ORDINAL_T *>;
#else
#error "What View-type is this?"
#endif

KOKKOS_INLINE_FUNCTION
ORDINAL_T get(Generator_t::generator_type &g, ORDINAL_T start,
              ORDINAL_T range) {
  return start + g.urand64(range);
}

int main(int argc, char *argv[]) {
  ORDINAL_T num_elems = (SIZE << 10) / sizeof(ORDINAL_T);
  ORDINAL_T num_iters = NUM_ITER;
  float spread        = SPREAD;
  int league_size     = LEAGUE_SIZE;
  int team_size       = TEAM_SIZE;
  int vec_len         = VEC_LEN;

  string view_name;

  option gopt[] = {
      {"help", no_argument, NULL, 'h'},
      {"size", required_argument, NULL, 's'},
      {"spread", required_argument, NULL, 'p'},
      {"leage_size", required_argument, NULL, 'l'},
      {"team_size", required_argument, NULL, 't'},
  };

  int ch;
  bool help = false;
  while ((ch = getopt_long(argc, argv, "hs:g:l:t:p:", gopt, NULL)) != -1) {
    switch (ch) {
      case 0:
        // this set an input flag
        break;
      case 'h': help = true; break;
      case 's':
        num_elems = (std::atoi(optarg) << 10) / sizeof(ORDINAL_T);
        break;
      case 'p': spread = std::atof(optarg); break;
      case 'l': league_size = std::atoi(optarg); break;
      case 't': team_size = std::atoi(optarg); break;
    }
  }

  if (help) {
    std::cout << "randomaccess <optional_args>"
                 "\n-s/--size:             The size of the problem in kB "
                 "\n-p/--spread:           The spread over array from 0 to 1 "
                 "(local to global)"
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

  {
    Kokkos::ScopeGuard guard(argc, argv);
    TeamPolicy policy = TeamPolicy(league_size, team_size, vec_len);

    ORDINAL_T num_elems_per_rank;
    ORDINAL_T iters_per_team;
    num_elems_per_rank = ceil(1.0 * num_elems / num_ranks);
    num_elems          = num_ranks * num_elems_per_rank;

    ORDINAL_T default_start = my_rank * num_elems_per_rank;
    ORDINAL_T default_end   = (my_rank + 1) * num_elems_per_rank;

    ORDINAL_T start_idx =
        my_rank == 0 ? 0 : default_start - default_start * spread;
    ORDINAL_T end_idx = my_rank == num_ranks - 1
                            ? num_elems
                            : default_end + (num_elems - default_end) * spread;
    ORDINAL_T idx_range = end_idx - start_idx;

#if defined(USE_PARTITIONED_LAYOUT)
    View_t v("PartitionedView", num_ranks, num_elems_per_rank);
#elif defined(USE_GLOBAL_LAYOUT)
    View_t v("GlobalView", num_elems);
#elif defined(USE_LOCAL_LAYOUT)
    View_t v("LocalView", num_elems_per_rank);
    start_idx   = 0;
    default_end = num_elems_per_rank;
#endif

#ifdef USE_LOCAL_LAYOUT
    assert(spread == 0);
#endif

    Generator_t gen_pool(5374857);
    do {
      // Set execution parameters
      num_iters      = next_iters;
      iters_per_team = ceil(num_iters / league_size);
      num_iters      = iters_per_team * league_size;
      Kokkos::Timer timer;

      Kokkos::parallel_for(
          "Outer", policy, KOKKOS_LAMBDA(const TeamPolicy::member_type &team) {
            auto g = gen_pool.get_state();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, 0, iters_per_team),
                [&](const int i) {
                  ORDINAL_T index;
                  index = get(g, start_idx, idx_range);
#ifdef USE_PARTITIONED_LAYOUT
                  int rank;
                  rank  = index / num_elems_per_rank;
                  index = index % num_elems_per_rank;
                  v(rank, index) ^= 0xC0FFEE;
#else
                  v(index) ^= 0xC0FFEE;
#endif
                });
            gen_pool.free_state(g);
          });

      Kokkos::fence();
      RemoteSpace_t().fence();
      time = timer.seconds();

      // Increase iteration space to reach a 2 seconds of execution time.
      if (next_iters * 4 > std::numeric_limits<ORDINAL_T>::max() / 4) break;
      next_iters *= 4;
    } while (time <= 2.0);

    view_name = v.label();
  }

  if (my_rank == 0) {
    float MB             = num_elems * sizeof(ORDINAL_T) / 1024.0 / 1024.0;
    float MBs            = MB / time;
    float access_latency = time / num_iters * 1.0e6;

    printf("%s, %i, %i, %i, %i, %ld, %.3f, %.3f, %.3f, %.3f\n",
           view_name.c_str(), num_ranks, league_size, team_size, vec_len,
           num_elems, MB, access_latency, time, MBs);
  }

  MPI_Finalize();
}
