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
#include <string>

//#define USE_GLOBAL_LAYOUT
//#define USE_PARTITIONED_LAYOUT
#define USE_LOCAL_LAYOUT

//#define GENMODE genmode::random_sequence
#define GENMODE genmode::linear_sequence

// Default values
#define SIZE 1024
#define NUM_ITER 10000
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

enum class genmode { random_sequence, linear_sequence };

template <genmode gm>
KOKKOS_INLINE_FUNCTION auto get(Generator_t::generator_type &g, int i,
                                auto team_id, auto team_range, ORDINAL_T start,
                                ORDINAL_T range) {
  if constexpr (gm == genmode::random_sequence) return start + g.urand64(range);
  if constexpr (gm == genmode::linear_sequence)
    return start + team_id * team_range + i % team_range;
}

int main(int argc, char *argv[]) {
  ORDINAL_T num_elems = (SIZE << 10) / sizeof(ORDINAL_T);
  ORDINAL_T num_iters = NUM_ITER;
  float spread        = SPREAD;
  int league_size     = LEAGUE_SIZE;
  int team_size       = TEAM_SIZE;
  int vec_len         = VEC_LEN;

  std::string view_name;

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

  ORDINAL_T idx_range;
  ORDINAL_T start_idx;

  {
    Kokkos::ScopeGuard guard(argc, argv);
    TeamPolicy policy = TeamPolicy(league_size, team_size, vec_len);

    ORDINAL_T num_elems_per_rank, elems_per_team, iters_per_team;
    num_elems_per_rank = ceil(1.0 * num_elems / num_ranks);
    num_elems          = num_ranks * num_elems_per_rank;

    ORDINAL_T default_start = my_rank * num_elems_per_rank;
    ORDINAL_T default_end   = (my_rank + 1) * num_elems_per_rank;

    start_idx = my_rank == 0 ? 0 : default_start - default_start * spread;
    ORDINAL_T end_idx = my_rank == num_ranks - 1
                            ? num_elems
                            : default_end + (num_elems - default_end) * spread;
    idx_range = end_idx - start_idx;

#if defined(USE_PARTITIONED_LAYOUT)
    View_t v("PartitionedView", num_ranks, num_elems_per_rank);
#elif defined(USE_GLOBAL_LAYOUT)
    View_t v("GlobalView", num_elems);
#elif defined(USE_LOCAL_LAYOUT)
    View_t v("LocalView", num_elems_per_rank);
    start_idx   = 0;
    default_end = num_elems_per_rank;
    assert(spread == 0);
#endif

    ORDINAL_T next_iters = num_iters;
    Generator_t gen_pool(5374857);
    do {
      // Set execution parameters
      num_iters      = next_iters;
      iters_per_team = ceil(num_iters / league_size);
      elems_per_team = floor(num_elems_per_rank / league_size);
      num_iters      = iters_per_team * league_size;
      Kokkos::Timer timer;

      Kokkos::parallel_for(
          "Outer", policy, KOKKOS_LAMBDA(const TeamPolicy::member_type &team) {
            auto g       = gen_pool.get_state();
            auto team_id = team.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, 0, iters_per_team),
                [&](const int i) {
                  ORDINAL_T index;
                  index = get<GENMODE>(g, i, team_id, elems_per_team, start_idx,
                                       idx_range);
#ifdef USE_PARTITIONED_LAYOUT
                  int rank;
                  rank  = index / num_elems_per_rank;
                  index = index % num_elems_per_rank;
                  v(rank, index) ^= 1;
#else
                  v(index) ^= 1;
#endif
                });
            gen_pool.free_state(g);
          });

      Kokkos::fence();
      RemoteSpace_t().fence();
      time = timer.seconds();

      // Increase iteration space to reach a 1 seconds of execution time.
      if (next_iters * 4 > std::numeric_limits<ORDINAL_T>::max() / 4) break;
      next_iters *= 4;
    } while (time <= 1.0);

    view_name = v.label();
  }

  if (my_rank == 0) {
    float MB             = num_elems * sizeof(ORDINAL_T) / 1024.0 / 1024.0;
    float MUPs           = static_cast<float>(num_iters) * 1.0e-6 / time;
    float MBs            = MUPs * static_cast<float>(sizeof(ORDINAL_T));
    float access_latency = time / num_iters * 1.0e6;

    printf("%s,%i,%i,%i,%i,%ld,%ld,%ld,%.3f,%.3f,%.3f,%.3f,%.3f\n",
           view_name.c_str(), num_ranks, league_size, team_size, vec_len,
           num_elems, start_idx, idx_range, MB, access_latency, time, MUPs,
           MBs);
  }

  MPI_Finalize();
}
