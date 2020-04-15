add_executable(poisson poisson.cpp)
target_link_libraries(poisson PRIVATE Kokkos::kokkosremote)
