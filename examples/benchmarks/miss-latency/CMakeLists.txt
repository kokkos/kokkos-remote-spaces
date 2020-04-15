add_executable(latency latency.cpp)
target_link_libraries(latency PRIVATE Kokkos::kokkosremote)
