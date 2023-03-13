add_executable(misslatency misslatency.cpp)
target_link_libraries(misslatency PRIVATE Kokkos::kokkosremote)
