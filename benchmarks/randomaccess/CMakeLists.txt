add_executable(randomaccess randomaccess.cpp)
target_link_libraries(randomaccess PRIVATE Kokkos::kokkosremotespaces)
