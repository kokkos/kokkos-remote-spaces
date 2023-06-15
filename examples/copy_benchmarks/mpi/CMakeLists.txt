add_executable(mpi_cpybench cpybench.cpp)
target_link_libraries(mpi_cpybench PRIVATE Kokkos::kokkosremote)
target_include_directories(mpi_cpybench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
