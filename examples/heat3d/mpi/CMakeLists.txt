add_executable(mpi_heat3d heat3d.cpp)
target_link_libraries(mpi_heat3d PRIVATE Kokkos::kokkosremotespaces)
target_include_directories(mpi_heat3d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
