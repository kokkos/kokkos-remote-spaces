add_executable(shift_single vectorshift.cpp)
target_link_libraries(shift_single PRIVATE Kokkos::kokkosremotespaces)
target_include_directories(shift_single PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


