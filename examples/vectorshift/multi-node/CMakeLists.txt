add_executable(shift_multi vectorshift.cpp)
target_link_libraries(shift_multi PRIVATE Kokkos::kokkosremote)
target_include_directories(shift_multi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


