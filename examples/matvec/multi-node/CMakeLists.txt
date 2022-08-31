add_executable(matvec_multi matvec.cpp)
target_link_libraries(matvec_multi PRIVATE Kokkos::kokkosremote)
target_include_directories(matvec_multi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


