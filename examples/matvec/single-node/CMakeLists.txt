add_executable(matvec_single matvec.cpp)

target_link_libraries(matvec_single PRIVATE Kokkos::kokkosremote)
target_include_directories(matvec_single PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


