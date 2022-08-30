add_executable(cgsolve cgsolve.cpp)
target_link_libraries(cgsolve PRIVATE Kokkos::kokkosremote)
target_include_directories(cgsolve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(cgsolve PRIVATE KRS_ENABLE_NVSHMEM_PTR)
