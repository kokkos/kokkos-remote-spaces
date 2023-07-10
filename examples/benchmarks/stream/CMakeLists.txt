add_executable(stream stream_benchmark.cpp)
target_link_libraries(stream PRIVATE Kokkos::kokkosremote)
target_include_directories(stream PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(stream PRIVATE KRS_ENABLE_NVSHMEM_PTR)
