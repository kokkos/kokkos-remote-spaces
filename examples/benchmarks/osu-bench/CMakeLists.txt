add_executable(osu_bench osu_bench.cpp)
target_link_libraries(osu_bench PRIVATE Kokkos::kokkosremote)
target_include_directories(osu_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(osu_bench PRIVATE KRS_ENABLE_NVSHMEM_PTR)
