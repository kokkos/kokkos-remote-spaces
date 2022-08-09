add_executable(cgsolve cgsolve.cpp)

if (Kokkos_ENABLE_NVSHMEMSPACE)
  target_link_libraries(cgsolve PRIVATE "-lnvidia-ml")
endif()

target_link_libraries(cgsolve PRIVATE Kokkos::kokkosremote)
target_include_directories(cgsolve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(cgsolve PRIVATE KOKKOS_ENABLE_NVSHMEM_PTR)
