if(Kokkos_ENABLE_NVSHMEM)
add_subdirectory(poisson-misses)
add_subdirectory(miss-latency)
endif()
add_subdirectory(randomaccess)

