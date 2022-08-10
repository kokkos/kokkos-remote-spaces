add_executable(misslatency misslatency.cpp)

if (KRS_ENABLE_NVSHMEMSPACE)
  target_link_libraries(misslatency PRIVATE "-lnvidia-ml")
endif()

target_link_libraries(misslatency PRIVATE Kokkos::kokkosremote)
