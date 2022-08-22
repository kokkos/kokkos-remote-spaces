add_executable(randomaccess randomaccess.cpp)

if (KRS_ENABLE_NVSHMEMSPACE)
  target_link_libraries(randomaccess PRIVATE "-lnvidia-ml")
endif()

target_link_libraries(randomaccess PRIVATE Kokkos::kokkosremote)
