add_executable(poissonaccess poissonaccess.cpp)

if (KRS_ENABLE_NVSHMEMSPACE)
  target_link_libraries(poissonaccess PRIVATE "-lnvidia-ml")
endif()

target_link_libraries(poissonaccess PRIVATE Kokkos::kokkosremote)
