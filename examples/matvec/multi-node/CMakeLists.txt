add_executable(matvec_multi matvec.cpp)

if (Kokkos_ENABLE_NVSHMEMSPACE)
  target_link_libraries(matvec_multi PRIVATE "-lnvidia-ml")
endif()

target_link_libraries(matvec_multi PRIVATE Kokkos::kokkosremote)
target_include_directories(matvec_multi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


