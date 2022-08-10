add_executable(matvec_single matvec.cpp)

if (KRS_ENABLE_NVSHMEMSPACE)
  target_link_libraries(matvec_single PRIVATE "-lnvidia-ml")
endif()


target_link_libraries(matvec_single PRIVATE Kokkos::kokkosremote)
target_include_directories(matvec_single PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


