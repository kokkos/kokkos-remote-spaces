add_executable(randomaccess randomaccess.cpp)
target_link_libraries(randomaccess PRIVATE Kokkos::kokkosremote)
target_include_directories(randomaccess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
#target_compile_definitions(randomaccess PRIVATE KOKKOS_ENABLE_NVSHMEM_PTR)


