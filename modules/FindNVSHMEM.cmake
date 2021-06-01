
find_library(nvshmem_lib_found nvshmem PATHS ${NVSHMEM_ROOT} SUFFIXES lib lib64 NO_DEFAULT_PATHS)
find_path(nvshmem_headers_found nvshmem.h PATHS ${NVSHMEM_ROOT}/include NO_DEFAULT_PATHS)

find_package_handle_standard_args(NVSHMEM DEFAULT_MSG nvshmem_lib_found nvshmem_headers_found)

if (nvshmem_lib_found AND nvshmem_headers_found)
  add_library(NVSHMEMSPACE INTERFACE)
  add_library(Kokkos::NVSHMEMSPACE ALIAS NVSHMEMSPACE)
  set_target_properties(NVSHMEMSPACE PROPERTIES
    INTERFACE_LINK_LIBRARIES ${nvshmem_lib_found}
    INTERFACE_INCLUDE_DIRECTORIES ${nvshmem_headers_found}
  )
endif()
