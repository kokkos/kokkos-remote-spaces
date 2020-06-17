
find_library(nvlib_found nvshmem PATHS ${NVSHMEM_ROOT} SUFFIXES lib lib64 NO_DEFAULT_PATHS)
find_path(nvhdr_found nvshmem.h PATHS ${NVSHMEM_ROOT}/include NO_DEFAULT_PATHS)

find_package_handle_standard_args(NVSHMEMSPACE DEFAULT_MSG nvlib_found nvhdr_found)

if (nvlib_found AND nvhdr_found)
  add_library(NVSHMEMSPACE INTERFACE)
  add_library(Kokkos::NVSHMEMSPACE ALIAS NVSHMEMSPACE)
  set_target_properties(NVSHMEMSPACE PROPERTIES
    INTERFACE_LINK_LIBRARIES ${nvlib_found}
    INTERFACE_INCLUDE_DIRECTORIES ${nvhdr_found}
  )
endif()
