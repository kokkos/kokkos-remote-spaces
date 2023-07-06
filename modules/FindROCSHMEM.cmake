
find_library(rocshmem_lib_found rocshmem PATHS ${ROCSHMEM_ROOT} SUFFIXES lib lib64 NO_DEFAULT_PATHS)
find_path(rocshmem_headers_found roc_shmem.hpp PATHS ${ROCSHMEM_ROOT}/include NO_DEFAULT_PATHS)

find_package_handle_standard_args(ROCSHMEM DEFAULT_MSG rocshmem_lib_found rocshmem_headers_found)

if (rocshmem_lib_found AND rocshmem_headers_found)
  add_library(ROCSHMEMSPACE INTERFACE)
  add_library(Kokkos::ROCSHMEMSPACE ALIAS ROCSHMEMSPACE)
  set_target_properties(ROCSHMEMSPACE PROPERTIES
    INTERFACE_LINK_LIBRARIES ${rocshmem_lib_found}
    INTERFACE_INCLUDE_DIRECTORIES ${rocshmem_headers_found}
  )
endif()
