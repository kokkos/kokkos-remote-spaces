# libsma.{so/a} are for Cray-SHMEM and Cray-OpenSHMEMX
find_library(shlib_found NAMES oshmem sma PATHS ${SHMEM_ROOT} SUFFIXES lib lib64 NO_DEFAULT_PATHS)
find_path(shhdr_found shmem.h PATHS ${SHMEM_ROOT}/include NO_DEFAULT_PATHS)

find_package_handle_standard_args(SHMEMSPACE DEFAULT_MSG shlib_found shhdr_found)

if (shlib_found AND shhdr_found)
  add_library(SHMEMSPACE INTERFACE)
  set_target_properties(SHMEMSPACE PROPERTIES
    INTERFACE_LINK_LIBRARIES ${shlib_found}
    INTERFACE_INCLUDE_DIRECTORIES ${shhdr_found}
  )
endif()
