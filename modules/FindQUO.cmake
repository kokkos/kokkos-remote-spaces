
find_library(libquo_found nvshmem PATHS ${QUO_ROOT} SUFFIXES lib lib64 NO_DEFAULT_PATHS)
find_path(quohdr_found nvshmem.h PATHS ${QUO_ROOT}/include NO_DEFAULT_PATHS)

find_package_handle_standard_args(QUO DEFAULT_MSG libquo_found quohdr_found)

if (libquo_found AND quohdr_found)
  add_library(QUO INTERFACE)
  set_target_properties(QUO PROPERTIES
    INTERFACE_LINK_LIBRARIES ${libquo_found}
    INTERFACE_INCLUDE_DIRECTORIES ${quohdr_found}
  )
endif()
