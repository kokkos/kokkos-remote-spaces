include(FindPackageHandleStandardArgs)
find_library(verbs_found ibverbs PATHS ${IBVERBS_ROOT} SUFFIXES lib lib64 NO_DEFAULT_PATHS)
find_path(verbs_h_found infiniband/verbs.h PATHS ${IBVERBS_ROOT}/include NO_DEFAULT_PATHS)

if (NOT verbs_found)
find_library(verbs_found ibverbs)
find_path(verbs_h_found infiniband/verbs.h)
endif()

find_package_handle_standard_args(IBVERBS DEFAULT_MSG verbs_h_found verbs_found)

if (verbs_found AND verbs_h_found)
  if (NOT TARGET IBVERBS)
    add_library(IBVERBS INTERFACE)
    set_target_properties(IBVERBS PROPERTIES
      INTERFACE_LINK_LIBRARIES ${verbs_found}
      INTERFACE_INCLUDE_DIRECTORIES ${verbs_h_found}
    )
  endif()
endif()
