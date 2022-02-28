cmake_minimum_required (VERSION 3.16)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)

project(KokkosRemote
  LANGUAGES CXX
  VERSION 0.1.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)

find_package(Kokkos REQUIRED)

option(Kokkos_ENABLE_NVSHMEMSPACE "Whether to build with NVSHMEM space" OFF)
option(Kokkos_ENABLE_SHMEMSPACE "Whether to build with SHMEMS space" OFF)
option(Kokkos_ENABLE_MPISPACE "Whether to build with MPI space" OFF)
option(Kokkos_ENABLE_TESTS "Whether to enable tests" OFF)
option(Kokkos_ENABLE_DEBUG "Whether to enable debugging output" OFF)
option(KOKKOS_ENABLE_ACCESS_CACHING_AND_AGGREGATION "Whether to enable view caching and aggregation" OFF)

set(SOURCE_DIRS)
set(PUBLIC_DEPS)
set(PRIVATE_DEPS)

set(PREFIX_BACKEND_SRC_PATH "src/impl")

set(NVSHMEMSPACE_PATH "${PREFIX_BACKEND_SRC_PATH}/nvshmemspace")
set(SHMEMSPACE_PATH "${PREFIX_BACKEND_SRC_PATH}/shmemspace")
set(MPISPACE_PATH "${PREFIX_BACKEND_SRC_PATH}/mpispace")

find_package(MPI REQUIRED)
add_library(MPI INTERFACE)
list(APPEND MPI_CXX_LINK_FLAGS ${MPI_CXX_LIBRARIES})
set_target_properties(MPI PROPERTIES
  INTERFACE_COMPILE_OPTIONS "${MPI_CXX_COMPILE_FLAGS}"
  INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}"
  INTERFACE_LINK_LIBRARIES "${MPI_CXX_LINK_FLAGS}"
)

list(APPEND PUBLIC_DEPS MPI)

if (Kokkos_ENABLE_NVSHMEMSPACE)
  message(STATUS "Configuring for NVSHMEMSPACE.")
  # Make sure kokkos has RDC and lambda support
  kokkos_check(
    OPTIONS CUDA_LAMBDA CUDA_RELOCATABLE_DEVICE_CODE
    DEVICES CUDA)
  set(BACKEND_NAME NVSHMEMSPACE)
  add_library(${BACKEND_NAME} INTERFACE)
  add_library(Kokkos::${BACKEND_NAME} ALIAS ${BACKEND_NAME})
  find_package(NVSHMEM REQUIRED)
  list(APPEND SOURCE_DIRS ${NVSHMEMSPACE_PATH})
  list(APPEND PUBLIC_DEPS ${BACKEND_NAME})

  if(KOKKOS_ENABLE_ACCESS_CACHING_AND_AGGREGATION)
  # Make sure IBVerbs are present for Remote Access Caching and Aggregation (RACER)
    find_package(IBVERBS REQUIRED)
    list(APPEND PUBLIC_DEPS IBVERBS)
    set(ENABLE_ACCESS_CACHING_AND_AGGREGATION GLOBAL)
    message(STATUS "Enabling view caching and aggregation. ")
  endif()
endif()
if (Kokkos_ENABLE_SHMEMSPACE)
  message(STATUS "Configuring for SHMEMSPACE.")
  set(BACKEND_NAME SHMEMSPACE)
  add_library(${BACKEND_NAME} INTERFACE)
  add_library(Kokkos::${BACKEND_NAME} ALIAS ${BACKEND_NAME})
  find_package(SHMEM REQUIRED)
  list(APPEND SOURCE_DIRS ${SHMEMSPACE_PATH})
  list(APPEND PUBLIC_DEPS ${BACKEND_NAME})
  if(KOKKOS_ENABLE_ACCESS_CACHING_AND_AGGREGATION)
    message(FATAL_ERROR "View caching and aggregation is currently not supported for this backend. ")
  endif()
endif()
if (Kokkos_ENABLE_MPISPACE)
  message("Configuring for MPISPACE.")
  set(BACKEND_NAME MPISPACE)
  add_library(${BACKEND_NAME} INTERFACE)
  list(APPEND SOURCE_DIRS ${MPISPACE_PATH})
  list(APPEND PUBLIC_DEPS ${BACKEND_NAME})
  if(KOKKOS_ENABLE_ACCESS_CACHING_AND_AGGREGATION)
    message(FATAL_ERROR "View caching and aggregation is currently not supported for this backend. ")
  endif()
endif()

if (NOT SOURCE_DIRS)
  message(FATAL_ERROR "Must give at least one valid backend")
endif()

set(SOURCES)
set(HEADERS)
foreach(DIR ${SOURCE_DIRS})
  file(GLOB DIR_SRCS ${DIR}/*.cpp)
  file(GLOB DIR_HDRS ${DIR}/*.hpp)
  list(APPEND SOURCES ${DIR_SRCS})
  list(APPEND HEADERS ${DIR_HDRS})
endforeach()

if (ENABLE_ACCESS_CACHING_AND_AGGREGATION)
  message(STATUS "Adding global sources to target for: RACERlib")
  file(GLOB DIR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/features/racerlib/*.cpp)
  file(GLOB DIR_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/src/features/racerlib/*.hpp)
  list(APPEND SOURCES ${DIR_SRCS})
  list(APPEND HEADERS ${DIR_HDRS})
 endif()

add_library(kokkosremote ${SOURCES} ${HEADERS})
add_library(Kokkos::kokkosremote ALIAS kokkosremote)
target_link_libraries(kokkosremote PUBLIC Kokkos::kokkos)

foreach(DIR ${SOURCE_DIRS})
  target_include_directories(kokkosremote PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${DIR}>)
  target_compile_definitions(kokkosremote PUBLIC KOKKOS_ENABLE_${BACKEND_NAME})
endforeach()

if (ENABLE_ACCESS_CACHING_AND_AGGREGATION)
  message(STATUS "Adding to compile target: RACERlib")
  target_include_directories(kokkosremote PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/features/racerlib>)
  target_compile_definitions(kokkosremote PUBLIC KOKKOS_ENABLE_ACCESS_CACHING_AND_AGGREGATION)
endif()

target_include_directories(kokkosremote PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/core>)
target_include_directories(kokkosremote PUBLIC $<INSTALL_INTERFACE:include>)

if(Kokkos_ENABLE_DEBUG)
 target_compile_definitions(kokkosremote PUBLIC KOKKOS_IBV_DEBUG)
endif()

if (PRIVATE_DEPS)
  target_link_libraries(kokkosremote PRIVATE ${PRIVATE_DEPS})
endif()
if (PUBLIC_DEPS)
  target_link_libraries(kokkosremote PUBLIC ${PUBLIC_DEPS})
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  KokkosRemoteConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/KokkosRemoteConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}/cmake
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/KokkosRemoteConfigVersion.cmake"
  VERSION "${KokkosRemote_VERSION}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/KokkosRemoteConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/KokkosRemoteConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosRemote)

install(FILES
  ${HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  TARGETS kokkosremote ${PUBLIC_DEPS}
  EXPORT KokkosRemoteTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT
  KokkosRemoteTargets
  NAMESPACE Kokkos::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KokkosRemote
)

add_subdirectory(examples)

IF (Kokkos_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(unit_tests)
ENDIF()
