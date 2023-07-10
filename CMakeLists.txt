cmake_minimum_required (VERSION 3.12)

project(KokkosRemote
  LANGUAGES CXX
  VERSION 0.1.0)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)

find_package(Kokkos REQUIRED)

option(KRS_ENABLE_NVSHMEMSPACE "Whether to build with NVSHMEM space" OFF)
option(KRS_ENABLE_ROCSHMEMSPACE "Whether to build with ROCSHMEM space" OFF)
option(KRS_ENABLE_SHMEMSPACE "Whether to build with SHMEMS space" OFF)
option(KRS_ENABLE_MPISPACE "Whether to build with MPI space" OFF)
option(KRS_ENABLE_EXAMPLES "Whether to enable tests" OFF)
option(KRS_ENABLE_TESTS "Whether to enable tests" OFF)
option(KRS_ENABLE_DEBUG "Whether to enable debugging output" OFF)

set(SOURCE_DIRS)
set(PUBLIC_DEPS)
set(PRIVATE_DEPS)
set(BACKENDS)

# Append public headers DIR to SOURCE_DIRS
list(APPEND SOURCE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src/core)

set(PREFIX_BACKEND_SRC_PATH "src/impl")
set(NVSHMEMSPACE_PATH "${PREFIX_BACKEND_SRC_PATH}/nvshmemspace")
set(ROCSHMEMSPACE_PATH "${PREFIX_BACKEND_SRC_PATH}/rocshmemspace")
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

if (KRS_ENABLE_NVSHMEMSPACE)
# Requiere Kokkos with RDC and Lambda support
  kokkos_check(
    OPTIONS CUDA_LAMBDA CUDA_RELOCATABLE_DEVICE_CODE
    DEVICES CUDA)
  find_package(NVSHMEM REQUIRED)
  set(BACKEND_NAME NVSHMEMSPACE)
  list(APPEND SOURCE_DIRS ${NVSHMEMSPACE_PATH})
  list(APPEND PUBLIC_DEPS ${BACKEND_NAME})
  list(APPEND BACKENDS ${BACKEND_NAME})
endif()
if (KRS_ENABLE_ROCSHMEMSPACE)
# Require Kokkos with RDC
  kokkos_check(
    OPTIONS HIP_RELOCATABLE_DEVICE_CODE
    DEVICES HIP)
  find_package(ROCSHMEM REQUIRED)
  set(BACKEND_NAME ROCSHMEMSPACE)
  list(APPEND SOURCE_DIRS ${ROCSHMEMSPACE_PATH})
  list(APPEND PUBLIC_DEPS ${BACKEND_NAME})
  list(APPEND BACKENDS ${BACKEND_NAME})
endif()
if (KRS_ENABLE_SHMEMSPACE)
  find_package(SHMEM REQUIRED)
  set(BACKEND_NAME SHMEMSPACE)
  list(APPEND SOURCE_DIRS ${SHMEMSPACE_PATH})
  list(APPEND PUBLIC_DEPS ${BACKEND_NAME})
  list(APPEND BACKENDS ${BACKEND_NAME})
endif()
if (KRS_ENABLE_MPISPACE)
  add_library(MPISPACE INTERFACE)
  set(BACKEND_NAME MPISPACE)
  list(APPEND SOURCE_DIRS ${MPISPACE_PATH})
  list(APPEND PUBLIC_DEPS ${BACKEND_NAME})
  list(APPEND BACKENDS ${BACKEND_NAME})
endif()
if (KRS_ENABLE_RACERLIB)
  find_package(IBVERBS REQUIRED)
  list(APPEND PUBLIC_DEPS IBVERBS)
endif()

message(STATUS "Enabled remote spaces: ${BACKENDS}")

list(LENGTH BACKENDS N_BACKENDS)
if (NOT ${N_BACKENDS} EQUAL "1")
  message(FATAL_ERROR "Must give a single valid backend. ${N_BACKENDS} given.")
endif()

set(SOURCES)
set(HEADERS)
foreach(DIR ${SOURCE_DIRS})
  file(GLOB DIR_SRCS ${DIR}/*.cpp)
  file(GLOB DIR_HDRS ${DIR}/*.hpp)
  list(APPEND SOURCES ${DIR_SRCS})
  list(APPEND HEADERS ${DIR_HDRS})
endforeach()

if (KRS_ENABLE_RACERLIB)
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
  foreach(BACKEND_NAME ${PUBLIC_DEPS})
    target_compile_definitions(kokkosremote PUBLIC KRS_ENABLE_${BACKEND_NAME})
  endforeach()
endforeach()

if (KRS_ENABLE_RACERLIB)
  target_include_directories(kokkosremote PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/features/racerlib>)
  target_compile_definitions(kokkosremote PUBLIC KRS_ENABLE_RACERLIB)
endif()

target_include_directories(kokkosremote PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/core>)
target_include_directories(kokkosremote PUBLIC $<INSTALL_INTERFACE:include>)

if(KRS_ENABLE_DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
 target_compile_definitions(kokkosremote PUBLIC KOKKOS_REMOTE_SPACES_ENABLE_DEBUG)
 message(STATUS "Enabled build mode: debug")
endif()

if (PRIVATE_DEPS)
  target_link_libraries(kokkosremote PRIVATE ${PRIVATE_DEPS})
endif()
if (PUBLIC_DEPS)
  target_link_libraries(kokkosremote PUBLIC ${PUBLIC_DEPS})
endif()

if (KRS_ENABLE_NVSHMEMSPACE)
  target_link_libraries(kokkosremote PRIVATE "-lnvidia-ml")
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

IF (KRS_ENABLE_EXAMPLES)
  add_subdirectory(examples)
ENDIF()

IF (KRS_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(unit_tests)
ENDIF()
