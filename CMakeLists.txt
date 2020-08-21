cmake_minimum_required (VERSION 3.12)

project(KokkosRemote 
  LANGUAGES CXX 
  VERSION 0.1.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)

find_package(Kokkos REQUIRED)

option(Kokkos_ENABLE_NVSHMEMSPACE "Whether to build with NVSHMEM space" OFF)
option(Kokkos_ENABLE_SHMEMSPACE   "Whether to build with SHMEMS space" OFF)
option(Kokkos_ENABLE_MPISPACE     "Whether to build with MPI space" OFF)
option(Kokkos_ENABLE_TESTS   "Whether to enable tests" OFF)

set(SOURCE_DIRS)
set(PUBLIC_DEPS)
set(PRIVATE_DEPS)

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
# Make sure kokkos has RDC and lambda support
  kokkos_check(
    OPTIONS CUDA_LAMBDA CUDA_RELOCATABLE_DEVICE_CODE
    DEVICES CUDA
  )
  find_package(NVSHMEM REQUIRED)
  list(APPEND SOURCE_DIRS NVSHMEMSPACE)
  list(APPEND PUBLIC_DEPS NVSHMEMSPACE)
endif()
if (Kokkos_ENABLE_SHMEMSPACE)
  find_package(SHMEM REQUIRED)
  list(APPEND SOURCE_DIRS SHMEMSPACE)
  list(APPEND PUBLIC_DEPS SHMEMSPACE)
endif()
if (Kokkos_ENABLE_MPISPACE)
  add_library(MPISPACE INTERFACE)
  add_library(Kokkos::MPISPACE ALIAS MPISPACE)
  list(APPEND SOURCE_DIRS MPISPACE)
  list(APPEND PUBLIC_DEPS MPISPACE)
endif()

if (NOT SOURCE_DIRS)
  message(FATAL_ERROR "Must give at least one valid backend")
endif()

set(SOURCES)
set(HEADERS)
foreach(DIR ${SOURCE_DIRS})
  file(GLOB DIR_SRCS src/${DIR}/*.cpp)
  file(GLOB DIR_HDRS src/${DIR}/*.hpp)
  list(APPEND SOURCES ${DIR_SRCS})
  list(APPEND HEADERS ${DIR_HDRS})
endforeach()
list(APPEND HEADERS src/Kokkos_RemoteSpaces.hpp)
list(APPEND HEADERS src/Kokkos_RemoteSpaces_DeepCopy.hpp)

add_library(kokkosremote ${SOURCES} ${HEADERS})
add_library(Kokkos::kokkosremote ALIAS kokkosremote)
target_link_libraries(kokkosremote PUBLIC Kokkos::kokkos)

foreach(DIR ${SOURCE_DIRS})
  target_include_directories(kokkosremote PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/${DIR}>)
  target_compile_definitions(kokkosremote PUBLIC KOKKOS_ENABLE_${DIR})
endforeach()
target_include_directories(kokkosremote PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_include_directories(kokkosremote PUBLIC $<INSTALL_INTERFACE:include>)

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
