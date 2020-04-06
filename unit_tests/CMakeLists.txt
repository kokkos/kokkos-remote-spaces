include(CTest)

function(kokkos_add_test NAME SOURCE)
  add_executable(KokkosRemote_${NAME} ${SOURCE})
  target_link_libraries(KokkosRemote_${NAME} PRIVATE Kokkos::kokkosremote)
  target_link_libraries(KokkosRemote_${NAME} PRIVATE gtest_main)
  add_test(KokkosRemote_${NAME} COMMAND KokkosRemote_${NAME} ${ARGN})
endfunction()

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           master
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR}) 
endif()

if (Kokkos_ENABLE_MPI)
  kokkos_add_test(Allocation Test_Allocation.cpp)
endif()
if (Kokkos_ENABLE_SHMEM)
  kokkos_add_test(Main Test_Main.cpp)
endif()
