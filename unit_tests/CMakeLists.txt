include(CTest)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           main
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

SET(NAME KokkosRemote_TestAll)

if (KRS_ENABLE_RACERLIB) #For RACERlib, compile just this unit test
  file(GLOB TEST_SRCS_EXPERIMENTAL experimental/*.cpp)
  list(APPEND TEST_SRCS_EXPERIMENTAL "Test_Main.cpp")
  list(APPEND TEST_SRCS ${TEST_SRCS_EXPERIMENTAL})
  message(STATUS "Enabled experimental unit test(s)")
else()
  if (KRS_ENABLE_MPISPACE)
    FILE(GLOB TEST_SRCS *.cpp)
    list(FILTER TEST_SRCS EXCLUDE REGEX ".*Test_Subview\\.cpp$")
    message(STATUS "Skipping unit test(s):${TEST_SRCS}")
  else()
    FILE(GLOB TEST_SRCS *.cpp)
  endif()
endif()

add_executable(${NAME} ${TEST_SRCS})

target_link_libraries(${NAME} PRIVATE Kokkos::kokkosremote)
target_link_libraries(${NAME} PRIVATE gtest_main)
