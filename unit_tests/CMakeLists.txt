include(CTest)

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

SET(NAME KokkosRemote_TestAll)
FILE(GLOB TEST_SRCS *.cpp) 
add_executable(${NAME} ${TEST_SRCS})
target_link_libraries(${NAME} PUBLIC Kokkos::kokkosremote)
target_link_libraries(${NAME} PUBLIC gtest_main)
