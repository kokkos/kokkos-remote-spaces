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

SET(NAME KokkosRemoteTest)
file(GLOB TEST_SRCS *.cpp)
list(APPEND TEST_SOURCES_ALL ${TEST_SRCS})
if (Kokkos_ENABLE_RACERLIB) 
  file(GLOB TEST_SRCS experimental/*.cpp)
  list(APPEND TEST_SOURCES_ALL ${TEST_SRCS})
endif()
add_executable(${NAME} ${TEST_SOURCES_ALL})
target_link_libraries(${NAME} PUBLIC Kokkos::kokkosremote)
target_link_libraries(${NAME} PUBLIC gtest_main)
