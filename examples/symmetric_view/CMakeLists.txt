add_executable(symmetric_view main.cpp)
target_link_libraries(symmetric_view PRIVATE Kokkos::kokkosremote)
target_include_directories(symmetric_view PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
