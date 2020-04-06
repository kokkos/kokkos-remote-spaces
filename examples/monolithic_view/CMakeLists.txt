add_executable(monolithic_view main.cpp)
target_link_libraries(monolithic_view PRIVATE Kokkos::kokkosremote)
target_include_directories(monolithic_view PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
