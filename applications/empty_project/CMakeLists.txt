cmake_minimum_required(VERSION 3.25)
project(MyProject)

find_package(KokkosRemoteSpaces REQUIRED)

add_executable(project project.cpp)
target_link_libraries(project PRIVATE Kokkos::kokkosremotespaces)
target_include_directories(project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
