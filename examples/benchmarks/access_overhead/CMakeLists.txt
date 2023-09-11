FILE(GLOB SRCS *.cpp)

foreach(file ${SRCS})
  get_filename_component(test_name ${file} NAME_WE)
  add_executable(${test_name} ${file})
  target_link_libraries(${test_name} PRIVATE Kokkos::kokkosremotespaces)
endforeach()
