FILE(GLOB SRCS *.cpp)

set(flavor rma)

foreach(file ${SRCS})
  get_filename_component(test_name_file ${file} NAME_WE)
  set(test_name ${test_name_file}_${flavor})
  add_executable(${test_name} ${file})
  target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${test_name} PRIVATE Kokkos::kokkosremotespaces)
endforeach()
