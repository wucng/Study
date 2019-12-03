cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(mnist)

set(Torch_DIR /opt/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_executable(mnist mnist.cpp)
# target_compile_features(mnist PUBLIC cxx_range_for)
target_link_libraries(mnist ${TORCH_LIBRARIES})
set_property(TARGET mnist PROPERTY CXX_STANDARD 11)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET mnist
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:mnist>)
endif (MSVC)
