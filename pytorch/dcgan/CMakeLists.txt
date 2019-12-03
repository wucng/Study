cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)

set(Torch_DIR /opt/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 11)
