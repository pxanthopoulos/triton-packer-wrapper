cmake_minimum_required(VERSION 3.27)
project(triton_packer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLVM_DIR "/usr/lib/llvm-14/lib/cmake/llvm")

find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(triton_packer main.cpp)

llvm_map_components_to_libnames(llvm_libs support core irreader)
target_link_libraries(triton_packer ${llvm_libs})