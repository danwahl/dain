cmake_minimum_required(VERSION 3.15)
project(dain LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_library(kernels SHARED
    src/cuda/kernels.cu
)

target_include_directories(kernels PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

install(TARGETS kernels
    LIBRARY DESTINATION dain)
