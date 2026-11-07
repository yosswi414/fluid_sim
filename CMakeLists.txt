cmake_minimum_required(VERSION 3.18)
project(FluidSim LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "120")

add_executable(fluid_sim main.cpp kernel.cu)

target_link_libraries(fluid_sim PRIVATE sfml-graphics sfml-window sfml-system)
