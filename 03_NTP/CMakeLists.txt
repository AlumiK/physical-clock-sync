cmake_minimum_required(VERSION 3.17)
project(03_NTP)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER mpicc)

add_executable(03_NTP main.cpp)
