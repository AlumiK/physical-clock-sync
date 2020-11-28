cmake_minimum_required(VERSION 3.16)
project(PhysicalClocksSync)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(${PROJECT_SOURCE_DIR}/01_Cristian)
add_subdirectory(${PROJECT_SOURCE_DIR}/02_BerkeleyAlgorithm)
add_subdirectory(${PROJECT_SOURCE_DIR}/03_NTP)
