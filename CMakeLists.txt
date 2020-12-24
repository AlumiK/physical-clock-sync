cmake_minimum_required(VERSION 3.16)
project(physical_clock_sync)

set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cristian_algorithm src/cristian_algorithm.cc)
target_link_libraries(cristian_algorithm PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(berkeley_algorithm src/berkeley_algorithm.cc)
target_link_libraries(berkeley_algorithm PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(ntp src/ntp.cc)
target_link_libraries(ntp PRIVATE MPI::MPI_CXX)

enable_testing()
set(TEST_PARAMS --allow-run-as-root --oversubscribe ${MPIEXEC_NUMPROC_FLAG} 4)
add_test(NAME cristian_algorithm COMMAND ${MPIEXEC} ${TEST_PARAMS} ./cristian_algorithm)
add_test(NAME berkeley_algorithm COMMAND ${MPIEXEC} ${TEST_PARAMS} ./berkeley_algorithm)
add_test(NAME ntp COMMAND ${MPIEXEC} ${TEST_PARAMS} ./ntp)
