cmake_minimum_required(VERSION 3.24)
project(mpi_kmeans)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fopenmp")

add_executable(mpi_kmeans main.cpp)

target_link_libraries(trash ${MPI_CXX_LIBRARIES})