cmake_minimum_required(VERSION 3.30)
project(HPC_Assignment1_Problem1 LANGUAGES C CXX)
include_directories(../Common)
find_package(OpenMP REQUIRED)

add_executable(knn
        knn.cpp
)

target_link_libraries(knn PUBLIC OpenMP::OpenMP_CXX)
