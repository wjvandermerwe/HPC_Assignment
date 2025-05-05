cmake_minimum_required(VERSION 3.30)
project(HPC_Assignment1_Project LANGUAGES C CXX CUDA)
include_directories(../Common)
# set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(convolution
        conv.cu
)

target_compile_options(convolution PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
#set_target_properties(convolution PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON
#)
set_target_properties(convolution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # put the .exe into build/Problem2 instead of build root
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Problem2"
    # if you ever do multi-config generators, you can also set
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Problem2"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Problem2"
)

# copy data files
add_custom_command(TARGET convolution POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/data"
        "$<TARGET_FILE_DIR:convolution>/data"
)
