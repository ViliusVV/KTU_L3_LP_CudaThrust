cmake_minimum_required(VERSION 3.12)
project(simple_cuda)


# project name
project(l2-cuda_a)

set(CMAKE_CXX_STANDARD 11)
enable_language(CUDA)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -pedantic -fpermissive -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fpermissive")
# include dirs
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
include_directories(../include/)
include_directories(../include/nlohmann)

# puts all .cpp files inside src to the SOURCES variable
file(GLOB SOURCES ../src/main2.cu ../include/person.cpp)
# outputs executable binary file to bin folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

# # specify cuda compiler
# SET(CMAKE_C_COMPILER nvcc)
# SET(CMAKE_CXX_COMPILER nvcc)

# compiles the files defined by SOURCES to generante the executable defined by EXEC
add_executable(main ${SOURCES})