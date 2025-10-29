cmake_minimum_required(VERSION 3.5)  
PROJECT(range_lib)  
  
file(GLOB SRCS   
main.cpp   
"vendor/lodepng/lodepng.cpp" )  
  
option(WITH_CUDA "Compile CUDA ray cast varients" ON)  
set(CHUNK_SIZE 262144)  
set(CHUNK_THREADS 256)  
  
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")  
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")  
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")  
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)  
  
SET(BASEPATH "${CMAKE_SOURCE_DIR}")   
add_definitions ( -DBASEPATH=${BASEPATH} )  
add_subdirectory(${BASEPATH}/vendor/gflags)  
  
include_directories(${BASEPATH})  
  
include(CheckCXXCompilerFlag)  
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)  
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)  
if(COMPILER_SUPPORTS_CXX11)  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")  
elseif(COMPILER_SUPPORTS_CXX0X)  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")  
else()  
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")  
endif()  
  
set(CMAKE_CXX_FLAGS "-march=native -ffast-math -fno-math-errno ${CMAKE_CXX_FLAGS} -O3")  
set(CMAKE_BUILD_TYPE Release)  
  
if (WITH_CUDA)  
add_definitions(-DUSE_CUDA=1)  
add_definitions(-DCHUNK_SIZE=${CHUNK_SIZE})  
add_definitions(-DNUM_THREADS=${CHUNK_THREADS})  
  
file(GLOB CUDA_SRCS   
main.cpp   
"includes/kernels.cu"  
"vendor/lodepng/lodepng.cpp" )  
  
find_package(CUDA REQUIRED)  
cuda_add_executable(range_lib ${CUDA_SRCS})  
else()  
add_definitions(-DUSE_CUDA=0)  
add_executable(range_lib ${SRCS})  
endif()  
  
target_link_libraries(range_lib gflags pthread)  
