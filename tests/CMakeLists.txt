cmake_minimum_required(VERSION 3.8)
project(learnfft)

# C++ standard
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})

add_definitions(-Dlearn_fft_scalar=double)
add_definitions(-Dkiss_fft_scalar=double)

SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O2 -Wall -fPIC -mavx -march=native")
SET(CMAKE_C_FLAGS "$ENV{CFLAGS} -O2 -Wall -fPIC -mavx -march=native")

include_directories(../src)
include_directories(kissfft/)

file(GLOB learnfft_srcs ../src/*.cpp)
# add_library(learnfft STATIC ${learnfft_srcs})

file(GLOB dep_srcs  kissfft/kiss_fft.c
                    kissfft/kiss_fftr.c
                    pffft/pffft.c)

add_executable(test_dft test_dft.cpp ${learnfft_srcs} ${dep_srcs})
target_link_libraries(test_dft PRIVATE learnfft)

add_executable(test_fft test_fft.cpp ${learnfft_srcs} ${dep_srcs})
target_link_libraries(test_fft PRIVATE learnfft)

cuda_add_executable(test_cuda test_cuda.cu)
target_link_libraries(test_cuda ${CUDA_LIBRARIES})

file(GLOB learnfft_cu  ../src/*.cu)
cuda_add_executable(test_fft_cuda test_fft_cuda.cpp ${learnfft_cu})
target_link_libraries(test_fft_cuda learnfft ${CUDA_LIBRARIES})
