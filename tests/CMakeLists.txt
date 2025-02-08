cmake_minimum_required(VERSION 3.10)
project(learnfft)

# C++ standard
set(CMAKE_CXX_STANDARD 17)


add_definitions(-Dlearn_fft_scalar=double)
add_definitions(-Dkiss_fft_scalar=double)

SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O2 -Wall -fPIC")
SET(CMAKE_C_FLAGS "$ENV{CFLAGS} -O2 -Wall -fPIC")

include_directories(../src)
include_directories(kissfft/)

file(GLOB learnfft_srcs ../src/*.cpp)
# add_library(learnfft STATIC ${learnfft_srcs})

file(GLOB dep_srcs  kissfft/kiss_fft.c
                    kissfft/kiss_fftr.c
                    pffft/pffft.c)

add_executable(test_dft test_dft.cpp ${learnfft_srcs} ${dep_srcs})

add_executable(test_fft test_fft.cpp ${learnfft_srcs} ${dep_srcs})

add_executable(test_fft_simd test_fft_simd.cpp ${learnfft_srcs} ${dep_srcs})
target_compile_options(test_fft_simd PRIVATE -mavx )

add_executable(test_avx test_avx.cpp ${learnfft_srcs} ${dep_srcs})
target_compile_options(test_avx PRIVATE -mavx )

find_package(CUDA)
IF (CUDA_VERSION_STRING)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_executable(test_cuda test_cuda.cu)
target_link_libraries(test_cuda ${CUDA_LIBRARIES})

file(GLOB learnfft_cu  ../src/*.cu)
cuda_add_executable(test_fft_cuda test_fft_cuda.cpp ${learnfft_cu})
target_link_libraries(test_fft_cuda learnfft ${CUDA_LIBRARIES})
ENDIF ()
