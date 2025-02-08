# LearnFFT
Learn to implement FFT step by step using C++.

# Tutorial
### English
  - [DFT Implementation](docs/DFT.md)
  - FFT Implementation (Radix-2)
  - FFT Implementation using SIMD
  - FFT Implementation using CUDA

### Chinese
  - [DFT实现](docs/DFT_cn.md)
  - [FFT实现（基2FFT）](docs/FFT_Radix2_cn.md)
  - 使用SIMD实现FFT
  - 使用CUDA实现FFT

# Test
```cpp
cd tests

git clone https://github.com/mborgerding/kissfft.git
git clone https://bitbucket.org/jpommier/pffft.git

mkdir build && cd build
cmake ..
make
./test_dft
./test_fft
```
