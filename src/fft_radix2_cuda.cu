/*
 * Copyright (c) 2025, wjchen, BSD 3-Clause License
 */
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <vector>
#include "fft_radix2_cuda.h"

namespace learnfft
{
    __global__ void kernelFFTRadix2(double* real_data, double* imag_data, bool forward, int btfly, int len)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= len) return;
        
        int m = len / btfly;
        int step = btfly / 2;
        int k = i % btfly;
        if (k >= step) return;

        int even = i;
        int odd = even + step;

        double arg = (double(k) * double(m) * M_PI * 2.0) / len;
        double _sin = sin(arg);
        double _cos = cos(arg);

        double odd_twiddle_real;
        double odd_twiddle_imag;
        if (forward)
        {
            odd_twiddle_real =
                double(real_data[odd] * _cos + imag_data[odd] * _sin);
            odd_twiddle_imag =
                double(-real_data[odd] * _sin + imag_data[odd] * _cos);
        }
        else
        {
            odd_twiddle_real =
                double(real_data[odd] * _cos - imag_data[odd] * _sin);
            odd_twiddle_imag =
                double(real_data[odd] * _sin + imag_data[odd] * _cos);
        }

        real_data[odd] = real_data[even] - odd_twiddle_real;
        imag_data[odd] = imag_data[even] - odd_twiddle_imag;
        real_data[even] = real_data[even] + odd_twiddle_real;
        imag_data[even] = imag_data[even] + odd_twiddle_imag;
    }

    class FFTCUDAImpl
    {
    public:
        FFTCUDAImpl(size_t size);
        ~FFTCUDAImpl();

        void FFTRadix2CUDA(double* real_data, double* imag_data, bool forward);

    private:
        size_t m_size;
        double *m_dev_real;
        double *m_dev_imag;

    };

    FFTCUDAImpl::FFTCUDAImpl(size_t size) : m_size(size)
    {
        int n_bytes = m_size * sizeof(double);
        cudaMalloc((void**)&m_dev_real, n_bytes);
        cudaMalloc((void**)&m_dev_imag, n_bytes);
    }

    FFTCUDAImpl::~FFTCUDAImpl() {
        cudaFree(m_dev_real);
        cudaFree(m_dev_imag);
    }

    void FFTCUDAImpl::FFTRadix2CUDA(double* real_data, double* imag_data, bool forward)
    {

        int n_bytes = m_size * sizeof(double);
        cudaMemcpy((void*)m_dev_real, (void*)real_data, n_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)m_dev_imag, (void*)imag_data, n_bytes, cudaMemcpyHostToDevice);
        int tx = 1024;
        int bx = (m_size + tx - 1) / tx;
        const dim3 blockSize(tx);
        const dim3 gridSize(bx);
        for (int btfly = 2; btfly <= m_size; btfly *= 2)
        {
            kernelFFTRadix2<<<gridSize, blockSize>>>(m_dev_real, m_dev_imag, forward, btfly, m_size);
            cudaDeviceSynchronize();
        }
        cudaMemcpy((void*)real_data, (void*)m_dev_real, n_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy((void*)imag_data, (void*)m_dev_imag, n_bytes, cudaMemcpyDeviceToHost);
    }
    
    FFTRadix2CUDA::FFTRadix2CUDA(size_t size)
        : m_size(size), m_bit_reverse_idx(size)
    {
        GenBitReverseOrder(m_size, m_bit_reverse_idx);
        m_impl = new FFTCUDAImpl(size);
    }
     FFTRadix2CUDA::~FFTRadix2CUDA() {
        delete m_impl;
    }

    void FFTRadix2CUDA::Forward(const double* real_in, const double* imag_in, double* real_out, double* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = real_in[m_bit_reverse_idx[i]];
            imag_out[i] = imag_in[m_bit_reverse_idx[i]];
        }
        m_impl->FFTRadix2CUDA(real_out, imag_out, true);
    }

    
    void FFTRadix2CUDA::Inverse(const double* real_in, const double* imag_in, double* real_out, double* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = real_in[m_bit_reverse_idx[i]];
            imag_out[i] = imag_in[m_bit_reverse_idx[i]];
        }
        m_impl->FFTRadix2CUDA(real_out, imag_out, false);
    }

} // namespace learnfft
