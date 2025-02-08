/*
 * Copyright (c) 2025, wjchen, BSD 3-Clause License
 */
#pragma once

#include "learn_fft_utils.h"
#include <iostream>
#include <math.h>
#include <vector>

namespace learnfft
{
    class FFTCUDAImpl;
    class FFTRadix2CUDA
    {
    public:
        FFTRadix2CUDA(size_t size);
        ~FFTRadix2CUDA();

        void Forward(const double* real_in, const double* imag_in, double* real_out,
                     double* imag_out);
        void Inverse(const double* real_in, const double* imag_in, double* real_out,
                     double* imag_out);

    private:
        const size_t m_size;
        std::vector<size_t> m_bit_reverse_idx;
        FFTCUDAImpl* m_impl;
    };

} // namespace learnfft
