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
    class FFTRadix2CUDA
    {
    public:
        FFTRadix2CUDA(size_t size);
        ~FFTRadix2CUDA();

        void Forward(const double* real_in, const double* imag_in, double* real_out, double* imag_out);
        void Inverse(const double* real_in, const double* imag_in, double* real_out, double* imag_out);

    private:
        void FFTRadix2Core(double* real_out, double* imag_out, bool forward);

        const size_t m_size;
        std::vector<size_t> m_bit_reverse_idx;
        std::vector<std::vector<double>> m_sin;
        std::vector<std::vector<double>> m_cos;
    };
} // namespace learnfft
