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
    template <typename T> class FFTRadix2
    {
    public:
        FFTRadix2(size_t size);
        ~FFTRadix2();

        void Forward(const T* real_in, const T* imag_in, T* real_out, T* imag_out);
        void Inverse(const T* real_in, const T* imag_in, T* real_out, T* imag_out);

    private:
        void FFTRadix2Core(T* real_out, T* imag_out, bool forward);

        const size_t m_size;
        std::vector<size_t> m_bit_reverse_idx;
        std::vector<std::vector<T>> m_sin;
        std::vector<std::vector<T>> m_cos;
    };

    template <typename T>
    FFTRadix2<T>::FFTRadix2(size_t size)
        : m_size(size), m_bit_reverse_idx(size), m_sin(size, std::vector<T>(size)),
          m_cos(size, std::vector<T>(size))
    {
        for (int i = 0; i < m_size; ++i)
        {
            for (int j = 0; j < m_size; ++j)
            {
                double arg = (double(i) * double(j) * M_PI * 2.0) / m_size;
                m_sin[i][j] = sin(arg);
                m_cos[i][j] = cos(arg);
            }
        }
        GenBitReverseOrder(m_size, m_bit_reverse_idx);
    }
    template <typename T> FFTRadix2<T>::~FFTRadix2() {}

    template <typename T>
    void FFTRadix2<T>::Forward(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = real_in[m_bit_reverse_idx[i]];
            imag_out[i] = imag_in[m_bit_reverse_idx[i]];
        }
        FFTRadix2Core(real_out, imag_out, true);
    }

    template <typename T>
    void FFTRadix2<T>::Inverse(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = real_in[m_bit_reverse_idx[i]];
            imag_out[i] = imag_in[m_bit_reverse_idx[i]];
        }
        FFTRadix2Core(real_out, imag_out, false);
    }

    template <typename T> void FFTRadix2<T>::FFTRadix2Core(T* real_out, T* imag_out, bool forward)
    {
        for (int btfly = 2, step = 1; btfly <= m_size; btfly *= 2, step *= 2)
        {
            int m = m_size / btfly;
            for (int i = 0; i < m_size; i += btfly)
            {
                for (int k = 0; k < step; ++k)
                {
                    int even = i + k;
                    int odd = even + step;
                    T odd_twiddle_real;
                    T odd_twiddle_imag;
                    if (forward)
                    {
                        odd_twiddle_real =
                            T(real_out[odd] * m_cos[k][m] + imag_out[odd] * m_sin[k][m]);
                        odd_twiddle_imag =
                            T(-real_out[odd] * m_sin[k][m] + imag_out[odd] * m_cos[k][m]);
                    }
                    else
                    {
                        odd_twiddle_real =
                            T(real_out[odd] * m_cos[k][m] - imag_out[odd] * m_sin[k][m]);
                        odd_twiddle_imag =
                            T(real_out[odd] * m_sin[k][m] + imag_out[odd] * m_cos[k][m]);
                    }

                    real_out[odd] = real_out[even] - odd_twiddle_real;
                    imag_out[odd] = imag_out[even] - odd_twiddle_imag;
                    real_out[even] = real_out[even] + odd_twiddle_real;
                    imag_out[even] = imag_out[even] + odd_twiddle_imag;
                }
            }
        }
    }
} // namespace learnfft
