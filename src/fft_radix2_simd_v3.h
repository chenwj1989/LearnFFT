/*
 * Copyright (c) 2025, wjchen, BSD 3-Clause License
 */
#pragma once

#include "learn_fft_utils.h"
#include <immintrin.h>
#include <iostream>
#include <math.h>
#include <vector>

namespace learnfft
{
    template <typename T> class FFTRadix2SIMD
    {
    public:
        FFTRadix2SIMD(size_t size);
        ~FFTRadix2SIMD();

        void Forward(const T* real_in, const T* imag_in, T* real_out, T* imag_out);
        void Inverse(const T* real_in, const T* imag_in, T* real_out, T* imag_out);
        void Forward(const T* data_in, T* data_out);
        void Inverse(const T* data_in, T* data_out);

    private:
        void FFTRadix2SIMDInterleaved(T* data, bool forward);

        const size_t m_size;
        std::vector<size_t> m_bit_reverse_idx;
        std::vector<std::vector<T>> m_sin;
        std::vector<std::vector<T>> m_cos;
        std::vector<T> m_cpx_data;
    };

    template <typename T>
    FFTRadix2SIMD<T>::FFTRadix2SIMD(size_t size)
        : m_size(size), m_bit_reverse_idx(size), m_sin(size, std::vector<T>(size)),
          m_cos(size, std::vector<T>(size)), m_cpx_data(size * 2)
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
    template <typename T> FFTRadix2SIMD<T>::~FFTRadix2SIMD() {}

    template <typename T>
    void FFTRadix2SIMD<T>::Forward(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            m_cpx_data[i * 2] = real_in[m_bit_reverse_idx[i]];
            m_cpx_data[i * 2 + 1] = imag_in[m_bit_reverse_idx[i]];
        }
        FFTRadix2SIMDInterleaved(m_cpx_data.data(), true);
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = m_cpx_data[i * 2];
            imag_out[i] = m_cpx_data[i * 2 + 1];
        }
    }

    template <typename T>
    void FFTRadix2SIMD<T>::Inverse(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            m_cpx_data[i] = real_in[m_bit_reverse_idx[i]];
            m_cpx_data[i] = imag_in[m_bit_reverse_idx[i]];
        }
        FFTRadix2SIMDInterleaved(m_cpx_data.data(), false);
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = m_cpx_data[i * 2];
            imag_out[i] = m_cpx_data[i * 2 + 1];
        }
    }

    template <typename T> void FFTRadix2SIMD<T>::Forward(const T* data_in, T* data_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            data_out[i * 2] = data_in[m_bit_reverse_idx[i] * 2];         // real
            data_out[i * 2 + 1] = data_in[m_bit_reverse_idx[i] * 2 + 1]; // imag
        }
        FFTRadix2SIMD(data_out, true);
    }

    template <typename T> void FFTRadix2SIMD<T>::Inverse(const T* data_in, T* data_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            data_out[i * 2] = data_in[m_bit_reverse_idx[i] * 2];         // real
            data_out[i * 2 + 1] = data_in[m_bit_reverse_idx[i] * 2 + 1]; // imag
        }
        FFTRadix2SIMDInterleaved(data_out, false);
    }

    template <typename T> void FFTRadix2SIMD<T>::FFTRadix2SIMDInterleaved(T* cpx_data, bool forward)
    {
        for (int btfly = 2, step = 1; btfly <= m_size; btfly *= 2, step *= 2)
        {
            int m = m_size / btfly;

            if (step < 2)
            {
                for (int i = 0; i < m_size; i += btfly)
                {
                    int even = i * 2;
                    int odd = even + 2;
                    __m128d odd_twiddle_128d = _mm_load_pd(cpx_data + odd);
                    __m128d even_128d = _mm_load_pd(cpx_data + even);
                    __m128d odd_128d = _mm_add_pd(even_128d, odd_twiddle_128d);
                    even_128d = _mm_sub_pd(even_128d, odd_twiddle_128d);
                    _mm_store_pd(cpx_data + odd, odd_128d);
                    _mm_store_pd(cpx_data + even, even_128d);
                }
            }
            else
            {
                for (int i = 0; i < m_size; i += btfly)
                {
                    for (int k = 0, even = 0, odd = step * 2; k < step; k += 2, even += 4, odd += 4)
                    {
                        __m256d odd_256d = _mm256_load_pd(cpx_data + odd);   // | r | i | r | i |
                        __m256d even_256d = _mm256_load_pd(cpx_data + even); // | r | i | r | i |
                        __m128d cos_128d = _mm_load_pd(&m_cos[m][k]);
                        __m128d sin_128d = _mm_load_pd(&m_sin[m][k]);
                        __m256d cos_256d = _mm256_set_m128d(cos_128d, cos_128d);
                        __m256d sin_256d = _mm256_set_m128d(sin_128d, sin_128d);

                        // __m256d ac_bd_256d = _mm256_mul_pd(odd_256d, cd_256d);
                        // __m256d ad_bc_256d = _mm256_mul_pd(odd_256d, dc_256d);
                        __m256d ac_bc_256d = _mm256_mul_pd(odd_256d, cos_256d);
                        __m256d ad_bd_256d = _mm256_mul_pd(odd_256d, sin_256d);
                        __m256d odd_twiddle_256d;

                        if (forward)
                        {
                            odd_twiddle_256d = _mm256_addsub_pd(ac_bc_256d, ad_bd_256d);
                            // odd_twiddle_256d = _mm256_addsub_pd(ac_bd_256d, ad_bc_256d);
                        }
                        else
                        {
                            odd_twiddle_256d = _mm256_addsub_pd(ad_bd_256d, ac_bc_256d);
                            // odd_twiddle_256d = _mm256_addsub_pd(ad_bc_256d, ac_bd_256d);
                        }

                        odd_256d = _mm256_sub_pd(even_256d, odd_twiddle_256d);
                        even_256d = _mm256_add_pd(even_256d, odd_twiddle_256d);

                        _mm256_store_pd(cpx_data + odd, odd_256d);
                        _mm256_store_pd(cpx_data + even, even_256d);
                    }
                }
            }
        }
    }
} // namespace learnfft
