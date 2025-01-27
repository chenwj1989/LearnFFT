/*
 * Copyright (c) 2025, wjchen, BSD 3-Clause License
 */
#pragma once

#include <iostream>
#include <math.h>
#include <vector>
#include <immintrin.h>
#include "learn_fft_utils.h"

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
        std::vector<T> m_data;
    };

    template <typename T>
    FFTRadix2SIMD<T>::FFTRadix2SIMD(size_t size)
        : m_size(size), m_bit_reverse_idx(size), m_sin(size, std::vector<T>(size)),
          m_cos(size, std::vector<T>(size)), m_data(size)
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
            m_data[i * 2] = real_in[m_bit_reverse_idx[i]];
            m_data[i * 2 + 1] = imag_in[m_bit_reverse_idx[i]];
        }
        FFTRadix2SIMDInterleaved(m_data.data(), true);
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = m_data[i * 2];
            imag_out[i] = m_data[i * 2 + 1];
        }
    }

    template <typename T>
    void FFTRadix2SIMD<T>::Inverse(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            m_data[i] = real_in[m_bit_reverse_idx[i]];
            m_data[i] = imag_in[m_bit_reverse_idx[i]];
        }
        FFTRadix2SIMDInterleaved(m_data.data(), false);
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = m_data[i * 2];
            imag_out[i] = m_data[i * 2 + 1];
        }
    }

    template <typename T>
    void FFTRadix2SIMD<T>::Forward(const T* data_in, T* data_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            data_out[i * 2] = data_in[m_bit_reverse_idx[i] * 2]; // real
            data_out[i * 2 + 1] = data_in[m_bit_reverse_idx[i] * 2 + 1]; // imag
        }
        FFTRadix2SIMDInterleaved(data_out, true);
    }

    template <typename T>
    void FFTRadix2SIMD<T>::Inverse(const T* data_in, T* data_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            data_out[i * 2] = data_in[m_bit_reverse_idx[i] * 2]; // real
            data_out[i * 2 + 1] = data_in[m_bit_reverse_idx[i] * 2 + 1]; // imag
        }
        FFTRadix2SIMDInterleaved(data_out, false);
    }

    template <typename T>
    void FFTRadix2SIMD<T>::FFTRadix2SIMDInterleaved(T* data, bool forward)
    {
        for (int btfly = 2, step = 1; btfly <= m_size; btfly *= 2, step *= 2)
        {
            int m = m_size / btfly;
            for (int i = 0; i < m_size; i += btfly)
            {
                if (step < 2) {
                    int k = 0;
                    int even = i;
                    int even_r = even * 2;
                    int even_i = even_r + 1;
                    int odd = even_i + 1;
                    int odd_r = odd * 2;
                    int odd_i = odd_r + 1;

                    __m256d odd_256d;
                    __m256d trig_256d;
                    __m256d res_256d;
                    if (forward)
                    {
                        trig_256d = _mm256_set_pd(m_cos[k][m], m_sin[k][m], -m_sin[k][m], m_cos[k][m]);
                    }
                    else
                    {
                        trig_256d = _mm256_set_pd(m_cos[k][m], -m_sin[k][m], m_sin[k][m], m_cos[k][m]);
                    }
                    odd_256d = _mm256_set_pd(data[odd_r], data[odd_i], data[odd_r], data[odd_i]);
                    res_256d = _mm256_mul_pd(odd_256d, trig_256d);
                    __m256d odd_twiddle_256d = _mm256_hadd_pd(res_256d, res_256d);
                    __m256d even_256d = _mm256_set_pd(data[even_r], data[even_r], data[even_i], data[even_i]);
                    res_256d = _mm256_addsub_pd(even_256d, odd_twiddle_256d);
                    
                    double* p = (double*)&res_256d;	
                    data[odd_r] = p[2];
                    data[odd_i] = p[0];
                    data[even_r] = p[3];
                    data[even_i] = p[1];
                       
                } else {
                    for (int k = 0; k < step; k += 2)
                    {
                        int even = i + k * 2;
                        int odd = even + step * 2;

                         __m256d out_odd_256d = _mm256_load_pd(data + odd);  // | r | i | r | i |
                         __m256d out_even_256d = _mm256_load_pd(data + even);  // | r | i | r | i |
                         __m256d cos_256d = _mm256_set_pd(m_cos[m][k+1], m_cos[m][k+1], m_cos[m][k], m_cos[m][k]);
                         __m256d sin_256d = _mm256_set_pd( m_sin[m][k+1], m_sin[m][k+1], m_sin[m][k], m_sin[m][k]);

                         __m256d ac_bc_256d = _mm256_mul_pd(out_odd_256d, cos_256d);
                         __m256d ad_bd_256d = _mm256_mul_pd(out_odd_256d, sin_256d);
                    
                         
                        //  __m256d cd_256d = _mm256_set_pd(m_cos[m][k+1], m_sin[m][k+1], m_cos[m][k], m_sin[m][k]);
                        //  __m256d dc_256d = _mm256_set_pd(m_sin[m][k+1], m_cos[m][k+1], m_sin[m][k], m_cos[m][k]);

                        //  __m256d ac_bd_256d = _mm_mul_pd(out_odd_256d, cd_256d);
                        //  __m256d ad_bc_256d = _mm_mul_pd(out_odd_256d, dc_256d);
                    
                        __m256d odd_twiddle_256d;

                        if (forward)
                        {
                            // odd_twiddle_256d = _mm256_addsub_pd(ac_bc_256d, bd_ad_256d);
                            odd_twiddle_256d = _mm256_addsub_pd(ac_bc_256d, ad_bd_256d);
                        }
                        else
                        {
                            // odd_twiddle_256d = _mm256_addsub_pd(ad_bd_256d, bc_ac_256d);
                            odd_twiddle_256d = _mm256_addsub_pd(ad_bd_256d, ac_bc_256d);
                        }

                        out_odd_256d = _mm256_sub_pd(out_even_256d, odd_twiddle_256d);
                        out_even_256d = _mm256_add_pd(out_even_256d, odd_twiddle_256d);

                        _mm256_store_pd(data + odd, out_odd_256d);
                        _mm256_store_pd(data + even, out_even_256d);

                    }
                }

            }
        }
    }
} // namespace learnfft
