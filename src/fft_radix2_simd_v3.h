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

    private:
        void FFTRadix2SIMDCore(T* real_out, T* imag_out, bool forward);

        const size_t m_size;
        std::vector<size_t> m_bit_reverse_idx;
        std::vector<std::vector<T>> m_sin;
        std::vector<std::vector<T>> m_cos;
    };

    template <typename T>
    FFTRadix2SIMD<T>::FFTRadix2SIMD(size_t size)
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
    template <typename T> FFTRadix2SIMD<T>::~FFTRadix2SIMD() {}

    template <typename T>
    void FFTRadix2SIMD<T>::Forward(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = real_in[m_bit_reverse_idx[i]];
            imag_out[i] = imag_in[m_bit_reverse_idx[i]];
        }
        FFTRadix2SIMDCore(real_out, imag_out, true);
    }

    template <typename T>
    void FFTRadix2SIMD<T>::Inverse(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = real_in[m_bit_reverse_idx[i]];
            imag_out[i] = imag_in[m_bit_reverse_idx[i]];
        }
        FFTRadix2SIMDCore(real_out, imag_out, false);
    }

    template <typename T>
    void FFTRadix2SIMD<T>::FFTRadix2SIMDCore(T* real_out, T* imag_out, bool forward)
    {
        for (int btfly = 2, step = 1; btfly <= m_size; btfly *= 2, step *= 2)
        {
            int m = m_size / btfly;
            if (step < 2)
            {
                for (int i = 0; i < m_size; i += btfly)
                {
                    int even = i;
                    int odd = even + 1;
                    __m128 odd_twiddle_4x =
                        _mm_set_ps(real_out[odd], real_out[odd], imag_out[odd], imag_out[odd]);
                    __m128 even_4x = _mm_set_ps(real_out[even], real_out[even],
                                                      imag_out[even], imag_out[even]);
                    __m128 res_4x = _mm_addsub_ps(even_4x, odd_twiddle_4x);

                    float* p = (float*)&res_4x;
                    real_out[odd] = p[2];
                    imag_out[odd] = p[0];
                    real_out[even] = p[3];
                    imag_out[even] = p[1];
                }
            }
            else if (step < 4)
            {
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
            else
            {
                for (int i = 0; i < m_size; i += btfly)
                {
                    for (int k = 0; k < step; k += 4)
                    {
                        int even = i + k;
                        int odd = even + step;

                        __m128 real_out_odd_4x = _mm_load_ps(real_out + odd);
                        __m128 imag_out_odd_4x = _mm_load_ps(imag_out + odd);
                        __m128 real_out_even_4x = _mm_load_ps(real_out + even);
                        __m128 imag_out_even_4x = _mm_load_ps(imag_out + even);
                        // __m128 cos_4x = _mm_set_ps(m_cos[k+3][m], m_cos[k+2][m],
                        // m_cos[k+1][m], m_cos[k][m]);
                        // __m128 sin_4x = _mm_set_ps(m_sin[k+3][m], m_sin[k+2][m],
                        // m_sin[k+1][m], m_sin[k][m]);
                        __m128 cos_4x = _mm_load_ps(&m_cos[m][k]);
                        __m128 sin_4x = _mm_load_ps(&m_sin[m][k]);

                        __m128 ac_4x = _mm_mul_ps(real_out_odd_4x, cos_4x);
                        __m128 bd_4x = _mm_mul_ps(imag_out_odd_4x, sin_4x);
                        __m128 ad_4x = _mm_mul_ps(real_out_odd_4x, sin_4x);
                        __m128 bc_4x = _mm_mul_ps(imag_out_odd_4x, cos_4x);

                        __m128 odd_twiddle_real_4x;
                        __m128 odd_twiddle_imag_4x;
                        if (forward)
                        {
                            odd_twiddle_real_4x = _mm_add_ps(ac_4x, bd_4x);
                            odd_twiddle_imag_4x = _mm_sub_ps(bc_4x, ad_4x);
                        }
                        else
                        {
                            odd_twiddle_real_4x = _mm_sub_ps(ac_4x, bd_4x);
                            odd_twiddle_imag_4x = _mm_add_ps(bc_4x, ad_4x);
                        }

                        real_out_odd_4x =
                            _mm_sub_ps(real_out_even_4x, odd_twiddle_real_4x);
                        imag_out_odd_4x =
                            _mm_sub_ps(imag_out_even_4x, odd_twiddle_imag_4x);
                        real_out_even_4x =
                            _mm_add_ps(real_out_even_4x, odd_twiddle_real_4x);
                        imag_out_even_4x =
                            _mm_add_ps(imag_out_even_4x, odd_twiddle_imag_4x);

                        _mm_store_ps(real_out + odd, real_out_odd_4x);
                        _mm_store_ps(imag_out + odd, imag_out_odd_4x);
                        _mm_store_ps(real_out + even, real_out_even_4x);
                        _mm_store_ps(imag_out + even, imag_out_even_4x);
                    }
                }
            }
        }
    }
} // namespace learnfft
