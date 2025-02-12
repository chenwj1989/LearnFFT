/*
 * Copyright (c) 2025, wjchen, BSD 3-Clause License
 */
#pragma once

#include "learnfft_utils.h"
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
        T* m_sin;
        T* m_cos;
        T* m_real;
        T* m_imag;
    };

    template <typename T>
    FFTRadix2SIMD<T>::FFTRadix2SIMD(size_t size)
        : m_size(size), m_bit_reverse_idx(size), m_sin(nullptr), m_cos(nullptr), m_real(nullptr),
          m_imag(nullptr)
    {

        m_sin = (T*)aligned_alloc(64, size * size * sizeof(T));
        m_cos = (T*)aligned_alloc(64, size * size * sizeof(T));
        m_real = (T*)aligned_alloc(64, size * sizeof(T));
        m_imag = (T*)aligned_alloc(64, size * sizeof(T));

        size_t p = 0;
        for (int i = 0; i < m_size; ++i)
        {
            for (int j = 0; j < m_size; ++j, p++)
            {
                double arg = (double(i) * double(j) * M_PI * 2.0) / m_size;
                m_sin[p] = sin(arg);
                m_cos[p] = cos(arg);
            }
        }
        GenBitReverseOrder(m_size, m_bit_reverse_idx);
    }
    template <typename T> FFTRadix2SIMD<T>::~FFTRadix2SIMD()
    {
        free(m_sin);
        free(m_cos);
        free(m_real);
        free(m_imag);
    }

    template <typename T>
    void FFTRadix2SIMD<T>::Forward(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            m_real[i] = real_in[m_bit_reverse_idx[i]];
            m_imag[i] = imag_in[m_bit_reverse_idx[i]];
        }
        FFTRadix2SIMDCore(m_real, m_imag, true);
        // memcpy(real_out, m_real, m_size);
        // memcpy(imag_out, m_imag, m_size);
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = m_real[i];
            imag_out[i] = m_imag[i];
        }
    }

    template <typename T>
    void FFTRadix2SIMD<T>::Inverse(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            m_real[i] = real_in[m_bit_reverse_idx[i]];
            m_imag[i] = imag_in[m_bit_reverse_idx[i]];
        }
        FFTRadix2SIMDCore(m_real, m_imag, false);
        // memcpy(real_out, m_real, m_size);
        // memcpy(imag_out, m_imag, m_size);
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = m_real[i];
            imag_out[i] = m_imag[i];
        }
    }

    template <typename T>
    void FFTRadix2SIMD<T>::FFTRadix2SIMDCore(T* real_out, T* imag_out, bool forward)
    {
        for (int btfly = 2, step = 1; btfly <= m_size; btfly *= 2, step *= 2)
        {
            int m = m_size / btfly;
            size_t ptr = m * m_size;

            if (step < 2)
            {
                for (int i = 0; i < m_size; i += btfly)
                {
                    int even = i;
                    int odd = even + 1;
                    __m256d odd_twiddle_256d =
                        _mm256_set_pd(real_out[odd], real_out[odd], imag_out[odd], imag_out[odd]);
                    __m256d even_256d = _mm256_set_pd(real_out[even], real_out[even],
                                                      imag_out[even], imag_out[even]);
                    __m256d res_256d = _mm256_addsub_pd(even_256d, odd_twiddle_256d);

                    double* p = (double*)&res_256d;
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
                    for (int k = 0; k < step; k += 2)
                    {
                        int even = i + k;
                        int odd = even + step;

                        __m128d real_out_odd_128d = _mm_load_pd(real_out + odd);
                        __m128d imag_out_odd_128d = _mm_load_pd(imag_out + odd);
                        __m128d real_out_even_128d = _mm_load_pd(real_out + even);
                        __m128d imag_out_even_128d = _mm_load_pd(imag_out + even);
                        __m128d cos_128d = _mm_load_pd(&m_cos[ptr + k]);
                        __m128d sin_128d = _mm_load_pd(&m_sin[ptr + k]);

                        __m128d ac_128d = _mm_mul_pd(real_out_odd_128d, cos_128d);
                        __m128d bd_128d = _mm_mul_pd(imag_out_odd_128d, sin_128d);
                        __m128d ad_128d = _mm_mul_pd(real_out_odd_128d, sin_128d);
                        __m128d bc_128d = _mm_mul_pd(imag_out_odd_128d, cos_128d);

                        __m128d odd_twiddle_real_128d;
                        __m128d odd_twiddle_imag_128d;
                        if (forward)
                        {
                            odd_twiddle_real_128d = _mm_add_pd(ac_128d, bd_128d);
                            odd_twiddle_imag_128d = _mm_sub_pd(bc_128d, ad_128d);
                        }
                        else
                        {
                            odd_twiddle_real_128d = _mm_sub_pd(ac_128d, bd_128d);
                            odd_twiddle_imag_128d = _mm_add_pd(bc_128d, ad_128d);
                        }

                        real_out_odd_128d = _mm_sub_pd(real_out_even_128d, odd_twiddle_real_128d);
                        imag_out_odd_128d = _mm_sub_pd(imag_out_even_128d, odd_twiddle_imag_128d);
                        real_out_even_128d = _mm_add_pd(real_out_even_128d, odd_twiddle_real_128d);
                        imag_out_even_128d = _mm_add_pd(imag_out_even_128d, odd_twiddle_imag_128d);

                        _mm_store_pd(real_out + odd, real_out_odd_128d);
                        _mm_store_pd(imag_out + odd, imag_out_odd_128d);
                        _mm_store_pd(real_out + even, real_out_even_128d);
                        _mm_store_pd(imag_out + even, imag_out_even_128d);
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

                        __m256d real_out_odd_256d = _mm256_load_pd(real_out + odd);
                        __m256d imag_out_odd_256d = _mm256_load_pd(imag_out + odd);
                        __m256d real_out_even_256d = _mm256_load_pd(real_out + even);
                        __m256d imag_out_even_256d = _mm256_load_pd(imag_out + even);
                        // __m256d cos_256d = _mm256_set_pd(m_cos[k+3][m], m_cos[k+2][m],
                        // m_cos[k+1][m], m_cos[k][m]);
                        // __m256d sin_256d = _mm256_set_pd(m_sin[k+3][m], m_sin[k+2][m],
                        // m_sin[k+1][m], m_sin[k][m]);
                        __m256d cos_256d = _mm256_load_pd(&m_cos[ptr + k]);
                        __m256d sin_256d = _mm256_load_pd(&m_sin[ptr + k]);

                        __m256d ac_256d = _mm256_mul_pd(real_out_odd_256d, cos_256d);
                        __m256d bd_256d = _mm256_mul_pd(imag_out_odd_256d, sin_256d);
                        __m256d ad_256d = _mm256_mul_pd(real_out_odd_256d, sin_256d);
                        __m256d bc_256d = _mm256_mul_pd(imag_out_odd_256d, cos_256d);

                        __m256d odd_twiddle_real_256d;
                        __m256d odd_twiddle_imag_256d;
                        if (forward)
                        {
                            odd_twiddle_real_256d = _mm256_add_pd(ac_256d, bd_256d);
                            odd_twiddle_imag_256d = _mm256_sub_pd(bc_256d, ad_256d);
                        }
                        else
                        {
                            odd_twiddle_real_256d = _mm256_sub_pd(ac_256d, bd_256d);
                            odd_twiddle_imag_256d = _mm256_add_pd(bc_256d, ad_256d);
                        }

                        real_out_odd_256d =
                            _mm256_sub_pd(real_out_even_256d, odd_twiddle_real_256d);
                        imag_out_odd_256d =
                            _mm256_sub_pd(imag_out_even_256d, odd_twiddle_imag_256d);
                        real_out_even_256d =
                            _mm256_add_pd(real_out_even_256d, odd_twiddle_real_256d);
                        imag_out_even_256d =
                            _mm256_add_pd(imag_out_even_256d, odd_twiddle_imag_256d);

                        _mm256_store_pd(real_out + odd, real_out_odd_256d);
                        _mm256_store_pd(imag_out + odd, imag_out_odd_256d);
                        _mm256_store_pd(real_out + even, real_out_even_256d);
                        _mm256_store_pd(imag_out + even, imag_out_even_256d);
                    }
                }
            }
        }
    }
} // namespace learnfft
