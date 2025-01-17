/*
 * Copyright (c) 2025, wjchen, BSD 3-Clause License
 */
#pragma once

#include <iostream>
#include <math.h>
#include <vector>

namespace learnfft
{
    template <typename T> class FFTRecursive
    {
    public:
        FFTRecursive(size_t size);
        ~FFTRecursive();

        void Forward(const T* real_in, const T* imag_in, T* real_out, T* imag_out);
        void Inverse(const T* real_in, const T* imag_in, T* real_out, T* imag_out);

    private:
        void FFTCoreRecursive(const int len, T* real_in, T* imag_in, T* real_out, T* imag_out,
                              bool forward);
        void DeInterleave(const int len, T* data);

        const size_t m_size;
        std::vector<std::vector<T>> m_sin;
        std::vector<std::vector<T>> m_cos;
        std::vector<T> m_tmp_real;
        std::vector<T> m_tmp_imag;
        std::vector<T> m_tmp;
    };

    template <typename T>
    FFTRecursive<T>::FFTRecursive(size_t size)
        : m_size(size), m_sin(size, std::vector<T>(size)), m_cos(size, std::vector<T>(size)),
          m_tmp_real(size), m_tmp_imag(size), m_tmp(size)
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
    }
    template <typename T> FFTRecursive<T>::~FFTRecursive() {}

    template <typename T>
    void FFTRecursive<T>::Forward(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        memcpy(m_tmp_real.data(), real_in, sizeof(T) * m_size);
        memcpy(m_tmp_imag.data(), imag_in, sizeof(T) * m_size);
        FFTCoreRecursive(m_size, m_tmp_real.data(), m_tmp_imag.data(), real_out, imag_out, true);
    }

    template <typename T>
    void FFTRecursive<T>::Inverse(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        memcpy(m_tmp_real.data(), real_in, sizeof(T) * m_size);
        memcpy(m_tmp_imag.data(), imag_in, sizeof(T) * m_size);
        FFTCoreRecursive(m_size, m_tmp_real.data(), m_tmp_imag.data(), real_out, imag_out, false);
    }

    template <typename T> void FFTRecursive<T>::DeInterleave(const int len, T* data)
    {
        int half_len = len / 2;
        for (int i = 0; i < half_len; i++)
        {
            data[i] = data[i * 2];
            m_tmp[i] = data[i * 2 + 1];
        }
        for (int i = 0; i < half_len; i++)
        {
            data[i + half_len] = m_tmp[i];
        }
    }

    template <typename T>
    void FFTRecursive<T>::FFTCoreRecursive(const int len, T* real_in, T* imag_in, T* real_out,
                                           T* imag_out, bool forward)
    {
        if (len == 1)
        {
            real_out[0] = T(real_in[0]);
            imag_out[0] = T(imag_in[0]);
        }
        else
        {
            int half_len = len / 2;
            int m = m_size / len;
            DeInterleave(len, real_in);
            DeInterleave(len, imag_in);
            T* even_in_real = real_in;
            T* even_in_imag = imag_in;
            T* odd_in_real = real_in + half_len;
            T* odd_in_imag = imag_in + half_len;
            T* even_out_real = real_out;
            T* even_out_imag = imag_out;
            T* odd_out_real = real_out + half_len;
            T* odd_out_imag = imag_out + half_len;

            FFTCoreRecursive(half_len, even_in_real, even_in_imag, even_out_real, even_out_imag,
                             forward);
            FFTCoreRecursive(half_len, odd_in_real, odd_in_imag, odd_out_real, odd_out_imag,
                             forward);
            for (int k = 0; k < half_len; ++k)
            {
                T odd_twiddle_real;
                T odd_twiddle_imag;
                if (forward)
                {
                    odd_twiddle_real =
                        T(odd_out_real[k] * m_cos[k][m] + odd_out_imag[k] * m_sin[k][m]);
                    odd_twiddle_imag =
                        T(-odd_out_real[k] * m_sin[k][m] + odd_out_imag[k] * m_cos[k][m]);
                }
                else
                {
                    odd_twiddle_real =
                        T(odd_out_real[k] * m_cos[k][m] - odd_out_imag[k] * m_sin[k][m]);
                    odd_twiddle_imag =
                        T(odd_out_real[k] * m_sin[k][m] + odd_out_imag[k] * m_cos[k][m]);
                }

                real_out[k + half_len] = even_out_real[k] - odd_twiddle_real;
                imag_out[k + half_len] = even_out_imag[k] - odd_twiddle_imag;
                real_out[k] = even_out_real[k] + odd_twiddle_real;
                imag_out[k] = even_out_imag[k] + odd_twiddle_imag;
            }
        }
    }
} // namespace learnfft