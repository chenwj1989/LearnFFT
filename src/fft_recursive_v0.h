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
        void ForwardRecursive(const int len, const T* real_in, const T* imag_in, T* real_out,
                              T* imag_out);
        void InverseRecursive(const int len, const T* real_in, const T* imag_in, T* real_out,
                              T* imag_out);

        const size_t m_size;
        std::vector<std::vector<T>> m_sin;
        std::vector<std::vector<T>> m_cos;
    };

    template <typename T>
    FFTRecursive<T>::FFTRecursive(size_t size)
        : m_size(size), m_sin(size, std::vector<T>(size)), m_cos(size, std::vector<T>(size))
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
        ForwardRecursive(m_size, real_in, imag_in, real_out, imag_out);
    }

    template <typename T>
    void FFTRecursive<T>::Inverse(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        InverseRecursive(m_size, real_in, imag_in, real_out, imag_out);
    }

    template <typename T>
    void FFTRecursive<T>::ForwardRecursive(const int len, const T* real_in, const T* imag_in,
                                           T* real_out, T* imag_out)
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
            std::vector<T> tmp_real(len);
            std::vector<T> tmp_imag(len);
            for (int i = 0; i < half_len; i++)
            {
                tmp_real[i] = real_in[i * 2];
                tmp_imag[i] = imag_in[i * 2];
                tmp_real[i + half_len] = real_in[i * 2 + 1];
                tmp_imag[i + half_len] = imag_in[i * 2 + 1];
            }
            std::cout << " len " << len << std::endl;
            for (int i = 0; i < len; ++i)
            {
                std::cout << "  " << i << " " << real_in[i] << " , " << imag_in[i] << std::endl;
            }
            std::cout << " len " << len << std::endl;
            for (int i = 0; i < len; ++i)
            {
                std::cout << "  " << i << " " << tmp_real[i] << " , " << tmp_imag[i] << std::endl;
            }
            T* even_in_real = tmp_real.data();
            T* even_in_imag = tmp_imag.data();
            T* odd_in_real = tmp_real.data() + half_len;
            T* odd_in_imag = tmp_imag.data() + half_len;
            T* even_out_real = real_out;
            T* even_out_imag = imag_out;
            T* odd_out_real = real_out + half_len;
            T* odd_out_imag = imag_out + half_len;

            ForwardRecursive(half_len, even_in_real, even_in_imag, even_out_real, even_out_imag);
            ForwardRecursive(half_len, odd_in_real, odd_in_imag, odd_out_real, odd_out_imag);
            for (int k = 0; k < half_len; ++k)
            {

                T odd_twiddle_real =
                    T(odd_out_real[k] * m_cos[k][m] + odd_out_imag[k] * m_sin[k][m]);
                T odd_twiddle_imag =
                    T(-odd_out_real[k] * m_sin[k][m] + odd_out_imag[k] * m_cos[k][m]);

                real_out[k + half_len] = even_out_real[k] - odd_twiddle_real;
                imag_out[k + half_len] = even_out_imag[k] - odd_twiddle_imag;
                real_out[k] = even_out_real[k] + odd_twiddle_real;
                imag_out[k] = even_out_imag[k] + odd_twiddle_imag;
            }
        }
    }

    template <typename T>
    void FFTRecursive<T>::InverseRecursive(const int len, const T* real_in, const T* imag_in,
                                           T* real_out, T* imag_out)
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
            std::vector<T> tmp_real(len);
            std::vector<T> tmp_imag(len);
            for (int i = 0; i < half_len; i++)
            {
                tmp_real[i] = real_in[i * 2];
                tmp_imag[i] = imag_in[i * 2];
                tmp_real[i + half_len] = real_in[i * 2 + 1];
                tmp_imag[i + half_len] = imag_in[i * 2 + 1];
            }
            T* even_in_real = tmp_real.data();
            T* even_in_imag = tmp_imag.data();
            T* odd_in_real = tmp_real.data() + half_len;
            T* odd_in_imag = tmp_imag.data() + half_len;
            T* even_out_real = real_out;
            T* even_out_imag = imag_out;
            T* odd_out_real = real_out + half_len;
            T* odd_out_imag = imag_out + half_len;

            InverseRecursive(half_len, even_in_real, even_in_imag, even_out_real, even_out_imag);
            InverseRecursive(half_len, odd_in_real, odd_in_imag, odd_out_real, odd_out_imag);
            for (int k = 0; k < half_len; ++k)
            {

                T odd_twiddle_real =
                    T(odd_out_real[k] * m_cos[k][m] - odd_out_imag[k] * m_sin[k][m]);
                T odd_twiddle_imag =
                    T(odd_out_real[k] * m_sin[k][m] + odd_out_imag[k] * m_cos[k][m]);

                real_out[k + half_len] = even_out_real[k] - odd_twiddle_real;
                imag_out[k + half_len] = even_out_imag[k] - odd_twiddle_imag;
                real_out[k] = even_out_real[k] + odd_twiddle_real;
                imag_out[k] = even_out_imag[k] + odd_twiddle_imag;
            }
        }
    }
} // namespace learnfft