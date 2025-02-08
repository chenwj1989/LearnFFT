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
    template <typename T> class FFTReal
    {
    public:
        FFTReal(size_t size);
        ~FFTReal();

        void Forward(const T* real_in, T* real_out, T* imag_out);
        void Inverse(const T* real_in, const T* imag_in, T* real_out);

    private:
        void FFTRealCore(T* real_out, T* imag_out, bool forward);

        const size_t m_real_size;
        const size_t m_size;
        const size_t m_bins;
        std::vector<size_t> m_bit_reverse_idx;
        std::vector<std::vector<T>> m_sin;
        std::vector<std::vector<T>> m_cos;
        std::vector<T> m_sin2n;
        std::vector<T> m_cos2n;
        std::vector<T> m_tmp_real;
        std::vector<T> m_tmp_imag;
    };

    template <typename T>
    FFTReal<T>::FFTReal(size_t size)
        : m_real_size(size), m_size(size / 2), m_bins(size / 2 + 1), m_bit_reverse_idx(size / 2),
          m_sin(size / 2, std::vector<T>(size / 2)), m_cos(size / 2, std::vector<T>(size / 2)),
          m_sin2n(size), m_cos2n(size), m_tmp_real(size), m_tmp_imag(size)
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
        for (int i = 0; i < m_real_size; ++i)
        {
            double arg = (double(i) * M_PI * 2.0) / m_real_size;
            m_sin2n[i] = sin(arg);
            m_cos2n[i] = cos(arg);
        }
        GenBitReverseOrder(m_size, m_bit_reverse_idx);
    }
    template <typename T> FFTReal<T>::~FFTReal() {}

    template <typename T> void FFTReal<T>::Forward(const T* real_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            real_out[i] = real_in[2 * m_bit_reverse_idx[i]];
            imag_out[i] = real_in[2 * m_bit_reverse_idx[i] + 1];
        }
        FFTRealCore(real_out, imag_out, true);
        for (int k = 1; k < m_size / 2; ++k)
        {
            int n_k = m_size - k;
            T r_1 = real_out[k] + real_out[n_k];
            T i_1 = imag_out[k] - imag_out[n_k];
            T r_2 = real_out[k] - real_out[n_k];
            T i_2 = imag_out[k] + imag_out[n_k];
            T r_2_tw = T(-r_2 * m_sin2n[k] + i_2 * m_cos2n[k]);
            T i_2_tw = -T(r_2 * m_cos2n[k] + i_2 * m_sin2n[k]);
            real_out[k] = (r_1 + r_2_tw) / 2.;
            imag_out[k] = (i_1 + i_2_tw) / 2.;

            T r_2_nk_tw = T(r_2 * m_sin2n[n_k] + i_2 * m_cos2n[n_k]);
            T i_2_nk_tw = -T(-r_2 * m_cos2n[n_k] + i_2 * m_sin2n[n_k]);

            real_out[n_k] = (r_1 + r_2_nk_tw) / 2.;
            imag_out[n_k] = (-i_1 + i_2_nk_tw) / 2.;
        }
        int k = m_size / 2;
        real_out[k] = T(real_out[k] + imag_out[k] * m_cos2n[k]);
        imag_out[k] = -T(imag_out[k] * m_sin2n[k]);

        real_out[m_size] = (real_out[0] - imag_out[0]);
        real_out[0] = (real_out[0] + imag_out[0]);
        imag_out[m_size] = 0.;
        imag_out[0] = 0.;
    }

    template <typename T> void FFTReal<T>::Inverse(const T* real_in, const T* imag_in, T* real_out)
    {
        for (int k = 1; k <= m_size / 2; ++k)
        {
            // X1[k] = G[k] + G[k+N];
            // X2[k] = W-k/2N(G[k] - G[k+N]);
            // X[k] = X1[k] + jX2[k];
            // x1 = G[k] + G[k+N];
            // x2 = W-k/2N(G[k] - G[k+N]);
            int n_k = m_size - k;
            T x1_real = real_in[k] + real_in[n_k];
            T x1_imag = imag_in[k] - imag_in[n_k];
            T x2_real_tw = real_in[k] - real_in[n_k];
            T x2_imag_tw = imag_in[k] + imag_in[n_k];
            T x2_real = -x2_real_tw * m_sin2n[k] - x2_imag_tw * m_cos2n[k];
            T x2_imag = x2_real_tw * m_cos2n[k] - x2_imag_tw * m_sin2n[k];

            m_tmp_real[k] = (x1_real + x2_real) / 2.;
            m_tmp_imag[k] = (x1_imag + x2_imag) / 2.;

            if (n_k != k)
            {
                T x2_nk_real = x2_real_tw * m_sin2n[n_k] - x2_imag_tw * m_cos2n[n_k];
                T x2_nk_imag = -x2_real_tw * m_cos2n[n_k] - x2_imag_tw * m_sin2n[n_k];

                m_tmp_real[n_k] = (x1_real + x2_nk_real) / 2.;
                m_tmp_imag[n_k] = (-x1_imag + x2_nk_imag) / 2.;
            }
        }

        m_tmp_real[0] = (real_in[0] + real_in[m_size]) / 2.;
        m_tmp_imag[0] = (real_in[0] - real_in[m_size]) / 2.;

        for (int i = 0; i < m_size; i++)
        {
            if (i < m_bit_reverse_idx[i])
            {
                std::swap(m_tmp_real[i], m_tmp_real[m_bit_reverse_idx[i]]);
                std::swap(m_tmp_imag[i], m_tmp_imag[m_bit_reverse_idx[i]]);
            }
        }

        FFTRealCore(m_tmp_real.data(), m_tmp_imag.data(), false);
        for (int i = 0; i < m_size; i++)
        {
            real_out[2 * i] = m_tmp_real[i] / m_size;
            real_out[2 * i + 1] = m_tmp_imag[i] / m_size;
        }
    }

    template <typename T> void FFTReal<T>::FFTRealCore(T* real_out, T* imag_out, bool forward)
    {
        int f_sign = forward ? 1 : -1;
        for (int btfly = 2, step = 1; btfly <= m_size; btfly *= 2, step *= 2)
        {
            int m = m_size / btfly;
            for (int i = 0; i < m_size; i += btfly)
            {
                for (int k = 0; k < step; ++k)
                {
                    int even = i + k;
                    int odd = even + step;

                    T odd_twiddle_real =
                        T(real_out[odd] * m_cos[k][m] + f_sign * imag_out[odd] * m_sin[k][m]);
                    T odd_twiddle_imag =
                        T(imag_out[odd] * m_cos[k][m] - f_sign * real_out[odd] * m_sin[k][m]);

                    real_out[odd] = real_out[even] - odd_twiddle_real;
                    imag_out[odd] = imag_out[even] - odd_twiddle_imag;
                    real_out[even] = real_out[even] + odd_twiddle_real;
                    imag_out[even] = imag_out[even] + odd_twiddle_imag;
                }
            }
        }
    }
} // namespace learnfft
