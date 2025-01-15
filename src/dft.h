/*
 * Copyright (c) 2025, wjchen, BSD 3-Clause License
 */
#pragma once

#include <iostream>
#include <math.h>
#include <vector>

namespace learnfft
{
    template <typename T> class DFT
    {
    public:
        DFT(size_t size);
        ~DFT();

        void Forward(const T* real_in, const T* imag_in, T* real_out, T* imag_out);
        void Inverse(const T* real_in, const T* imag_in, T* real_out, T* imag_out);

        void ForwardReal(const T* real_in, T* real_out, T* imag_out);
        void InverseReal(const T* real_in, const T* imag_in, T* real_out);

    private:
        const size_t m_size;
        const size_t m_bins;
        std::vector<std::vector<T>> m_sin;
        std::vector<std::vector<T>> m_cos;
        std::vector<std::vector<T>> m_tmp;
    };

    template <typename T>
    DFT<T>::DFT(size_t size)
        : m_size(size), m_bins(size / 2 + 1), m_sin(size, std::vector<T>(size)),
          m_cos(size, std::vector<T>(size)), m_tmp(size, std::vector<T>(size))
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
    template <typename T> DFT<T>::~DFT() {}

    template <typename T>
    void DFT<T>::Forward(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            T re = 0.0, im = 0.0;
            for (int j = 0; j < m_size; ++j)
                re += real_in[j] * m_cos[i][j] + imag_in[j] * m_sin[i][j];
            for (int j = 0; j < m_size; ++j)
                im -= real_in[j] * m_sin[i][j] - imag_in[j] * m_cos[i][j];
            real_out[i] = re;
            imag_out[i] = im;
        }
    }

    template <typename T>
    void DFT<T>::Inverse(const T* real_in, const T* imag_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            T re = 0.0, im = 0.0;
            for (int j = 0; j < m_size; ++j)
                re += real_in[j] * m_cos[i][j] - imag_in[j] * m_sin[i][j];
            for (int j = 0; j < m_size; ++j)
                im += real_in[j] * m_sin[i][j] + imag_in[j] * m_cos[i][j];
            real_out[i] = re;
            imag_out[i] = im;
        }
    }

    template <typename T> void DFT<T>::ForwardReal(const T* real_in, T* real_out, T* imag_out)
    {
        for (int i = 0; i < m_bins; ++i)
        {
            T re = 0.0, im = 0.0;
            for (int j = 0; j < m_size; ++j)
                re += real_in[j] * m_cos[i][j];
            for (int j = 0; j < m_size; ++j)
                im -= real_in[j] * m_sin[i][j];
            real_out[i] = re;
            imag_out[i] = im;
        }
    }

    template <typename T> void DFT<T>::InverseReal(const T* real_in, const T* imag_in, T* real_out)
    {
        for (int i = 0; i < m_bins; ++i)
        {
            m_tmp[0][i] = real_in[i];
            m_tmp[1][i] = imag_in[i];
        }
        for (int i = m_bins; i < m_size; ++i)
        {
            m_tmp[0][i] = real_in[m_size - i];
            m_tmp[1][i] = -imag_in[m_size - i];
        }
        for (int i = 0; i < m_size; ++i)
        {
            T re = 0.0;
            for (int j = 0; j < m_size; ++j)
                re += m_tmp[0][j] * m_cos[i][j];
            for (int j = 0; j < m_size; ++j)
                re -= m_tmp[1][j] * m_sin[i][j];
            real_out[i] = re;
        }
    }
} // namespace learnfft
