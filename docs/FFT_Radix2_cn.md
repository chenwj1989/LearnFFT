Fast Fourier transform (FFT): Efficient algorithm to compute the discrete Fourier transform (DFT) and its inverse

# DFT性质

$$\begin{aligned}
DFT: X[k] & = \sum_{n=0}^{N-1} x[n] e^{-j2\pi \frac{k}{N}n}, for\ k = 0,1,..., N-1 \\
IDFT: x[n] & = \frac{1}{N}\sum_{n=0}^{N-1} X[k] e^{j2\pi \frac{k}{N}n} ,for\ k = 0,1,..., N-1 
\end{aligned}$$

# 用分治策略计算DFT

$$\begin{aligned}
X[k] & = \sum_{n=0}^{N-1} x[n] e^{-j2\pi \frac{k}{N}n} \\
      & = \sum_{m=0}^{N/2-1} x[2m] e^{-j2\pi \frac{k}{N}2m}  + \sum_{m=0}^{N/2-1} x[2m+1] e^{-j2\pi \frac{k}{N}2m+1} \\
      & = \sum_{m=0}^{N/2-1} x[2m] e^{-j2\pi \frac{k}{N/2}m}  + e^{-j2\pi \frac{k}{N}}\sum_{m=0}^{N/2-1} x[2m+1] e^{-j2\pi \frac{k}{N/2}m} \\
\end{aligned}$$


$$\begin{aligned}
X[k] & = \sum_{m=0}^{M-1} x_e[m] e^{-j2\pi \frac{k}{M}m}  + e^{-j2\pi \frac{k}{N}}\sum_{m=0}^{M-1} x_o[m] e^{-j2\pi \frac{k}{M}m} \\
& = X_e[k] + e^{-j2\pi \frac{k}{N}}X_o[k] \\
\end{aligned}$$

$$\begin{aligned}
X[k] & = X[j] = X_e[j] + e^{-j2\pi \frac{j}{N}}X_o[j], for\ k =0,...N/2-1\\
X[k] & = X[j+N/2] = X_e[j] - e^{-j2\pi \frac{j}{N}}X_o[j], for\ k =N/2,...N-1\\
\end{aligned}$$


# FFT的递归实现

```cpp
    template <typename T>
    void DeInterleave(const int len, T* data)
    {
        int half_len = len / 2;
        vector<T> tmp(half_len);
        for (int i = 0; i < half_len; i++)
        {
            data[i] = data[i * 2];
            tmp[i] = data[i * 2 + 1];
        }
        for (int i = 0; i < half_len; i++)
        {
            data[i + half_len] = tmp[i];
        }
    }
```

```cpp
    void FFTCoreRecursive(const int len, double* real_in, double* imag_in, 
                          double* real_out,double* imag_out, bool forward)
    {
        if (len == 1)
        {
            real_out[0] = real_in[0];
            imag_out[0] = imag_in[0];
        }
        else
        {
            int half_len = len / 2;
            int m = m_size / len;
            DeInterleave(len, real_in);
            DeInterleave(len, imag_in);
            double* even_in_real = real_in;
            double* even_in_imag = imag_in;
            double* odd_in_real = real_in + half_len;
            double* odd_in_imag = imag_in + half_len;
            double* even_out_real = real_out;
            double* even_out_imag = imag_out;
            double* odd_out_real = real_out + half_len;
            double* odd_out_imag = imag_out + half_len;

            FFTCoreRecursive(half_len, even_in_real, even_in_imag, even_out_real, even_out_imag,
                             forward);
            FFTCoreRecursive(half_len, odd_in_real, odd_in_imag, odd_out_real, odd_out_imag,
                             forward);
            for (int k = 0; k < half_len; ++k)
            {
                double odd_twiddle_real;
                double odd_twiddle_imag;
                if (forward)
                {
                    odd_twiddle_real =
                        odd_out_real[k] * m_cos[k][m] + odd_out_imag[k] * m_sin[k][m];
                    odd_twiddle_imag =
                        -odd_out_real[k] * m_sin[k][m] + odd_out_imag[k] * m_cos[k][m];
                }
                else
                {
                    odd_twiddle_real =
                        odd_out_real[k] * m_cos[k][m] - odd_out_imag[k] * m_sin[k][m];
                    odd_twiddle_imag =
                        odd_out_real[k] * m_sin[k][m] + odd_out_imag[k] * m_cos[k][m];
                }

                real_out[k + half_len] = even_out_real[k] - odd_twiddle_real;
                imag_out[k + half_len] = even_out_imag[k] - odd_twiddle_imag;
                real_out[k] = even_out_real[k] + odd_twiddle_real;
                imag_out[k] = even_out_imag[k] + odd_twiddle_imag;
            }
        }
    }
```
# FFT的非递归实现

# 实数序列FFT的实现

