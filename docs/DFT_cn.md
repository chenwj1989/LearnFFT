
# 什么是傅里叶变换?

# 为什么需要DFT?
针对物理世界的连续信号，我们有连续时间傅立叶变换作为数学工具。但是电子计算机使用二进制器件进行计算、且只有固定的内存空间，故只能处理数字化的信号，也就是有限长的、离散化的、量化的信号。

而离散傅立叶变换（DFT）就是为数字世界设计的，一种时域信号离散且有限长、频域信号离散且有限长的变换。

有了DFT，我们就可以对现实世界的信号进行频率分析啦，而且可以使用数学或软件的手段进行计算加速，成为一种高效的数字信号处理工具。从而衍生出各式各样的数字信号处理的应用，DSP应用主要是围绕这三个阶段：
- Analysis 分析：从时域到频域的信号变换，如语音到语谱。
- Filtering 滤波：频域上的操作，如高通、低通、均衡。
- Synthesis 合成：从频域到时域的逆变换，如从语谱合成语音。

# DFT的代码实现
DFT变换和逆变换的公式如下
$$
\begin{aligned}
DFT: X[k] & = \sum_{n=0}^{N-1} x[n] e^{-2\pi \frac{k}{N}n} \\
IDFT: x[n] & = \frac{1}{N}\sum_{n=0}^{N-1} X[k] e^{2\pi \frac{k}{N}n} 
\end{aligned}
$$

其中时域信号$x[n]$和频域信号$X[k]$都是复数信号。因为计算机的算数运算都是实数运算，复数运算需要额外的库支持，所以我们可以将DFT公式展开成实数运算。首先使用欧拉公式展开复指数：
$$\begin{aligned}
X[k] & = \sum_{n=0}^{N-1} x[n] e^{-2\pi \frac{k}{N}n} \\
     & = \sum_{n=0}^{N-1} x[n] \left[ \cos(2\pi \frac{k}{N}n) - j\sin(2\pi \frac{k}{N}n)\right] \\
     & = \sum_{n=0}^{N-1} \left( x_r[n] + jx_i[n]\right) \left[ \cos(2\pi \frac{k}{N}n) - j\sin(2\pi \frac{k}{N}n)\right] 
\end{aligned}$$

然后分解实部和虚部，我们就得到了DFT和IDFT实部和虚部计算的四条公式：
$$\begin{aligned}
X_r[k] & = \sum_{n=0}^{N-1} \left[ x_r[n]\cos(2\pi \frac{k}{N}n) + x_i[n]\sin(2\pi \frac{k}{N}n)\right] \\
X_i[k] & = \sum_{n=0}^{N-1} \left[-x_r[n]\sin(2\pi \frac{k}{N}n) + x_i[n]\cos(2\pi \frac{k}{N}n)\right] \\
\end{aligned}$$

$$\begin{aligned}
x_r[n] & = \frac{1}{N}\sum_{n=0}^{N-1} \left[ X_r[k]\cos(2\pi \frac{k}{N}n) - X_r[k]\sin(2\pi \frac{k}{N}n)\right] \\
x_i[n] & = \frac{1}{N}\sum_{n=0}^{N-1} \left[X_r[k]\sin(2\pi \frac{k}{N}n) + X_r[k]\cos(2\pi \frac{k}{N}n)\right] \\
\end{aligned}$$

为了减少每次进行DF计算的时间，我们可以使用空间换时间。在DFT类初始化时，给定DFT的size，并且预创建三角函数表$t\_cos[k][n] = \cos(2\pi \frac{k}{N}n)$ 和$t\_sin[k][n] = \sin(2\pi \frac{k}{N}n)$, 后续每次计算DFT时查表即可。

```cpp
    for (int i = 0; i < m_size; ++i)
    {
        for (int j = 0; j < m_size; ++j)
        {
            double arg = (double(i) * double(j) * M_PI * 2.0) / m_size;
            m_sin[i][j] = sin(arg);
            m_cos[i][j] = cos(arg);
        }
    }
```
最后，我们通过下面四条公式即可实现基础的DFT变化和逆变换。
$$\begin{aligned}
X_r[k] & = \sum_{n=0}^{N-1} \left( x_r[n]*t\_cos[k][n] + x_i[n]*t\_sin[k][n]\right) \\
X_i[k] & = \sum_{n=0}^{N-1} \left(-x_r[n]*t\_sin[k][n] + x_i[n]*t\_cos[k][n]\right) \\
x_r[n] & = \frac{1}{N}\sum_{n=0}^{N-1} \left( X_r[k]*t\_cos[k][n] - X_i[k]*t\_sin[k][n]\right) \\
x_i[n] & = \frac{1}{N}\sum_{n=0}^{N-1} \left(X_r[k]*t\_sin[k][n] + X_i[k]*t\_cos[k][n]\right) \\
\end{aligned}$$

```cpp
void DFT::Forward(const double* real_in, const double* imag_in, double* real_out, double* imag_out) {
    for (int i = 0; i < m_size; ++i) {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < m_size; ++j) re += real_in[j] * m_cos[i][j] + imag_in[j] * m_sin[i][j];
        for (int j = 0; j < m_size; ++j) im -= real_in[j] * m_sin[i][j] - imag_in[j] * m_cos[i][j];
    }
}

void DFT::Inverse(const double* real_in, const double* imag_in, double* real_out, double* imag_out) {
    for (int i = 0; i < m_size; ++i) {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < m_size; ++j) re += real_in[j] * m_cos[i][j] - imag_in[j] * m_sin[i][j];
        for (int j = 0; j < m_size; ++j) im += real_in[j] * m_sin[i][j] + imag_in[j] * m_cos[i][j];
    }
}
```
# 实数序列DFT
现实世界采集的信号通常是实数序列，其虚部为零，所以虚部乘法全部可以去掉。

这就是实数序列DFT的共轭对称性

有了这个性质，对于实数序列DFT, 我们不仅节约了大半计算耗时，还节省接近一半存储空间。

```cpp
    void DFT::ForwardReal(const double* real_in, double* real_out, double* imag_out)
    {
        for (int i = 0; i < m_size / 2 + 1; ++i)
        {
            double re = 0.0, im = 0.0;
            for (int j = 0; j < m_size; ++j)
                re += real_in[j] * m_cos[i][j];
            for (int j = 0; j < m_size; ++j)
                im -= real_in[j] * m_sin[i][j];
            real_out[i] = re;
            imag_out[i] = im;
        }
    }
```

# DFT性能测试


```cpp
    start_time = clock();
    my_dft.Forward(in_real, in_imag, out_real, out_imag);
    end_time = clock();

```
```cpp
    start_time = clock();
    kiss_fft(forward_fft, in_cpx, out_cpx);
    end_time = clock();
```

```cpp
    start_time = clock();
    my_dft.ForwardReal(in_real, out_real, out_imag);
    end_time = clock();
 ```

```cpp
    start_time = clock();
    kiss_fftr(forward_fft, in_real, out_cpx);
    end_time = clock();
```

1024点double类型DFT性能测试
|        | KissFFT |  DFT  | KissFFTR| DFT Real |
| :-----:|  :----: | :----:| :----: | :----: |
| Time Per Pass   | 0.01055ms | 2.39988ms | 0.00657ms |  1.0395ms |
| Forward-Inverse Error |  1.208e-16  | 5.267e-14 | 1.213e-16 | 3.368e-14 |
| DFT-KissFFT Differene |    | 1.196e-12 |  | 2.122e-13 |

到这里，我们就使用C++实现了一个基础的DFT，可以在程序中实际运行且验证了计算的正确性。但是其计算复杂度是$O(log N^2)$,计算性能离可商用距离还比较遥远。

因此，我们需要引入DFT的计算优化，也就是下一步要讨论的快速傅里叶变换（FFT）。

# 参考资料
- Proakis, John G. Digital signal processing: principles algorithms and applications. Pearson Education India, 2001.
- [Mathematics of the Discrete Fourier Transform (DFT), with Audio Applications --- Second Edition, by Julius O. Smith III, W3K Publishing, 2007](https://ccrma.stanford.edu/~jos/mdft/mdft.html)
- [從傅立葉轉換到數位訊號處理](https://alan23273850.gitbook.io/signals-and-systems)
- [KISSFFT, by Mark Borgerding](https://github.com/mborgerding/kissfft)