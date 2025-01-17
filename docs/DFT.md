 The Discrete Fourier Transform (DFT) is a form of Fourier Transform suitable for computers. The goal of this repository is to learn how to implement DFT, optimize it step by step. 
 
 Before we start coding, some understand the background knowledge of the Fourier Transform and the Discrete Fourier Transform should be introduced.

# What's the Fourier Transform?

The most direct way to observe and measure signals in the physical world is to record how the signals change over time. This provides us the time-domain signal x(t), which represents the signal‘s amplitude as it changes over time. There are various methods for analyzing signals in the time domain, but certain signal characteristics may not be readily apparent in the time domain. Mathematicians/physicists might convert signals to a specific transform domain for analysis.

For instance, when you document a live performance. Different vocal and instrumental elements, such as bass and soprano singing, along with a variety of musical instruments, can be heard at different pitches. In the time domain, an audio signal looks like a chaotically changing sequence, where you can only see that the volume is sometimes high and sometimes low. How can we distinguish between bass, soprano, and instrumental voices, and identify the songs they are performing?

![](music_time.png)

Sound waves are a type of vibration that is defined by two factors: frequency of vibration and amplitude of vibration. As the frequency of vibration increases, the human ear perceives sharper sounds, while lower frequencies are felt as deeper tones. 

Adult males typically have thicker and wider vocal cords compared to females, resulting in slower vibration and deeper voices for males, and higher voices for females. Various musical instruments resonate at unique frequencies and harmonics. By determining the vibration frequencies present in the music's sound wave signal, we can identify the instruments, singers, and song playing. As show in he image blow, the peaks on the spectrum highlight the specific frequencies of the music being played at that moment.

![](music_freq.png)

The Fourier transform is a mathematical tool that converts signals from the time domain (amplitude, time) to the frequency domain (amplitude, frequency) for analysis and processing.

How can this transformation be accomplished? Let's revisit the music signal above and zoom in a certain section. We notice that certain waveforms are repeated, with smaller waveforms overlaid on the larger ones. The time domain sequence that appears to vary randomly is actually not random. It can be broken down into a combination of signals with varying period of time. Frequency decreases as the period lengthens and increases as the period shortens.

![](music_time_zoom.png)

Decomposing a time-domain signal into a linear superposition of periodic signals of different frequencies is the fundamental concept behind the Fourier transform.

## Fourier Series (FS)

The Fourier series offers a way to break down a periodic function into a combination of sine and cosine functions. Using the figure presented as a reference, a rectangular wave can be decomposed into a major sine wave which is then combined with multiple smaller sine waves.

![](Fourier_series_and_transform.gif)

*[Fourier transform time and frequency domains (small).gif, CC0, By Lucas Vieira](https://commons.wikimedia.org/w/index.php?curid=28399050)*

傅里叶级数利用了三角函数的周期性和正交性，将周期信号变换为三角函数正交基的权重。实数形式如下：

$$\begin{aligned}
f(t) & = \frac{a_0}{2}+\sum_{n=0}^{\infty} \left[a_n\cos(nt)+b_n\sin(nt)\right] \\
其中 \\
a_0 & = \frac{1}{\pi}\int_{-\pi}^{\pi} f(t)dt \\
a_n & = \frac{1}{\pi}\int_{-\pi}^{\pi} f(t)\cos(nt)dt \\
b_n & = \frac{1}{\pi}\int_{-\pi}^{\pi} f(t)\sin(nt)dt \\
\end{aligned}$$

复数形式的傅里叶级数形式如下，其中 $T_0$ 是时域周期，$f_0 = 1/T_0$。

$$\begin{aligned}
X(kf_0) & = \frac{1}{T_0}\int_{T0} x(t) e^{-j2\pi kf_0t}dt \\
x(t) & = \sum_{k=-\infty}^{\infty} X(kf_0)e^{j2\pi kf_0t} 
\end{aligned}$$

## 连续时间傅里叶变换CTFT

对于非周期信号，不能直接用傅里叶级数，但可认为这是一种特殊的周期函数，其周期趋近于无穷。这个条件下，傅里叶级数可以推导为傅里叶积分，得到的是一个频谱密度函数。

$$\begin{aligned}
X(f) & = \lim_{T_0->\infty} \frac{1}{T_0}\int_{T0} x(t) e^{-j2\pi kf_0t}dt 
\end{aligned}$$

傅里叶积分可用来分析非周期连续信号的频谱密度，也可以用频谱密度恢复时域信号。这就是连续时间傅里叶变换CTFT。

$$\begin{aligned}
X(f) & = \int_{t=-\infty}^{\infty} x(t) e^{-j2\pi ft}dt \\
x(t) & = \int_{k=-\infty}^{\infty} X(f)e^{j2\pi ft}df 
\end{aligned}$$

# 什么是DFT?
针对物理世界的连续信号，我们有连续时间傅里叶变换作为数学工具。但是电子计算机使用二进制器件进行计算、且只有固定的内存空间，故只能处理数字化的信号，也就是有限长的、离散化的、量化的信号。

而离散傅里叶变换（DFT）就是为数字世界设计的，一种时域信号离散且有限长、频域信号离散且有限长的变换。

## 时域离散化：离散时间傅里叶变换DTFT

首先，我们对连续信号时域上作离散化，按 $T_s$ 周期采样, 也就是将原信号乘以一个冲击串函数: $x(t)\delta(t-nT_s)$。

![](impulse_train_sampling.png)

将 $x(t)\delta(t-nT_s)$ 带入傅里叶变换公式，我们就得到一个周期采样的傅里叶变换。

$$\begin{aligned}
X(f) & = \int_{t=-\infty}^{\infty} x(t)\delta(t-nT_s) e^{-j2\pi ft}dt \\
      & = \sum_{k=-\infty}^{\infty} x(nTs) e^{-j2\pi f(nTs)}
\end{aligned}$$

根据傅里叶变换的性质，时域上乘以一个冲击串，相当于频域卷积一个冲击串的傅里叶变换，冲击串的傅里叶变换也是一个冲击串。《信号与系统》教材里说明了，这个结果，就是频域上以 $f_s = T_s$ 为周期重复。

![](impulse_train_convolution.png)

所以时域周期 $T_s$ 采样后的频谱，周期为 $f_s$ 的周期函数，以 $\omega = 2\pi f / f_s$ 归一化频率，我们可以得到归一化频率的离散时间傅里叶变换DTFT公式。

$$\begin{aligned}
X(\omega) & = \sum_{k=-\infty}^{\infty} x[n] e^{-j\omega n} \\
x[n] & = \frac{1}{2\pi}\int_{-\pi}^{\pi} X(\omega)e^{j\omega n} d\omega 
\end{aligned}$$

因为频谱被周期延拓了，实际的有效频谱只在一个周期内（通常取 $-\pi$ 到 $\pi$ ），也就是使用 $-\pi$ 到 $\pi$ 内的频谱作逆变换，即可恢复时域信号。所以DTFT逆变换的积分上下界取的是 $-\pi$ 到 $\pi$。

## 频域离散化：离散傅里叶变换DFT

DTFT实现了时域的离散化。类似地，我们可以在此基础上实现频域的离散化，也就是对频域进行采样。假设在DTFT的频域上按照 $\omega_k$ 采样, 则时域信号按照 $2\pi/\omega_k$ 周期延拓，于是我们就得到了一对离散信号和他们之间的变换与逆变换，这就是离散傅里叶变换DFT。

$$\begin{aligned}
DFT: X[k] & = \sum_{n=0}^{N-1} x[n] e^{-2\pi \frac{k}{N}n} \\
IDFT: x[n] & = \frac{1}{N}\sum_{n=0}^{N-1} X[k] e^{2\pi \frac{k}{N}n} 
\end{aligned}$$

所以从CTFT到DFT，我们经过几个步骤
- 时域上按 $T_s$ 周期采样（频域上成为周期为 $f_s = 1/T_s$ 的周期频谱）
- 时域信号截短成为有限长信号，并作周期 $T_k$ 延拓（也就是频域离散化，采样间隔 $\omega_k = 2 \pi/T_k$ ）
- 取时域上一个周期、频域上一个周期的序列作为离散傅里叶变化的输入和输出。

可以对比一下以上几种变换的性质：

|变换 |	时间	|频率|
|:-----:|  :----: | :----:| 
|傅里叶级数|	连续，周期性	|离散，非周期性|
|连续时间傅里叶变换|	连续，非周期性	|连续，非周期性|
|离散时间傅里叶变换	|离散，非周期性	|连续，周期性|
|离散傅里叶变换	|离散，周期性|	离散，周期性|

有了DFT，我们就可以对现实世界的信号进行频率分析啦，而且可以使用数学或软件的手段进行计算加速，成为一种高效的数字信号处理工具。从而衍生出各式各样的数字信号处理的应用，DSP应用主要是围绕这三个阶段：
- Analysis 分析：从时域到频域的信号变换，如语音到语谱。
- Filtering 滤波：频域上的操作，如高通、低通、均衡。
- Synthesis 合成：从频域到时域的逆变换，如从语谱合成语音。

要注意DFT是CTFT的离散化，是对真实连续信号的一种数学近似，从而可以在数字系统中落地。数字系统的离散化、量化是有代价的。时域和频域的采样、截短会带来频谱混叠和频谱泄漏，使用不同精度的数值类型进行运算也会带来不同的量化误差。这些误差的大小就是衡量数字系统的精密程度的指标，是我们工程实现中要时刻考虑的。

# DFT的代码实现
从前面的背景知识，我们已经得到N点的DFT变换和逆变换的公式如下，后面就是怎么用代码实现DFT了。

$$\begin{aligned}
DFT: X[k] & = \sum_{n=0}^{N-1} x[n] e^{-2\pi \frac{k}{N}n} \\
IDFT: x[n] & = \frac{1}{N}\sum_{n=0}^{N-1} X[k] e^{2\pi \frac{k}{N}n} 
\end{aligned}$$

上式中时域信号$x[n]$和频域信号$X[k]$都是复数信号。因为计算机的算数运算都是实数运算，复数运算需要额外的库支持，我们可以将DFT公式展开成实数运算。首先使用欧拉公式展开复指数：

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

为了减少每次进行DF计算的时间，我们可以使用空间换时间。在DFT类初始化时，给定DFT的size，并且预创建三角函数表 $t\_cos[k][n] = \cos(2\pi \frac{k}{N}n)$ 和 $t\_sin[k][n] = \sin(2\pi \frac{k}{N}n)$, 后续每次计算DFT时查表即可。

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

以下就是按照DFT和IDFT公式实现的C++代码。

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
现实世界采集的信号通常是实数序列，其虚部为零，所以虚部乘法全部可以去掉。于是DFT的公式可以简化如下：

$$\begin{aligned}
X_r[k] & = \sum_{n=0}^{N-1} \left[ x_r[n]\cos(2\pi \frac{k}{N}n)\right] \\
X_i[k] & = \sum_{n=0}^{N-1} \left[-x_r[n]\sin(2\pi \frac{k}{N}n)\right] \\
\end{aligned}$$

根据cosine和sine函数的对称性，我们可以得到实数序列DFT的共轭对称性：
$$\begin{aligned}
X_r[k] & = X_r[N-k] \\
X_i[k] & = -X_i[N-k] \\
X[k] & = X^*[N-k]
\end{aligned}$$

比如对于8点的DFT，$X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7$,  有

$$\begin{aligned}
X_1 = X_7^* \\
X_2 = X_6^* \\
X_3 = X_5^* \\
\end{aligned}$$

因此我们只要计算 $X_0, X_1, X_2, X_3, X_4$ 既可以，$X_5, X_6, X_7$ 可以从共轭推导出来。对于N点DFT，假设只考虑偶数N， 那我们只要计算前 $N/2 + 1$ 个点即可。

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

In the inverse transform, we need to use the real and imaginary parts in the frequency domain to calculate the real part in the time domain.

```cpp
    void DFT::InverseReal(const double* real_in, const double* imag_in, double* real_out)
    {
        for (int i = 0; i < m_size; ++i)
        {
            double re = 0.0;
            for (int j = 0; j < m_bins; ++j)
                re += real_in[j] * m_cos[i][j];
            for (int j = m_bins; j < m_size; ++j)
                re += real_in[m_size - j] * m_cos[i][j];

            for (int j = 0; j < m_bins; ++j)
                re -= imag_in[j] * m_sin[i][j];
            for (int j = m_bins; j < m_size; ++j)
                re -= -imag_in[m_size - j] * m_sin[i][j];

            real_out[i] = re;
        }
    }
```

# DFT Performance Test

Now let's verify the correctness and computational performance of the DFT implementation. First, we choose a commonly used open source FFT software[KISSFFT](https://github.com/mborgerding/kissfft) as a baseline to compare。These values would be recorded:

- Runtime per pass of my DFT and KISS.
- The error between the output of my_dft and KISS for the same input.。
- Run my_dft with DFT forward, and then IDFT to recover the signal., then performs the IDFT on the result. Recored the error between the recovered result and the original signal.

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

The same comparison is made for real sequence DFT.

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

Using 1024-point random double as input, the DFT performance test results are as follows:

|        | KissFFT |  my DFT  | KissFFTR| my DFT Real |
| :-----:|  :----: | :----:| :----: | :----: |
| Time Per Pass   | 0.01055ms | 2.39988ms | 0.00657ms |  1.0395ms |
| Forward-Inverse Error |  1.208e-16  | 5.267e-14 | 1.213e-16 | 3.368e-14 |
| DFT-KissFFT Differene |    | 1.196e-12 |  | 2.122e-13 |

So far, we have implemented a basic DFT using C++, which can actually run in a program. The correctness of the calculation was also vierified. However, its computational complexity is $O(log N^2)$, and the computational performance is still far from commercial use.

Therefore, we need to introduce DFT computational optimization, which is the fast Fourier transform (FFT) to be discussed next.

# 参考资料
- Oppenheim, Willsky, Nawab - Signals & Systems [2nd Edition]
- Proakis, John G. Digital signal processing: principles algorithms and applications. Pearson Education India, 2001.
- [Mathematics of the Discrete Fourier Transform (DFT), with Audio Applications --- Second Edition, by Julius O. Smith III, W3K Publishing, 2007](https://ccrma.stanford.edu/~jos/mdft/mdft.html)
- [從傅立葉轉換到數位訊號處理](https://alan23273850.gitbook.io/signals-and-systems)
- [KISSFFT, by Mark Borgerding](https://github.com/mborgerding/kissfft)