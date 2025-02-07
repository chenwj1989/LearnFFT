#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

static double random_value(double min, double max)
{
    return ((double)rand() / RAND_MAX) * (max - min) + min;
}

int testNaiveDFT(size_t len)
{
    std::vector<double> real_in(len);
    std::vector<double> imag_in(len);
    std::vector<double> real_out(len);
    std::vector<double> imag_out(len);
    std::vector<std::vector<double>> _sin(len, std::vector<double>(len));
    std::vector<std::vector<double>> _cos(len, std::vector<double>(len));

    for (size_t i = 0; i < len; i++)
    {
        real_in[i] = random_value(0.0, 1.0);
        imag_in[i] = random_value(0.0, 1.0);
    }

    for (size_t i = 0; i < len; ++i)
    {
        for (size_t j = 0; j < len; ++j)
        {
            double arg = (double(i) * double(j) * M_PI * 2.0) / len;
            _sin[i][j] = sin(arg);
            _cos[i][j] = cos(arg);
        }
    }
    printf("Init DFT \n");

    double start_time, end_time, total_time;
    start_time = clock();
    for (size_t i = 0; i < len; ++i)
    {
        double re = 0.0, im = 0.0;
        for (size_t j = 0; j < len; j++)
        {
            re += real_in[j] * _cos[i][j] + imag_in[j] * _sin[i][j];
            im -= real_in[j] * _sin[i][j] - imag_in[j] * _cos[i][j];
        }
        real_out[i] = re;
        imag_out[i] = im;
    }
    end_time = clock();
    total_time = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    printf("Serial calculation time of %zu-point DFT: %f ms.\n", len, total_time);

    start_time = clock();
    __m256d real_256d, imag_256d;
    __m256d cos_256d, sin_256d;
    __m256d a, b, c, d;
    for (size_t i = 0; i < len; ++i)
    {
        __m256d re_256d = _mm256_setzero_pd();
        __m256d im_256d = _mm256_setzero_pd();
        for (size_t j = 0; j < len; j += 4)
        {
            real_256d = _mm256_load_pd(real_in.data() + j);
            imag_256d = _mm256_load_pd(imag_in.data() + j);
            cos_256d = _mm256_load_pd(_cos[i].data() + j);
            sin_256d = _mm256_load_pd(_sin[i].data() + j);
            a = _mm256_mul_pd(real_256d, cos_256d);
            b = _mm256_add_pd(imag_256d, sin_256d);
            c = _mm256_mul_pd(real_256d, sin_256d);
            d = _mm256_add_pd(imag_256d, cos_256d);
            real_256d = _mm256_add_pd(a, b);
            imag_256d = _mm256_sub_pd(d, c);
            re_256d = _mm256_add_pd(re_256d, real_256d);
            im_256d = _mm256_add_pd(im_256d, imag_256d);
        }
        double* re = (double*)&re_256d;
        double* im = (double*)&im_256d;
        real_out[i] = re[3] + re[2] + re[1] + re[0];
        imag_out[i] = im[3] + im[2] + im[1] + im[0];
    }
    end_time = clock();
    total_time = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    printf("SIMD calculation time of %zu-point DFT: %f ms.\n", len, total_time);

    return 0;
}

int testAdd(size_t len)
{
    double start_time, end_time, total_time;
    double* x = (double*)_mm_malloc(len * sizeof(double), 64);
    double* y = (double*)_mm_malloc(len * sizeof(double), 64);
    double* z = (double*)_mm_malloc(len * sizeof(double), 64);

    for (size_t i = 0; i < len; i++)
    {
        x[i] = random_value(0.0, 1.0);
        y[i] = random_value(0.0, 1.0);
    }

    // serial
    start_time = clock();
    for (size_t i = 0; i < len; i++)
    {
        z[i] = x[i] + y[i];
    }
    end_time = clock();
    total_time = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    printf("Serial calculation time of %zu additions: %f ms.\n", len, total_time);

    // avx parallel
    start_time = clock();
    __m256d avx_vec_1, avx_vec_2;
    size_t i = 0;
    for (; i < len; i += 4)
    {
        avx_vec_1 = _mm256_load_pd(x + i);
        avx_vec_2 = _mm256_load_pd(y + i);
        avx_vec_1 = _mm256_add_pd(avx_vec_1, avx_vec_2);
        _mm256_store_pd(z + i, avx_vec_1);
    }
    for (; i < len; i++)
    {
        z[i] = x[i] + y[i];
    }
    end_time = clock();
    total_time = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    printf("SIMD calculation time of %zu additions: %f ms.\n", len, total_time);
    _mm_free(x);
    _mm_free(y);
    _mm_free(z);

    return 0;
}

int main()
{
    testAdd(65536);
    testNaiveDFT(4096);
    return 0;
}
