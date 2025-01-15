
#include "dft.h"
#include "kissfft/kiss_fft.h"
#include "kissfft/kiss_fftr.h"
#include <iostream>
#include <memory>
#include <string>
#include <time.h>

#ifndef learn_fft_scalar
#define learn_fft_scalar double
#endif

#define FFT_SIZE 1024
#define NUM_LOOPS 100

using namespace learnfft;

double start_time, end_time;

static learn_fft_scalar random_value(learn_fft_scalar min, learn_fft_scalar max)
{
    return ((learn_fft_scalar)rand() / RAND_MAX) * (max - min) + min;
}

int RunKissfft(const learn_fft_scalar* in_real, const learn_fft_scalar* in_imag,
               learn_fft_scalar* out_real, learn_fft_scalar* out_imag)
{
    kiss_fft_cpx in_cpx[FFT_SIZE];
    kiss_fft_cpx out_cpx[FFT_SIZE];

    kiss_fft_cfg forward_fft;
    kiss_fft_cfg inverse_fft;
    forward_fft = kiss_fft_alloc(FFT_SIZE, 0, 0, 0);
    inverse_fft = kiss_fft_alloc(FFT_SIZE, 1, 0, 0);

    double total_time = 0;
    for (int l = 0; l < NUM_LOOPS; l++)
    {

        for (int i = 0; i < FFT_SIZE; i++)
        {
            in_cpx[i].r = in_real[i];
            in_cpx[i].i = in_imag[i];
        }
        start_time = clock();
        kiss_fft(forward_fft, in_cpx, out_cpx);
        end_time = clock();
        for (int i = 0; i < FFT_SIZE; i++)
        {
            out_real[i] = out_cpx[i].r;
            out_imag[i] = out_cpx[i].i;
        }
        total_time += end_time - start_time;
    }

    // print time cost
    std::cout << "FFT time per pass: " << (total_time / NUM_LOOPS) * 1000 / CLOCKS_PER_SEC << " ms."
              << std::endl;

    // print out-in error
    learn_fft_scalar error = 0.0;
    kiss_fft(inverse_fft, out_cpx, in_cpx);
    for (int i = 0; i < FFT_SIZE; i++)
    {
        error += abs(in_cpx[i].r / FFT_SIZE - in_real[i]);
        error += abs(in_cpx[i].i / FFT_SIZE - in_imag[i]);
    }
    error /= FFT_SIZE * 2;
    std::cout << "FFT->IFFT v.s. input error per sample: " << error << std::endl;
    KISS_FFT_FREE(forward_fft);
    KISS_FFT_FREE(inverse_fft);
    return 0;
}

int RunKissfftr(learn_fft_scalar* in, learn_fft_scalar* out_real, learn_fft_scalar* out_imag)
{
    // learn_fft_scalar in[FFT_SIZE];
    learn_fft_scalar out[FFT_SIZE];
    kiss_fft_cpx out_cpx[FFT_SIZE];

    kiss_fftr_cfg forward_fft;
    kiss_fftr_cfg inverse_fft;
    forward_fft = kiss_fftr_alloc(FFT_SIZE, 0, 0, 0);
    inverse_fft = kiss_fftr_alloc(FFT_SIZE, 1, 0, 0);

    double total_time = 0;
    for (int l = 0; l < NUM_LOOPS; l++)
    {
        start_time = clock();
        kiss_fftr(forward_fft, in, out_cpx);
        end_time = clock();
        for (int i = 0; i < FFT_SIZE; i++)
        {
            out_real[i] = out_cpx[i].r;
            out_imag[i] = out_cpx[i].i;
        }
        total_time += end_time - start_time;
    }

    // print time cost
    std::cout << "FFT time per pass: " << (total_time / NUM_LOOPS) * 1000 / CLOCKS_PER_SEC << " ms."
              << std::endl;

    // check reverse error
    kiss_fftri(inverse_fft, out_cpx, out);
    learn_fft_scalar error = 0.0;
    for (int i = 0; i < FFT_SIZE; i++)
    {
        error += abs(out[i] / FFT_SIZE - in[i]);
    }
    error /= FFT_SIZE;
    std::cout << "FFT->IFFT v.s. input error per sample: " << error << std::endl;
    kiss_fftr_free(forward_fft);
    kiss_fftr_free(inverse_fft);
    return 0;
}

int RunDft(const learn_fft_scalar* in_real, const learn_fft_scalar* in_imag,
           learn_fft_scalar* out_real, learn_fft_scalar* out_imag)
{
    learn_fft_scalar tmp_real[FFT_SIZE];
    learn_fft_scalar tmp_imag[FFT_SIZE];

    DFT<learn_fft_scalar> my_dft(FFT_SIZE);

    double total_time = 0;
    for (int l = 0; l < NUM_LOOPS; l++)
    {
        start_time = clock();
        my_dft.Forward(in_real, in_imag, out_real, out_imag);
        end_time = clock();
        total_time += end_time - start_time;
    }

    // print time cost
    std::cout << "FFT time per pass: " << (total_time / NUM_LOOPS) * 1000 / CLOCKS_PER_SEC << " ms."
              << std::endl;

    // print out-in error
    learn_fft_scalar error = 0.0;
    my_dft.Inverse(out_real, out_imag, tmp_real, tmp_imag);
    for (int i = 0; i < FFT_SIZE; i++)
    {
        error += abs(tmp_real[i] / FFT_SIZE - in_real[i]);
        error += abs(tmp_imag[i] / FFT_SIZE - in_imag[i]);
    }

    error /= FFT_SIZE * 2;
    std::cout << "DFT->IDFT v.s. input error per sample: " << error << std::endl;
    return 0;
}

int RunDftr(const learn_fft_scalar* in_real, learn_fft_scalar* out_real, learn_fft_scalar* out_imag)
{
    learn_fft_scalar tmp_real[FFT_SIZE];

    DFT<learn_fft_scalar> my_dft(FFT_SIZE);

    double total_time = 0;
    for (int l = 0; l < NUM_LOOPS; l++)
    {
        start_time = clock();
        my_dft.ForwardReal(in_real, out_real, out_imag);
        end_time = clock();
        total_time += end_time - start_time;
    }

    // print time cost
    std::cout << "FFT time per pass: " << (total_time / NUM_LOOPS) * 1000 / CLOCKS_PER_SEC << " ms."
              << std::endl;

    // print out-in error
    learn_fft_scalar error = 0.0;
    my_dft.InverseReal(out_real, out_imag, tmp_real);
    for (int i = 0; i < FFT_SIZE; i++)
    {
        error += abs(tmp_real[i] / FFT_SIZE - in_real[i]);
    }

    error /= FFT_SIZE;
    std::cout << "DFTReal->IDFTReal v.s. input error per sample: " << error << std::endl;
    return 0;
}

int Test()
{
    learn_fft_scalar in_real[FFT_SIZE];
    learn_fft_scalar in_imag[FFT_SIZE];
    learn_fft_scalar out_real[FFT_SIZE];
    learn_fft_scalar out_imag[FFT_SIZE];
    learn_fft_scalar out_real_cmp[FFT_SIZE];
    learn_fft_scalar out_imag_cmp[FFT_SIZE];
    // set in buff with rand value
    for (int i = 0; i < FFT_SIZE; i++)
    {
        in_real[i] = random_value(0.0, 1.0);
        in_imag[i] = random_value(0.0, 1.0);
    }
    std::cout << std::endl << "=====> Run Kiss FFT " << std::endl;
    RunKissfft(in_real, in_imag, out_real_cmp, out_imag_cmp);
    std::cout << std::endl;

    std::cout << std::endl << "=====> Run DFT " << std::endl;
    RunDft(in_real, in_imag, out_real, out_imag);
    std::cout << std::endl;

    learn_fft_scalar error = 0.0;
    for (int i = 0; i < FFT_SIZE; i++)
    {
        error += abs(out_real_cmp[i] - out_real[i]);
        error += abs(out_imag_cmp[i] - out_imag[i]);
    }
    error /= FFT_SIZE * 2;
    std::cout << "DFT v.s. KissFFT error per sample: " << error << std::endl;

    // Real FFT
    // set in buff with rand value
    for (int i = 0; i < FFT_SIZE; i++)
    {
        in_imag[i] = 0.0;
    }
    std::cout << std::endl << "=====> Run Kiss FFTR " << std::endl;
    RunKissfftr(in_real, out_real_cmp, out_imag_cmp);
    std::cout << std::endl;

    std::cout << std::endl << "=====> Run DFT Real " << std::endl;
    RunDftr(in_real, out_real, out_imag);
    std::cout << std::endl;

    error = 0.0;
    for (int i = 0; i < FFT_SIZE / 2 + 1; i++)
    {
        error += abs(out_real_cmp[i] - out_real[i]);
        error += abs(out_imag_cmp[i] - out_imag[i]);
    }
    error /= FFT_SIZE * 2;
    std::cout << "DFTReal v.s. KissFFTr error per sample: " << error << std::endl;
    return 0;
}

int main(int argc, const char* argv[]) { Test(); }
