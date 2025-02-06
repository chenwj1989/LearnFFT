#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <time.h>

double start_time, end_time, total_time;

// nvcc --compiler-options=-Wall -g  test_cuda.cu -o test_cuda -lm

#define CHECK(call)                                                                                \
    {                                                                                              \
        const cudaError_t error = call;                                                            \
        if (error != cudaSuccess)                                                                  \
        {                                                                                          \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);                                           \
            printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));                       \
            exit(1);                                                                               \
        }                                                                                          \
    }tests/test_cuda.cu

__global__ void kernelHelloWorld(void)
{
    printf("Hello world! GPU threadIdx: %d, blockIdx: %d, blockDim: %d, gridDim: %d.\n",
           threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
}

__global__ void kernelSineCosine(double* d_sin, double* d_cos, int len)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len  || j >= len) return;

    int p = i * len + j;
    double arg = (double(i) * double(j) * M_PI * 2.0) / len;
    d_sin[p] = sin(arg);
    d_cos[p] = cos(arg);
}

__global__ void kernelDFT(const double* real_in, const double* imag_in, double* real_out, double* imag_out, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    double re = 0.0, im = 0.0;
    for (int j = 0; j < len; ++j) {
        double arg = (double(i) * double(j) * M_PI * 2.0) / len;
        double _sin = cos(arg);
        double _cos = cos(arg);
        re += real_in[j] * _cos + imag_in[j] * _sin;
        im -= real_in[j] * _sin - imag_in[j] * _cos;
    }
    real_out[i] = re;
    imag_out[i] = im;
}

int testSineCosine(int len)
{
    int len_lut = len * len;
    std::vector<double> lut_sin(len_lut);
    std::vector<double> lut_cos(len_lut);
    start_time = clock();
    for (int i = 0; i < len; ++i)
    {
        for (int j = 0, p = i * len; j < len; ++j, ++p)
        {
            double arg = (double(i) * double(j) * M_PI * 2.0) / len;
            lut_sin[p] = sin(arg);
            lut_cos[p] = cos(arg);
        }
    }
    end_time = clock();
    total_time = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    printf("CPU calculation time of sine and cos LUT: %f ms.\n", total_time);

    std::vector<double> h_sin(len_lut);
    std::vector<double> h_cos(len_lut);
    start_time = clock();
    int n_bytes = len_lut * sizeof(double);
    double *d_sin;
    double *d_cos;
    cudaMalloc((void**)&d_sin, n_bytes);
    cudaMalloc((void**)&d_cos, n_bytes);

    // dim2 blockSize(32, 32);
    // dim2 gridSize(len / 32, len / 32);
    int tx = 32;
    int ty = 32;
    int bx = (len + tx - 1) / tx;
    int by = (len + ty - 1) / ty;
    const dim3 blockSize(tx, ty);
    const dim3 gridSize(bx, by);
    kernelSineCosine<<<gridSize, blockSize>>>(d_sin, d_cos, len);
    cudaMemcpy((void*)h_sin.data(), (void*)d_sin, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)h_cos.data(), (void*)d_cos, n_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_sin);
    cudaFree(d_cos);

    end_time = clock();
    total_time = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    printf("GPU calculation time of sine and cos LUT: %f ms.\n", total_time);

    // Calculate ERROR
    double err_sin = 0.0;
    double err_cos = 0.0;
    for (int p = 0; p < len_lut; ++p)
    {
        err_sin += abs(lut_sin[p] - h_sin[p]) / len_lut;
        err_cos += abs(lut_cos[p] - h_cos[p]) / len_lut;
    }

    printf("Error between GPU and CPU calculation: ");
    printf("sine: %f, cosine: %f\n", err_sin, err_cos);
    return 0;
}

int testNaiveDFT(int len)
{
    std::vector<double> real_in(len);
    std::vector<double> imag_in(len);
    std::vector<double> real_out(len);
    std::vector<double> imag_out(len);
    start_time = clock();
    for (int i = 0; i < len; ++i)
    {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < len; ++j) {
            double arg = (double(i) * double(j) * M_PI * 2.0) / len;
            double _sin = cos(arg);
            double _cos = cos(arg);
            re += real_in[j] * _cos + imag_in[j] * _sin;
            im -= real_in[j] * _sin - imag_in[j] * _cos;
        }
        real_out[i] = re;
        imag_out[i] = im;
    }
    end_time = clock();
    total_time = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    printf("CPU calculation time of DFT: %f ms.\n", total_time);

    std::vector<double> h_real(len);
    std::vector<double> h_imag(len);
    start_time = clock();
    int n_bytes = len * sizeof(double);
    double *d_real_in;
    double *d_imag_in;
    double *d_real_out;
    double *d_imag_out;
    cudaMalloc((void**)&d_real_in, n_bytes);
    cudaMalloc((void**)&d_imag_in, n_bytes);
    cudaMalloc((void**)&d_real_out, n_bytes);
    cudaMalloc((void**)&d_imag_out, n_bytes);

    cudaMemcpy((void*)d_real_in, (void*)real_in.data(), n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_imag_in, (void*)imag_in.data(), n_bytes, cudaMemcpyHostToDevice);
    int tx = 1024;
    int bx = (len + tx - 1) / tx;
    const dim3 blockSize(tx);
    const dim3 gridSize(bx);
    kernelDFT<<<gridSize, blockSize>>>(d_real_in, d_imag_in, d_real_out, d_imag_out, len);
    cudaMemcpy((void*)h_real.data(), (void*)d_real_out, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)h_imag.data(), (void*)d_imag_out, n_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_real_in);
    cudaFree(d_imag_in);
    cudaFree(d_real_out);
    cudaFree(d_imag_out);

    end_time = clock();
    total_time = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    printf("GPU calculation time of DFT: %f ms.\n", total_time);

    // Calculate ERROR
    double err_real = 0.0;
    double err_imag = 0.0;
    for (int p = 0; p < len; ++p)
    {
        err_real += abs(real_out[p] - h_real[p]) / len;
        err_imag += abs(imag_out[p] - h_imag[p]) / len;
    }

    printf("Error between GPU and CPU calculation: ");
    printf("real: %f, imag: %f\n", err_real, err_imag);
    return 0;
}

int main(int argc, char** argv)
{
    printf("===>Test GPU Hello world!\n");
    kernelHelloWorld<<<1, 10>>>();
    cudaDeviceSynchronize();
    
    printf("===>Test GPU Sine/Cosine!\n");
    testSineCosine(1024); 

    printf("===>Test GPU DFT!\n");
    testNaiveDFT(1024);

    int dev = 0;
    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, dev));
    printf("===>Print GPU Properties!\n");
    printf("Using GPU device %d : %s\n", dev, devProp.name);
    printf("Streaming Multi-Processor Count: %d\n", devProp.multiProcessorCount);
    printf("Shared Memory Per Block: %f KB\n", devProp.sharedMemPerBlock / 1024.0);
    printf("Max Threads Per Block: %d\n", devProp.maxThreadsPerBlock);
    printf("Max Threads Per Multi-Processor: %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("Max Warps Per Multi-Processor: %d\n", devProp.maxThreadsPerMultiProcessor / 32);

    cudaDeviceReset(); // if no this line ,it can not output hello world from gpu
    return 0;
}