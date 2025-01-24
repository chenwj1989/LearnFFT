
## SIMD of loops

## SIMD inside a loop
在信号处理应用中，复数乘法是一项必须反复执行的耗时操作。我不会深入讨论这个理论，但每个复数都可以表示为a + bi，其中a和b是浮点值，i是-1的平方根。A是实部，b是虚部。如果(a + bi)和(c + di)相乘，乘积等于(ac - bd) + (ad + bc)i。

复数可以以交错的方式存储，这意味着每个实数部分后面跟着虚数部分。假设vec1是一个__m256d，存储两个复数(a + bi)和(x + yi)， vec2是一个__m256d，存储(c + di)和(z + wi)。图6说明了如何存储这些值。如图所示，prod向量存储了两个产物:(ac - bd) + (ad + bc)i和(xz - yw) + (xw + yz)i。

【图片丢失】

Figure 6: Complex Multiplication Using Vectors

我不知道用AVX/AVX2计算复杂乘积的最快方法。但我想出了一个方法，效果很好。它包括五个步骤:

将vec1和vec2相乘，并将结果存储在vec3中。
切换vec2的实/虚值。
求vec2的虚数的负数。
将vec1和vec2相乘，并将结果存储在vec4中。
对vec3和vec4进行水平相减，得到vec1中的答案。


## Optimization

### 内存对齐
- C: double*	a =(double*)memalign(32,9*sizeof(double));
- C++: double*	a =(double*)aligned_alloc(32,9*sizeof(double));
- intel: double*	a =(double*)_mm_malloc(9*sizeof(double),32);
- gcc: __attribute__ ((aligned(32)))double a[9]  ={1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,2.1};
- 不同内存对齐方式 memcpy之后数据错误
- vector通常会确保其元素按照类型的自然对齐方式存储，这可能导致某些类型的vector占用的内存比严格必要的要多一些。


#### 用户传入buffer不一定是aligned的
- 创建临时buffer，拷贝输入。
- 进行 FFTSIMD
- 将结果拷贝到输入buffer
  
#### 用户传入buffer保证aligned
- 为了保证传入bufferaligned，提供alloc和free函数，自己控制buffer的创建销毁，保证对齐


### interleave real and complex data


### 优化if-else

### 
