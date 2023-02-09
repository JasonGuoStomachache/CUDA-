# 上午部分

## 原子操作：

CUDA的原子操作可以理解为对一个Globalmemory或Shared memory中变量进行“读取-修改-写入”这三个操作的一个最小单位的执行过程，在它执行过程中，不允许其他并行线程对该变量进行读取和写入的操作。

基于这个机制，原子操作实现了对在多个线程间共享的变量的互斥保护，确保任何一次对变量的操作的结果的正确性。



## 原子操作常用函数

|                        |                                                              |
| :--------------------: | :----------------------------------------------------------: |
| atomicAdd(&value,num)  |                  加法: value = value + num                   |
| atomicSub(&value,num)  |                   减法: value =value - num                   |
| atomicExch(&value,num) |                      赋值: value = num                       |
| atomicMax(&value,num)  |                求最大: value = max[value,num)                |
| atomicMin(&value,num)  |                求最小: value = min[value,num)                |
| atomicInc(&value,num)  |      向上计数: 如果(value <= num)则value++,否则value =0      |
| atomicDec(&value, num) | 向下计数: 如果(value>num或value == 0),则value--,否则value =0 |
| atomicCAS(&value,num)  |          比较并交换:如果(value != num],则value =num          |
| atomicAnd(&value,num)  |                与运算: value = value and num                 |
| atomicOr(&value, num]  |                  或运算 value =value or num                  |
| atomicXor(&value,num)  |                异或运算 value =value xor num                 |

# CUDA编程模型--- 原子操作
#### 原子操作
原子函数对驻留在全局或共享内存中的一个 32 位或 64 位字执行读-修改-写原子操作。
#### 1. atomicAdd()
```C++
int atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicAdd(unsigned long long int* address,
                                 unsigned long long int val);
float atomicAdd(float* address, float val);
double atomicAdd(double* address, double val);
__half2 atomicAdd(__half2 *address, __half2 val);
__half atomicAdd(__half *address, __half val);
__nv_bfloat162 atomicAdd(__nv_bfloat162 *address, __nv_bfloat162 val);
__nv_bfloat16 atomicAdd(__nv_bfloat16 *address, __nv_bfloat16 val);
```
读取位于全局或共享内存中地址 `address` 的 16 位、32 位或 64 位字 `old`，计算 `(old + val)`，并将结果存储回同一地址的内存中。这三个操作在一个原子事务中执行。该函数返回`old`。

`atomicAdd()` 的 32 位浮点版本仅受计算能力 2.x 及更高版本的设备支持。

`atomicAdd()` 的 64 位浮点版本仅受计算能力 6.x 及更高版本的设备支持。

`atomicAdd()` 的 32 位 `__half2` 浮点版本仅受计算能力 6.x 及更高版本的设备支持。 `__half2` 或 `__nv_bfloat162` 加法操作的原子性分别保证两个 `__half` 或 `__nv_bfloat16` 元素中的每一个；不保证整个 `__half2` 或 `__nv_bfloat162` 作为单个 32 位访问是原子的。

`atomicAdd()` 的 16 位 `__half` 浮点版本仅受计算能力 7.x 及更高版本的设备支持。

`atomicAdd()` 的 16 位 `__nv_bfloat16` 浮点版本仅受计算能力 8.x 及更高版本的设备支持。

####  2. atomicSub()
```C++
int atomicSub(int* address, int val);
unsigned int atomicSub(unsigned int* address,
                       unsigned int val);
```
读取位于全局或共享内存中地址`address`的 32 位字 `old`，计算 `(old - val)`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

####  3. atomicExch()
```C++
int atomicExch(int* address, int val);
unsigned int atomicExch(unsigned int* address,
                        unsigned int val);
unsigned long long int atomicExch(unsigned long long int* address,
                                  unsigned long long int val);
float atomicExch(float* address, float val);
```
读取位于全局或共享内存中地址address的 32 位或 64 位字 `old` 并将 `val` 存储回同一地址的内存中。 这两个操作在一个原子事务中执行。 该函数返回`old`。

####  4. atomicMin()
```C++
int atomicMin(int* address, int val);
unsigned int atomicMin(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMin(unsigned long long int* address,
                                 unsigned long long int val);
long long int atomicMin(long long int* address,
                                long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `old` 和 `val` 的最小值，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicMin()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

####  5. atomicMax()
```C++
int atomicMax(int* address, int val);
unsigned int atomicMax(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMax(unsigned long long int* address,
                                 unsigned long long int val);
long long int atomicMax(long long int* address,
                                 long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `old` 和 `val` 的最大值，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicMax()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

####  6. atomicInc()
```C++
unsigned int atomicInc(unsigned int* address,
                       unsigned int val);
```

读取位于全局或共享内存中地址`address`的 32 位字 `old`，计算 `((old >= val) ? 0 : (old+1))`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

####  7. atomicDec()
```C++
unsigned int atomicDec(unsigned int* address,
                       unsigned int val);
```
读取位于全局或共享内存中地址`address`的 32 位字 `old`，计算 `(((old == 0) || (old > val)) ? val : (old-1) )`，并将结果存储回同一个地址的内存。 这三个操作在一个原子事务中执行。 该函数返回`old`。

####  8. atomicCAS()
```C++
int atomicCAS(int* address, int compare, int val);
unsigned int atomicCAS(unsigned int* address,
                       unsigned int compare,
                       unsigned int val);
unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int compare,
                                 unsigned long long int val);
unsigned short int atomicCAS(unsigned short int *address, 
                             unsigned short int compare, 
                             unsigned short int val);
```
读取位于全局或共享内存中地址`address`的 16 位、32 位或 64 位字 `old`，计算 `(old == compare ? val : old)` ，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`（Compare And Swap）。

###   Bitwise Functions

####  9. atomicAnd()
```C++
int atomicAnd(int* address, int val);
unsigned int atomicAnd(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicAnd(unsigned long long int* address,
                                 unsigned long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `(old & val)`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicAnd()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

####  10. atomicOr()
```C++
int atomicOr(int* address, int val);
unsigned int atomicOr(unsigned int* address,
                      unsigned int val);
unsigned long long int atomicOr(unsigned long long int* address,
                                unsigned long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `(old | val)`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicOr()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

####  11. atomicXor()
```C++
int atomicXor(int* address, int val);
unsigned int atomicXor(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicXor(unsigned long long int* address,
                                 unsigned long long int val);
```
读取位于全局或共享内存中地址`address`的 32 位或 64 位字 `old`，计算 `(old ^ val)`，并将结果存储回同一地址的内存中。 这三个操作在一个原子事务中执行。 该函数返回`old`。

`atomicXor()` 的 64 位版本仅受计算能力 3.5 及更高版本的设备支持。

## 作业

接下来我们完成下面的一个实例：  
给定一个数组A，它包含1000000个int类型的元素，求他所有的元素之和：  
输入：A[1000000]  
输出：output（A中所有元素之和）  

```c
#include<stdio.h>
#include<stdint.h>
#include<time.h>     //for time()
#include<stdlib.h>   //for srand()/rand()
#include<sys/time.h> //for gettimeofday()/struct timeval
#include"error.cuh"

#define N 10 000 000
#define BLOCK_SIZE 256
#define BLOCKS 32 


__managed__ int source[N];               //input data
__managed__ int final_result[1] = {0};   //scalar output


__global__ sum_all_gpu(int *input, int count, int *output)
{
    __shared__ int sum_for_per_block[BLOCK_SIZE];
    
    int read_temp = 0;
    for(int idx = threadIdx.x + blockDim.x * blockIdx.x;
       idx < count;
       idx += gridDim.x * blockDim.x){
        read_temp += input[idx];
    }
    sum_for_per_block[threadidx.x] = read_temp;
    __syncthreads();
    
    for(int length = BLOCK_SIZE/2; length>=1; length /= 2){
        int sum_temp = -1;
        if(threadIdx.x<length){
            sum_temp = sum_for_per_block[threadIdx.x] 
                + sum_for_per_block[threadIdx.x + length];
        }
        __syncthreads();
        if(threadIdx.x<length){
            sum_for_per_block[threadIdx.x] = sum_temp;
        }
        __snycthreads();
    }
    
    if(blockDim.x * blockIdx.x < count){
        if(threadIdx.x == 0) atomicAdd(output, sum_for_per_block[0]);
    }
    
    
}

void _init(int *ptr, int count)
{
     unit32_t seed = (unit32_t)time(null);
     srand(seed);
     for (int i = 0; i < count; i++) ptr[i] = rand();
}

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_usec * 0.000001 + tv.tv_sec);
}

int main()
{
    //**********************************
    fprintf(stderr, "filling the buffer with %d elements...\n", N);
    _init(source, N);

    //**********************************
    //Now we are going to kick start your kernel.
    cudaDeviceSynchronize(); //steady! ready! go!
    
    fprintf(stderr, "Running on GPU...\n");
    
double t0 = get_time();
    _sum_gpu<<<BLOCKS, BLOCK_SIZE>>>(source, N, final_result);
    CHECK(cudaGetLastError());  //checking for launch failures
    CHECK(cudaDeviceSynchronize()); //checking for run-time failurs
double t1 = get_time();

    int A = final_result[0];
    fprintf(stderr, "GPU sum: %u\n", A);


    //**********************************
    //Now we are going to exercise your CPU...
    fprintf(stderr, "Running on CPU...\n");

double t2 = get_time();
    int B = _sum_cpu(source, N);
double t3 = get_time();
    fprintf(stderr, "CPU sum: %u\n", B);

    //******The last judgement**********
    if (A == B)
    {
        fprintf(stderr, "Test Passed!\n");
    }
    else
    {
        fprintf(stderr, "Test failed!\n");
	exit(-1);
    }
    
    //****and some timing details*******
    fprintf(stderr, "GPU time %.3f ms\n", (t1 - t0) * 1000.0);
    fprintf(stderr, "CPU time %.3f ms\n", (t3 - t2) * 1000.0);

    return 0;
}	
```

编译，并执行程序
!make
!./sum
利用nvprof测试程序性能
!sudo /usr/local/cuda/bin/nvprof ./sum
课后作业：

1. 给定数组A[1000000]找出其中最大的值和最小的值
2. 给定数组A[1000000]找出其中最大的十个值
3. 修改02_2.4中的[im2gray.cu](im2gray.cu)文件, 完成灰度直方图的统计任务. 如果遇到问题, 请参考[cuda_hist.cu](cuda_hist.cu)
!/usr/local/cuda/bin/nvcc cuda_hist.cu -L /usr/lib/aarch64-linux-gnu/libopencv*.so -I /usr/include/opencv4 -o cuda_hist
!./cuda_hist





# 下午部分

## Unified Memory（统一内存）:

统一内存是可从系统中的任何处理器访问的单个内存地址空间。这种硬件/软件技术允许应用程序分配可以从CPUs或GPUs 上运行的代码读取或写入的数据。分配统一内存非常简单，只需将对 mallocll或 new 的调用替换为对 cudaMallocManaged] 的调用，这是一个分配函数，返回可从任何处理器访问的指针。

## Unified Memory: 两种实现方法

1. `cudaError_t  cudaMallocManaged(void **devPtr, size t size, unsigned int flags=0);`
2. `__managed__`



基于ARM平台的JETSON NANO的存储单元特点:

JETSON NANO 将GPU和CPU整合到了一起。可以不使用同意内存就互相访问。

## 作业

使用统一内存，优化矩阵乘法：

```c
#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define BLOCK_SIZE 16
__managed__ int a[1000 * 1000];
__managed__ int b[1000 * 1000];
__managed__ int c_gpu[1000 * 1000];
__managed__ int c_cpu[1000 * 1000];

__global__ void gpu_matrix_mult_shared(int* d_a, int* d_b, int* d_result, int M, int N, int K)
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub <= N/BLOCK_SIZE; ++sub)
    {
        int r = row;
        int c = sub * BLOCK_SIZE + threadIdx.x;
        idx = r * N + c;

        if (r >= M || c >= N)
        {
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        r = sub * BLOCK_SIZE + threadIdx.y;
        c = col;
        idx = r * K + c;
        if (c >= K || r >= N)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < K)
    {
        d_result[row * K + col] = tmp;
    }
}

int main(int argc, char const* argv[])
{
    int m = 1000;
    int n = 1000;
    int k = 1000;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = 0*rand() % 1024+1;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            b[i * k + j] = 0 * rand() % 1024 +1;
        }
    }


    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


    gpu_matrix_mult_shared << <dimGrid, dimBlock >> > (a, b, c_gpu, m, n, k);

    return 0;
}
```

