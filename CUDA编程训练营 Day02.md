# 上午部分

## GPU内存分配函数

CPU：`malloc()`  `memset()`  `free()`

GPU: `cudaMalloc()`  `cudaMemset()`  `cudaFree()`

cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)

- dst: destination memory address
- src: source memory address
- count: size in bytes to copy
- kind: direction of the copy

该函数中包含同步函数。

cudaMemcpyKind:

- cudaMemcpyHostToDevice
- cudaMemcpyDeviceToHost
- cudaMemcpyDeviceToDevice
- cudaMemcpyHostToHost

### 矩阵相乘案例

```c
int main(int argc, char const *argv[]){
    int m = 100;
    int n = 100;
    int k = 100;
    
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
    cudaMallocHost((void **) &h_c, sizeof(int)*m*k);
    cudaMallocHost((void **) &h_cc, sizeof(int)*m*n);
    // &h_a取了h_a 这个指针的地址，赋给了cudaMallocHost这个方法的形参，所以要使用 void **，第一个*是形参取到了h_a这个指针中所存放的数据（就是h_a这个指针指向的地址），第二个*才可以真正取到要分配的空间的地址。
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            h_a[i * n +j] = rand() % 1024;
        }
    }
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            h_b[i * n +j] = rand() % 1024;
        }
    }
    
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int)*n*k);
    cudaMalloc((void **) &d_c, sizeof(int)*m*k);
    
    cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);
    
    unsigned int grid_rows = (m + BLOCK_SIZE - 1)/BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1)/BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
}

__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int n. int k){
    int row =blockIdx.y*blockDim,y + threadIdx.y;
    int col =blockIdx.x*blockDim,x + threadIdx.x;
    int sum = 0;
    if(col < k && row < m){
        for(int i = 0; i < n; i++){
            sum = a[row*n+i] * a[i*k +col]
        }
        c[row * k + col] = sum;
    }
}
```



## CUDA的运行时检测函数

`__host__device__const char*  cudaGetErrorName ( cudaError_t error )`

Returns the string representation of an error code enum name.

` __host_device__const char*  cudaGetErrorString ( cudaError_t error）`

Returns the description string for an error code.

`__host__device __cudaError_t  cudaGetLastError ( void )`

Returns the last error from a runtime call.

`__host_device__cudaError_t  cudaPeekAtLastError ( void )`

Returns the last error from a runtime call.



```c
#pragma once
#include <stdio.h>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

```



## CUDA的事件(event)

CUDAevent本质是一个GPU时间戳，这个时间戳是在用户指定的时间点上记录的由于GPU本身支持记录时间戳，因此就避免了当使用CPU定时器来统计GPU执行时间时可能遇到的诸多问题。





# 下午部分

Global memory 所有线程都可以访问的存储单元

Texture cahche 所有线程都可以访问的存储单元 但是只能可读

Constant chache 所有线程都可以访问的存储单元 但是只能可读

Shared memory block内共享存储单元

Local memory 线程私有存储单元

Registers 线程私有存储单元



- Shared memory 和 Registers是速度最快的， Shared memory 速度无限接近与Registers，但不如它。
- Shared memory， Registers， Constant chache， Texture cahche属于onchip memory，位于GPU的芯片上，速度较快，存储空间较小。
- Local memory，Global memory，Constant chache，Texture cahche位于PCB板子上，属于onboard memory。

## Registers:

寄存器是GPU最快的memory，kernel中没有什么特殊声明的自动变量都是放在寄存器中的。当数组的索引是constant类型且在编译期能被确定的话，就是内置类型，数组也是放在寄存器中。

- 寄存器变量是每个线程私有的，一旦thread执行结束，寄存器变量就会失效。
- 寄存器是稀有资源。(省着点用，能让更多的block驻留在SM中，增加Occupancy)
- --maxrregcount 可以设置大小
- 不同设备架构，数量不同

## Shared Memory:

用 shared 修饰符修饰的变量存放在shared memory：

- On-chip
- 拥有高的多bandwidth和低很多的latencyo
- 同一个Block中的线程共享一块Shared Memory。
- __syncthreads()同步。
- 比较小，要节省着使用，不然会限制活动warp的数量

## Local Memory:

有时候，Registers 不够了，就会用Local Memory 来替代。但是，更多在以下情况，会使用LocalMemory.会在Registers 不够的时候自动分配到Local Memory

- 无法确定其索引是否为常量的数组。
- 会消耗太多寄存器空间的大型结构或数组。
- 如果内核使用了多于可用寄存器的任何变量(这也称为寄存器溢出)
- --ptxas-options=-V

## Constant Memory:

固定内存空间驻留在设备内存中，并缓存在固定缓存中(constant cache)，一般保存不会改变的数据。

- constant的范围是全局的，针对所有kernel。

- 在同一个编译单元，constant对所有kernel可见。

- kernel只能从constant Memory读取数据，因此其初始化必须在host端使用下面的function调用:

- cudaError_t  cudaMemcpyToSymbol（const void* symbol, const void* src, size _t count）:

- 当一个warp中所有thread都从同一个Memory地址读取数据时，constant Memory表现会非常好会触发广播机制。

## Texture Memory:

Texture Memory驻留在device Memory中，并且使用一个只读cache。Texture Mmeory是专门为那些在内存访问模式中存在大量空间局部性 (SpatialLocality)的图形应用程序而设计的。意思是，在某个计算应用程序中，这意味着一个Thread读取的位置可能与邻近Thread读取的位置“非常接近

- Texture Memory实际上也是qlobal Memory在一块，但是他有自己专有的只读cache。
- 纹理内存也是缓存在片上的，因此一些情况下相比从芯片外的DRAM上获取数据，纹理内存可以通过减少内存请求来提高带宽。
- 从数学的角度，下图中的4个地址并非连续的，在一般的CPU缓存中，这些地址将不会缓存。但由于GPU纹理缓存是专门为了加速这种访问模式而设计的，因此如果在这种情况中使用纹理内存而不是全局内存，那么将会获得性能的提升

## Global Memory:

空间最大，latency(延迟)最高，GPU最基础的memory:

- 驻留在Device memory中
- memory transaction对齐，合并访存



```c
__global__ void gpu_matrix_mult_shared(int *d_a, int *d_b, int *d_result, int m, int n, int k) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        tile_a[threadIdx.y][threadIdx.x] = row<n && (sub * BLOCK_SIZE + threadIdx.x)<n? d_a[idx]:0;
        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        tile_b[threadIdx.y][threadIdx.x] = col<n && (sub * BLOCK_SIZE + threadIdx.y)<n? d_b[idx]:0;
        
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}
```

