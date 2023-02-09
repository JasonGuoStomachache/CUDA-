---
typora-root-url: imgs
---



# CUDA 流的概念

- CUDA流在加速应用程序方面起到重要的作用，他表示一个GPU的操作队列操作在队列中按照一定的顺序执行，也可以向流中添加一定的操作如核函数的启动、内存的复制、事件的启动和结束等，添加的顺序也就是执行的顺序
- 一个流中的不同操作有着严格的顺序。但是不同流之间是没有任何限制的。多个流同时启动多个内核，就形成了网格级别的并行。
- CUDA流中排队的操作和主机都是异步的，所以排队的过程中并不耽误主机运行其他指令，所以这就隐藏了执行这些操作的开销。

## 流支持的并发操作

基于流的异步内核启动和数据传输支持以下类型的粗粒度并发：

- 重叠主机和设备计算
- 重叠主机计算和主机设备数据传输
- 重叠主机设备数据传输和设备计算
- 并发设备计算 (多个设备)

不支持并发:

- a page-locked host memory allocation,
- a device memory allocation,
- a device memory set,
- a memory copy between two addresses to the same device memory.
- any CUDA command to the NULL stream

## 流的创建与销毁

- cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_tcount, cudaMemcpyKind kind, cudaStream t stream = 0);
- cudaError_t cudaStreamCreate(cudaStream t* pStream);
  cudaStream_t a;
- kernel_name<<<grid, block, sharedMemSize, stream>>>(argument list);
- cudaError_t  cudaStreamDestroy(cudaStream t stream);

```c
int *h arr,*d_arr;
cudaStream t stream;
cudaMalloc((void **)&d_arr, nbytes);
cudaMallocHost((void **)&h arr, nbytes);
cudaStreamCreate(&stream);
cudaMemcpyAsync(d_arr, h_arr, nbytes, cudaMemcpyHostToDevice, stream);

kernel<<<grid, block, smem_size, stream>>>();

cudaStreamSynchronize(stream);
cudaFree(d_arr); 
cudaFreeHost(h_arr);
cudaStreamDestroy(stream)
```

![image-20230210020146886](/image-20230210020146886.png)

## 作业

接下来，我们就完成下面这个核函数，在两个流并发的实现：

```c++
__global__ void kernel( int *a, int *b, int *c ) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < N) {
	int idx1 = (idx + 1) % 256;
	int idx2 = (idx + 2) % 256;
	float  as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
	float  bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
	c[idx] = (as + bs) / 2;

  }

}
```



```c++
#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define N   (1024*1024)
#define FULL_DATA_SIZE   (N*20)


__global__ void kernel( int *a, int *b, int *c ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}


int main( void ) {
    

    cudaStream_t    stream0, stream1;
    int *host_a, *host_b, *host_c;
    int *dev_a0, *dev_b0, *dev_c0;
    int *dev_a1, *dev_b1, *dev_c1;

    // allocate the memory on the GPU
    cudaMalloc( (void**)&dev_a0, N * sizeof(int) );
    cudaMalloc( (void**)&dev_b0, N * sizeof(int) );
    cudaMalloc( (void**)&dev_c0, N * sizeof(int) );
    cudaMalloc( (void**)&dev_a1, N * sizeof(int) ); 
    cudaMalloc( (void**)&dev_b1, N * sizeof(int) ); 
    cudaMalloc( (void**)&dev_c1, N * sizeof(int) );

    // allocate host locked memory, used to stream
    cudaHostAlloc( (void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc( (void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc( (void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

    for (int i=0; i<FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    for (int i=0; i<FULL_DATA_SIZE; i+= N*2) {
        // enqueue copies of a in stream0 and stream1
        cudaMemcpyAsync( dev_a0, host_a+i, N * sizeof(int), cudaMemcpyHostToDevice, stream0 );
        cudaMemcpyAsync( dev_a1, host_a+i+N, N * sizeof(int), cudaMemcpyHostToDevice, stream1 );
        // enqueue copies of b in stream0 and stream1
        cudaMemcpyAsync( dev_b0, host_b+i, N * sizeof(int), cudaMemcpyHostToDevice, stream0 );
        cudaMemcpyAsync( dev_b1, host_b+i+N, N * sizeof(int), cudaMemcpyHostToDevice, stream1 );

        kernel<<<N/256,256,0,stream0>>>( dev_a0, dev_b0, dev_c0 );
        kernel<<<N/256,256,0,stream1>>>( dev_a1, dev_b1, dev_c1 );

        cudaMemcpyAsync( host_c+i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0 );
        cudaMemcpyAsync( host_c+i+N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1 );
    }
    cudaStreamSynchronize( stream0 );
    cudaStreamSynchronize( stream1 );


    // cleanup the streams and memory
    cudaFreeHost( host_a );
    cudaFreeHost( host_b );
    cudaFreeHost( host_c );
    cudaFree( dev_a0 );
    cudaFree( dev_b0 );
    cudaFree( dev_c0 );
    cudaFree( dev_a1 );
    cudaFree( dev_b1 );
    cudaFree( dev_c1 );
    cudaStreamDestroy( stream0 );
    cudaStreamDestroy( stream1 );

    return 0;
}

```



# CUDA库

