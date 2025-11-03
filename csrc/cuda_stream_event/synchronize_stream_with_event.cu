#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel1(int64_t* data, int64_t repeat)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++) { data[idx] += 1; }
}

__global__ void kernel2(int64_t* data, int64_t repeat)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++) { data[idx] += 2; }
}

__global__ void kernel3(int64_t* data, int64_t repeat)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++) { data[idx] -= 1; }
}

int main()
{
    const int dataSize  = 1024;
    const int printSize = 10;
    int64_t*  h_data    = new int64_t[dataSize];   // host data
    int64_t * d_data1, *d_data2;                   // device data

    // initialize host data
    for (size_t i = 0; i < dataSize; i++) { h_data[i] = 0; }

    // allocate device data
    cudaMalloc((void**)&d_data1, dataSize * sizeof(int64_t));
    cudaMalloc((void**)&d_data2, dataSize * sizeof(int64_t));

    // transfer host data to device
    cudaMemcpy(d_data1, h_data, dataSize * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, h_data, dataSize * sizeof(int64_t), cudaMemcpyHostToDevice);

    // define grid and block dimensions
    dim3 blockDim(256);
    dim3 gridDim((dataSize + blockDim.x - 1) / blockDim.x);

    // create stream and event
    cudaStream_t stream1, stream2;
    cudaEvent_t  event1;
    int          priorityHigh, priorityLow;
    cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh);
    cudaStreamCreateWithFlags(&stream1, cudaStreamDefault);
    cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, priorityHigh);
    cudaEventCreate(&event1);

    const int64_t repeat = 1000;

    // 在 stream1 上执行 kernel1
    kernel1<<<gridDim, blockDim, 0, stream1>>>(d_data1, repeat);
    cudaEventRecord(event1, stream1);

    // stream2 等待 event1，并执行 kernel2
    cudaStreamWaitEvent(stream2, event1, 0);
    kernel2<<<gridDim, blockDim, 0, stream2>>>(d_data1, repeat);

    // 在 stream1 上执行 kernel3
    kernel3<<<gridDim, blockDim, 0, stream1>>>(d_data2, repeat);

    // synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // transfer device data to host
    cudaMemcpy(h_data, d_data1, dataSize * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // display the result
    std::cout << "result after kernel1 and kernel2: " << std::endl;
    for (int i = 0; i < printSize; i++) { std::cout << h_data[i] << " "; }
    std::cout << std::endl;

    // free memory and destroy objects
    cudaFree(d_data1);
    cudaFree(d_data2);
    delete[] h_data;
    cudaEventDestroy(event1);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}