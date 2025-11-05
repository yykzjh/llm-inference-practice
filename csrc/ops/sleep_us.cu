#include <cuda_runtime_api.h>

#include "sleep_us.h"



__global__ void sleep_us_kernel(uint64_t sleep_cycles)
{
    uint64_t start_cycle = clock64();
    while ((clock64() - start_cycle) < sleep_cycles)
        ;
}



void sleep_us(uint64_t time_us, cudaStream_t stream)
{
    // 使用静态变量缓存设备属性，避免每次调用都查询
    static uint64_t cached_sleep_cycles = 0;
    if (cached_sleep_cycles == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        uint64_t cycles_per_second = prop.clockRate * 1000;
        cached_sleep_cycles = time_us * cycles_per_second / 1000000ULL;
    }
    sleep_us_kernel<<<1, 1, 0, stream>>>(cached_sleep_cycles);
}
