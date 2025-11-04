#include <cuda_runtime_api.h>

#include "sleep_us.h"



__global__ void sleep_us_kernel(uint64_t sleep_cycles)
{
    uint64_t start_cycle = clock64();
    while ((clock64() - start_cycle) < sleep_cycles)
        ;
}



void sleep_us(uint64_t time_us)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint64_t cycles_per_second = prop.clockRate * 1000;
    uint64_t sleep_cycles      = time_us * cycles_per_second / 1000000ULL;
    sleep_us_kernel<<<1, 1>>>(sleep_cycles);
}
