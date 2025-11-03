#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int         deviceCount = 0;
    cudaError_t error       = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << static_cast<int>(error) << ":"
                  << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    for (int device_id = 0; device_id < deviceCount; device_id++) {
        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, device_id);

        if (error != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed: " << static_cast<int>(error) << ":"
                      << cudaGetErrorString(error) << std::endl;
            return 1;
        }

        std::cout << "Device " << device_id << ": " << deviceProp.name << std::endl;
        std::cout << "  async engine count: " << deviceProp.asyncEngineCount << std::endl;
    }

    return 0;
}
