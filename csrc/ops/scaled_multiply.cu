#include <cuda_runtime.h>
#include <torch/torch.h>

#include "scaled_multiply.h"

__global__ void scaled_multiply_kernel(float* data, int size, float scale)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = data[idx] * scale;
    }
}

void scaled_multiply(torch::Tensor& data, float scale)
{
    // 确保 tensor 在 CUDA 上
    TORCH_CHECK(data.is_cuda(), "Tensor must be on CUDA device");
    TORCH_CHECK(data.dtype() == torch::kFloat32, "Tensor must be float32");
    
    int64_t num_elements = data.numel();
    if (num_elements == 0) {
        return;
    }
    
    // 计算合适的 block 和 thread 数量
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    // 启动 kernel
    scaled_multiply_kernel<<<num_blocks, threads_per_block>>>(
        data.data_ptr<float>(), 
        static_cast<int>(num_elements), 
        scale
    );
    
    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}