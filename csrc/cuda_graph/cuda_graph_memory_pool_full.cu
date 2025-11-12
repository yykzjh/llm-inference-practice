#include <cuda_runtime.h>
#include <iostream>

const int MAX_BATCHSIZE = 128;

// Error checking macro and function
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void addScaleKernel(float *in, float *out, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        out[idx] = in[idx] + 2;
    }
}

__global__ void addKernel(float *a, float *b, float *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

void logMemoryStatus(const char* message) {
    // Memory information variables
    size_t free_mem, total_mem;
    cudaCheck(cudaMemGetInfo(&free_mem, &total_mem));
    float free_gb = free_mem / (float)(1 << 20);  // Convert bytes to gigabytes
    float total_gb = total_mem / (float)(1 << 20);

    // Variables for graph memory attributes
    size_t usedMemGraphCurrent, usedMemGraphHigh, reservedMemGraphCurrent, reservedMemGraphHigh, usedMemPoolCurrent, reservedMemPoolCurrent;

    // Variables for mempool memory attributes
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, 0);
    cudaCheck(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &usedMemPoolCurrent));
    cudaCheck(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent, &reservedMemPoolCurrent));

    // Retrieve graph memory usage information
    cudaCheck(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemCurrent, &usedMemGraphCurrent));
    cudaCheck(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemHigh, &usedMemGraphHigh));
    cudaCheck(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrReservedMemCurrent, &reservedMemGraphCurrent));
    cudaCheck(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrReservedMemHigh, &reservedMemGraphHigh));

    // Print basic memory info
    std::cout << message << " - Free Memory: " << free_gb << " MB, Total Memory: " << total_gb 
    << " MB, Graph Memory Usage: " << usedMemGraphCurrent / (double)(1 << 20) << " MB, Graph Reserved Memory: " << reservedMemGraphCurrent / (double)(1 << 20) << " MB, "
    << "Mempool Memory Usage: " << usedMemPoolCurrent / (double)(1 << 20) << " MB, Mempool Reserved Memory: " << reservedMemPoolCurrent / (double)(1 << 20) << " MB\n";
}

int main() {
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, 0);
    uint64_t threshold = UINT64_MAX; // UINT64_MAX;
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);

    const int N = MAX_BATCHSIZE * 1024;
    const int bytes = N * sizeof(float);
    float *d1, * d2, *d3, *d4, *d5;

    // Initialize a and b on the host
    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 1.0f;
    }

    // Allocate host memory for the result
    float *h_out1 = new float[N];

    // Create a stream
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    logMemoryStatus("before capture");

    cudaGraph_t graphs[MAX_BATCHSIZE];
    for (int i = 0; i < MAX_BATCHSIZE; ++i) {
        // Begin graph capture
        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        cudaCheck(cudaMallocAsync(&d1, bytes, stream));
        cudaCheck(cudaMemcpyAsync(d1, h_a, bytes, cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMallocAsync(&d2, bytes, stream));
        cudaCheck(cudaMemcpyAsync(d2, h_b, bytes, cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMallocAsync(&d3, bytes, stream));
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);
        addScaleKernel<<<grid, block, 0, stream>>>(d1, d3, N);
        cudaCheck(cudaMallocAsync(&d4, bytes, stream));
        addScaleKernel<<<grid, block, 0, stream>>>(d2, d4, N);
        cudaCheck(cudaMallocAsync(&d5, bytes, stream));
        addKernel<<<grid, block, 0, stream>>>(d3, d4, d5, N);
        cudaCheck(cudaFreeAsync(d3, stream));
        cudaCheck(cudaFreeAsync(d4, stream));
        cudaCheck(cudaMemcpyAsync(h_out1, d5, bytes, cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaFreeAsync(d1, stream));
        cudaCheck(cudaFreeAsync(d2, stream));
        cudaCheck(cudaFreeAsync(d5, stream));

        // End graph capture
        cudaCheck(cudaStreamEndCapture(stream, &graphs[i]));

        if (i == 0) {
            cudaCheck(cudaGraphDebugDotPrint(graphs[i], "graph.dot", cudaGraphDebugDotFlagsVerbose));
        }
    }

    logMemoryStatus("after graph capture");

    cudaGraphExec_t graphExecs[MAX_BATCHSIZE];
    for (int i = 0; i < MAX_BATCHSIZE; ++i) {
        // Instantiate the graph
        cudaCheck(cudaGraphInstantiate(&graphExecs[i], graphs[i]));
        // Destroy the graph
        cudaCheck(cudaGraphDestroy(graphs[i]));
    }

    logMemoryStatus("after graph instantiation");


    for (int i = 0; i < MAX_BATCHSIZE; ++i) {
        cudaCheck(cudaGraphUpload(graphExecs[i], stream));
    }

    logMemoryStatus("after graph upload");

    for (int i = 0; i < MAX_BATCHSIZE; ++i) {
        cudaCheck(cudaGraphLaunch(graphExecs[i], stream));
    }
    cudaCheck(cudaDeviceSynchronize());

    logMemoryStatus("after graph launch");

    // Cleanup
    for (int i = 0; i < MAX_BATCHSIZE; ++i) {
        cudaCheck(cudaGraphExecDestroy(graphExecs[i]));
    }
    delete[] h_a;
    delete[] h_b;
    delete[] h_out1;
    cudaCheck(cudaStreamDestroy(stream));

    return 0;
}