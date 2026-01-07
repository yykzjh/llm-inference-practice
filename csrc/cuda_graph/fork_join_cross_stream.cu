#include <cuda_runtime.h>
#include <iostream>

#include "../utils/cuda_profiler.h"

const int   MAX_BATCHSIZE     = 128;
const float SCALE_MAIN        = 2.0f;
const float SCALE_FORK        = 3.0f;
const int   WARMUP_ITERATIONS = 2;
const int   RUN_ITERATIONS    = 10;

// Error checking macro and function
#define cudaCheck(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void addScaleKernelInPlace(float* data, float scale, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) { data[idx] = data[idx] * scale; }
}

void fork_join_cross_stream_computation(const int N, dim3 block, dim3 grid, float* d1, float* d2,
                                        cudaStream_t mainStream, cudaStream_t forkStream, cudaEvent_t forkEvent,
                                        cudaEvent_t joinEvent)
{
    for (int i = 0; i < RUN_ITERATIONS; ++i) {
        // Execute in the main stream
        addScaleKernelInPlace<<<grid, block, 0, mainStream>>>(d1, SCALE_MAIN, N);
        cudaCheck(cudaEventRecord(forkEvent, mainStream));
        // Fork stream waits for the fork event
        cudaCheck(cudaStreamWaitEvent(forkStream, forkEvent, 0));
        // Execute in the main stream at the same time
        addScaleKernelInPlace<<<grid, block, 0, mainStream>>>(d1, SCALE_MAIN, N);
        // Execute in the fork stream
        addScaleKernelInPlace<<<grid, block, 0, forkStream>>>(d2, SCALE_FORK, N);
        cudaCheck(cudaEventRecord(joinEvent, forkStream));
        // Main stream waits for the join event
        cudaCheck(cudaStreamWaitEvent(mainStream, joinEvent, 0));
        // Execute in the main stream
        addScaleKernelInPlace<<<grid, block, 0, mainStream>>>(d1, SCALE_MAIN, N);
    }
}

int main()
{
    // Define the data size
    const int N     = MAX_BATCHSIZE * 1024;
    const int bytes = N * sizeof(float);
    // Define the block and grid dimensions
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Allocate device memory
    float *d1, *d2;
    cudaCheck(cudaMalloc(&d1, bytes));
    cudaCheck(cudaMalloc(&d2, bytes));
    // Initialize the device memory
    cudaCheck(cudaMemset(d1, 1.0f, bytes));
    cudaCheck(cudaMemset(d2, 1.0f, bytes));

    // Create the main and fork streams
    cudaStream_t mainStream, forkStream;
    cudaCheck(cudaStreamCreateWithFlags(&mainStream, cudaStreamNonBlocking));
    cudaCheck(cudaStreamCreateWithFlags(&forkStream, cudaStreamNonBlocking));
    // Create the fork and join events
    cudaEvent_t forkEvent, joinEvent;
    cudaCheck(cudaEventCreate(&forkEvent));
    cudaCheck(cudaEventCreate(&joinEvent));


    // Warm up before capture
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        fork_join_cross_stream_computation(N, block, grid, d1, d2, mainStream, forkStream, forkEvent, joinEvent);
    }
    cudaCheck(cudaStreamSynchronize(mainStream));

    // Initialize graph
    cudaGraph_t graph;
    // Capture the computation
    cudaCheck(cudaStreamBeginCapture(mainStream, cudaStreamCaptureModeGlobal));
    fork_join_cross_stream_computation(N, block, grid, d1, d2, mainStream, forkStream, forkEvent, joinEvent);
    // End graph capture
    cudaCheck(cudaStreamEndCapture(mainStream, &graph));
    cudaCheck(cudaGraphDebugDotPrint(graph, "fork_join_cross_stream_graph.dot", cudaGraphDebugDotFlagsVerbose));

    // Instantiate the graph
    cudaGraphExec_t graphExec;
    cudaCheck(cudaGraphInstantiate(&graphExec, graph));
    // Upload the graph
    cudaCheck(cudaGraphUpload(graphExec, mainStream));


    // Warm up before launch
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        cudaCheck(cudaGraphLaunch(graphExec, mainStream));
        cudaCheck(cudaStreamSynchronize(mainStream));
    }

    // Profile the graph
    CudaProfiler profiler("fork_join_cross_stream_trace_", ".");
    profiler.start();
    cudaCheck(cudaGraphLaunch(graphExec, mainStream));
    cudaCheck(cudaStreamSynchronize(mainStream));
    profiler.stop();

    // Cleanup
    cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaGraphExecDestroy(graphExec));
    cudaCheck(cudaStreamDestroy(mainStream));
    cudaCheck(cudaStreamDestroy(forkStream));
    cudaCheck(cudaEventDestroy(forkEvent));
    cudaCheck(cudaEventDestroy(joinEvent));
    cudaCheck(cudaFree(d1));
    cudaCheck(cudaFree(d2));
    return 0;
}
