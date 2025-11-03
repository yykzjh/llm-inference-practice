// demo_default_stream.cu
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

// 模拟耗时 kernel：每秒打印一次，持续 10 秒
__global__ void timed_kernel(int kernel_id, long long cycles_per_second) {
    printf("Kernel %d started.\n", kernel_id);
    uint64_t start_cycle = clock64();
    printf("Clock Rate: %lld Hz\n", cycles_per_second);

    for (int i = 0; i < 10; ++i) {
        uint64_t elapsed_cycles = clock64() - start_cycle;
        // 注意：clockRate 单位是 kHz，需换算
        float elapsed_seconds = elapsed_cycles / cycles_per_second;

        printf("GPU Cycle=%llu | Elapsed=%.2fs | Kernel %d Step %d\n", elapsed_cycles, elapsed_seconds, kernel_id, i + 1);

        // ✅ 正确延时 1 秒
        uint64_t start = clock64();
        while ((clock64() - start) < cycles_per_second) ;
    }
}

void launch_in_thread(int thread_id) {
    cudaSetDevice(0);  // 确保每个线程绑定到同一设备
    // 在 kernel 外获取设备属性（可通过参数传入）
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    long long cycles_per_second = prop.clockRate * 1000;

    // blocking stream 与 default stream
    printf("Blocking stream 与 default stream\n");
    cudaStream_t blocking_stream;
    cudaStreamCreateWithFlags(&blocking_stream, cudaStreamDefault);
    timed_kernel<<<1, 1, 0, blocking_stream>>>(0, cycles_per_second);
    timed_kernel<<<1, 1, 0, 0>>>(1, cycles_per_second);
    timed_kernel<<<1, 1, 0, blocking_stream>>>(2, cycles_per_second);
    cudaDeviceSynchronize();
    printf("Blocking stream 与 default stream 完成\n");

    // non-blocking stream 与 default stream
    printf("Non-blocking stream 与 default stream\n");
    cudaStream_t non_blocking_stream;
    cudaStreamCreateWithFlags(&non_blocking_stream, cudaStreamNonBlocking);
    timed_kernel<<<1, 1, 0, non_blocking_stream>>>(0, cycles_per_second);
    timed_kernel<<<1, 1, 0, 0>>>(1, cycles_per_second);
    timed_kernel<<<1, 1, 0, non_blocking_stream>>>(2, cycles_per_second);
    cudaDeviceSynchronize();
    printf("Non-blocking stream 与 default stream 完成\n");

    // default stream 与 blocking stream
    printf("Default stream 与 blocking stream\n");
    timed_kernel<<<1, 1, 0, 0>>>(0, cycles_per_second);
    timed_kernel<<<1, 1, 0, blocking_stream>>>(1, cycles_per_second);
    timed_kernel<<<1, 1, 0, 0>>>(2, cycles_per_second);
    cudaDeviceSynchronize();
    printf("Default stream 与 blocking stream 完成\n");

    // default stream 与 non-blocking stream
    printf("Default stream 与 non-blocking stream\n");
    timed_kernel<<<1, 1, 0, 0>>>(0, cycles_per_second);
    timed_kernel<<<1, 1, 0, non_blocking_stream>>>(1, cycles_per_second);
    timed_kernel<<<1, 1, 0, 0>>>(2, cycles_per_second);
    cudaDeviceSynchronize();
    printf("Default stream 与 non-blocking stream 完成\n");

    // non-blocking stream 与 blocking stream
    printf("Non-blocking stream 与 blocking stream\n");
    timed_kernel<<<1, 1, 0, non_blocking_stream>>>(0, cycles_per_second);
    timed_kernel<<<1, 1, 0, blocking_stream>>>(1, cycles_per_second);
    timed_kernel<<<1, 1, 0, non_blocking_stream>>>(2, cycles_per_second);
    cudaDeviceSynchronize();
    printf("Non-blocking stream 与 blocking stream 完成\n");

    cout << "Thread " << thread_id << " finished.\n" << endl;
}

int main() {
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    printf("Per-thread default stream\n");
#else
    printf("Legacy default stream\n");
#endif

    // 设置更大的 printf 缓冲区
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 50ULL * 1024 * 1024);  // 10MB
    
    vector<thread> threads;

    cout << "Starting two threads...\n";
    cout << "Using default stream handle: 0\n";
    cout << "Behavior depends on compilation flag.\n\n";

    // 创建两个线程同时运行
    threads.emplace_back(launch_in_thread, 0);
    // threads.emplace_back(launch_in_thread, 1);

    // 等待完成
    for (auto& t : threads) {
        t.join();
    }

    cudaDeviceReset();

    return 0;
}
