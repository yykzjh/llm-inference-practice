// demo_default_stream.cu
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

// 模拟耗时 kernel：每秒打印一次，持续 10 秒
__global__ void timed_kernel(int thread_id, long long cycles_per_second) {
    printf("Thread %d started.\n", thread_id);
    uint64_t start_cycle = clock64();
    printf("Clock Rate: %lld Hz\n", cycles_per_second);

    for (int i = 0; i < 10; ++i) {
        uint64_t elapsed_cycles = clock64() - start_cycle;
        // 注意：clockRate 单位是 kHz，需换算
        float elapsed_seconds = elapsed_cycles / cycles_per_second;

        printf("GPU Cycle=%llu | Elapsed=%.2fs | Thread %d Step %d\n", elapsed_cycles, elapsed_seconds, thread_id, i + 1);

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

    // 启动 kernel，使用 "default stream"
    // 根据编译条件，stream 0 实际映射为 legacy 或 per-thread default stream
    timed_kernel<<<1, 1>>>(thread_id, cycles_per_second);

    // 同步：等待该 stream 完成
    // 如果是 legacy default stream，则会阻塞其他线程的 kernel
    // 如果是 per-thread default stream，则互不影响
    cudaStreamSynchronize(0);  // 传入 0，实际转为 1 或 2

    cout << "Thread " << thread_id << " finished.\n" << endl;
}

int main() {
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    printf("Per-thread default stream\n");
#else
    printf("Legacy default stream\n");
#endif

    // 设置更大的 printf 缓冲区
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10ULL * 1024 * 1024);  // 10MB
    
    vector<thread> threads;

    cout << "Starting two threads...\n";
    cout << "Using default stream handle: 0\n";
    cout << "Behavior depends on compilation flag.\n\n";

    // 创建两个线程同时运行
    threads.emplace_back(launch_in_thread, 0);
    threads.emplace_back(launch_in_thread, 1);

    // 等待完成
    for (auto& t : threads) {
        t.join();
    }

    cudaDeviceReset();

    return 0;
}
