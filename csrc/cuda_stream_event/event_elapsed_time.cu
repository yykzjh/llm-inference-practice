#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>

__global__ void sleepKernel(int n)
{
    unsigned long long total_ns     = static_cast<unsigned long long>(n) * 1000000000ULL;
    unsigned long long max_ns       = 1000000ULL;
    unsigned long long iterations   = total_ns / max_ns;
    unsigned long long remaining_ns = total_ns % max_ns;

    for (unsigned long long i = 0; i < iterations; ++i) { __nanosleep(max_ns); }
    if (remaining_ns > 0) { __nanosleep(remaining_ns); }
}

int main(int argc, char* argv[])
{
    // 默认 sleep 时间
    float default_sleep_time   = 0.5f;
    float first_sleep_seconds  = default_sleep_time;
    float second_sleep_seconds = default_sleep_time;

    if (argc >= 2) { first_sleep_seconds = std::atof(argv[1]); }
    if (argc >= 3) { second_sleep_seconds = std::atof(argv[2]); }

    // 初始化 CUDA Events
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // 启动第一个 kernel
    sleepKernel<<<1, 1>>>(1);

    // host sleep for the first sleep time
    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<int>(first_sleep_seconds * 1000.0f)));

    // record the start event
    cudaEventRecord(startEvent, 0);

    // 启动第二个 kernel
    sleepKernel<<<1, 1>>>(1);

    // host sleep for the second sleep time
    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<int>(second_sleep_seconds * 1000.0f)));

    // record the stop event
    cudaEventRecord(stopEvent, 0);

    // 同步
    cudaEventSynchronize(stopEvent);

    // 计算事件间隔
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    std::cout << "Elapsed time: " << elapsedTime << " milliseconds" << std::endl;

    // 销毁事件
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}