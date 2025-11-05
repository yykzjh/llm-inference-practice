#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>


void sleep_us(uint64_t time_us, cudaStream_t stream = 0);
