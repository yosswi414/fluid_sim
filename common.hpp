#pragma once

#include <iostream> // IWYU pragma: keep
#include <cuda_runtime.h>

const auto THREADS_PER_BLOCK = 256;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " -> " << cudaGetErrorString(err) << '\n'; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)
