#pragma once

#include <iostream>
#include <memory>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " -> " << cudaGetErrorString(err) << '\n'; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

struct CudaDeleter {
    void operator()(void* const ptr) const {
        CUDA_CHECK(cudaFree(ptr));
    }
};

template <typename T>
auto make_unique_cuda_array(std::size_t size) {
    using ValueType = std::remove_extent_t<T>;
    ValueType* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(ValueType) * size));
    return std::unique_ptr<T, CudaDeleter>(ptr);
}

template <typename T>
auto make_shared_cuda_array(std::size_t size) {
    using ValueType = std::remove_extent_t<T>;
    ValueType* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(ValueType) * size));
    return std::shared_ptr<T>(ptr, CudaDeleter{});
}
