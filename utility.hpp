#pragma once

#include <memory>
#include <cuda_runtime.h>
#include "common.hpp"

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
