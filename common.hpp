#include <iostream>

const auto THREADS_PER_BLOCK = 256;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " -> " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)
