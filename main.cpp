#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "common.hpp"
#include "kernel.hpp"

using real_t = float;

int main(){
    const int num_particles = 5;
    const real_t dt = 0.1;
    const int num_steps = 100;

    std::vector<real_t> px(num_particles);
    std::vector<real_t> py(num_particles);
    std::vector<real_t> vy(num_particles);

    for (int i = 0; i < num_particles; ++i) py[i] = 10.0 * (i + 1);

    std::cout << "[main] initializing cuda runtime" << std::endl;
    CUDA_CHECK(cudaFree(0));

    std::cout << "[main] allocating VRAM" << std::endl;
    real_t* px_d = nullptr;
    real_t* py_d = nullptr;
    real_t* vy_d = nullptr;
    std::size_t size = num_particles * sizeof(real_t);
    CUDA_CHECK(cudaMalloc(&px_d, size));
    CUDA_CHECK(cudaMalloc(&py_d, size));
    CUDA_CHECK(cudaMalloc(&vy_d, size));

    std::cout << "[main] copying (host -> dev)" << std::endl;
    CUDA_CHECK(cudaMemcpy(px_d, px.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(py_d, py.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vy_d, vy.data(), size, cudaMemcpyHostToDevice));

    std::cout << "[main] running sim" << std::endl;
    for (int step = 0; step < num_steps; ++step) {
        launchUpdateParticles(px_d, py_d, vy_d, num_particles, dt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(py.data(), py_d, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vy.data(), vy_d, size, cudaMemcpyDeviceToHost));
        std::cout << "[main] step #" << step << ": py = " << py[num_particles - 1] << ", vy = " << vy[num_particles - 1] << std::endl;
    }

    cudaFree(px_d);
    cudaFree(py_d);
    cudaFree(vy_d);

    return 0;
}