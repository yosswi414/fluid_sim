#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "common.hpp"
#include "kernel.hpp"
#include "utility.hpp"

using real_t = float;

int main(){
    const int num_particles = 5;
    const real_t dt = 0.1;
    const int num_steps = 100;

    std::vector<real_t> px(num_particles);
    std::vector<real_t> py(num_particles);
    std::vector<real_t> vy(num_particles);

    for (int i = 0; i < num_particles; ++i) py[i] = 10.0 * (i + 1);

    auto px_d = make_unique_cuda_array<real_t>(num_particles);
    auto py_d = make_unique_cuda_array<real_t>(num_particles);
    auto vy_d = make_unique_cuda_array<real_t>(num_particles);
    std::size_t size = num_particles * sizeof(real_t);

    CUDA_CHECK(cudaMemcpy(px_d.get(), px.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(py_d.get(), py.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vy_d.get(), vy.data(), size, cudaMemcpyHostToDevice));

    for (int step = 0; step < num_steps; ++step) {
        launchUpdateParticles(px_d.get(), py_d.get(), vy_d.get(), num_particles, dt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(py.data(), py_d.get(), size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vy.data(), vy_d.get(), size, cudaMemcpyDeviceToHost));
        std::cout << "[main] step #" << step << ": py = " << py[num_particles - 1] << ", vy = " << vy[num_particles - 1] << '\n';
    }

    return 0;
}
