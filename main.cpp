#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "common.hpp"
#include "kernel.hpp"
#include "utility.hpp"

int main(){
    const int num_particles = 10000;
    const real_t dt = 1e-2;
    const int num_steps = 100;
    const std::size_t size = num_particles * sizeof(real_t);
    const std::size_t len_side = 100;

    std::vector<real_t> px(num_particles), py(px), vx(px), vy(px);

    auto px_d = make_unique_cuda_array<real_t>(num_particles);
    auto py_d = make_unique_cuda_array<real_t>(num_particles);
    auto vx_d = make_unique_cuda_array<real_t>(num_particles);
    auto vy_d = make_unique_cuda_array<real_t>(num_particles);

    for (int i = 0; i < num_particles; ++i) {
        px[i] = 2.0 + (i % len_side) * 0.05;    // x: 2.0 - 7.0
        py[i] = 5.0 + (i / len_side) * 0.05;    // y: 5.0 - 10.0
    }

    CUDA_CHECK(cudaMemcpy(px_d.get(), px.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(py_d.get(), py.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vx_d.get(), vx.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vy_d.get(), vy.data(), size, cudaMemcpyHostToDevice));

    for (int step = 0; step < num_steps; ++step) {
        launchUpdateParticles(px_d.get(), py_d.get(), vx_d.get(), vy_d.get(), num_particles, dt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&px[0], &px_d.get()[0], sizeof(real_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&py[0], &py_d.get()[0], sizeof(real_t), cudaMemcpyDeviceToHost));
        std::cout << "[main] step #" << step << ": p_0 = (" << px[0] << ", " << py[0] << ")\n";
    }

    return 0;
}
