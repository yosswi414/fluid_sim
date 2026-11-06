#include "common.hpp"

__global__ void updateParticlesKernel(float* px, float* py, float* vy, int num_particles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_particles) return;

    const float g = 9.8;

    vy[idx] += -g * dt;
    py[idx] += vy[idx] * dt;

    // floor collision
    const float e = 0.5;
    if (py[idx] < 0.0) {
        py[idx] = 0.0;
        vy[idx] *= -e;
    }
}

void launchUpdateParticles(float* px_d, float* py_d, float* vy_d, int num_particles, float dt) {
    const int threads_per_block = THREADS_PER_BLOCK;
    // ceil(num_particles / threads_per_block)
    int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;

    updateParticlesKernel<<<blocks_per_grid, threads_per_block>>>(px_d, py_d, vy_d, num_particles, dt);
}
