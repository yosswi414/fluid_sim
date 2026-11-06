#include <cmath>

#include "common.hpp"

__global__ void updateParticlesKernel(real_t* px, real_t* py, real_t* vx, real_t* vy, int num_particles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_particles) return;

    const real_t g = -9.8;
    const real_t r = 0.06;
    const real_t k = 500.0;
    const real_t e = 0.5;

    real_t ax = 0;
    real_t ay = 0;

    for (int i = 0; i < num_particles; ++i) {
        if (i == idx) continue;

        real_t dx = px[idx] - px[i];
        real_t dy = py[idx] - py[i];
        real_t dist = std::sqrt(dx * dx + dy * dy);
        if (dist < EPS) dist = EPS;

        real_t overlap = r - dist;
        if (overlap > 0.0) {
            real_t f = k * overlap / dist;
            ax += dx * f;
            ay += dy * f;
        }
    }

    vx[idx] += ax * dt;
    vy[idx] += (ay + g) * dt;

    px[idx] += vx[idx];
    py[idx] += vy[idx];

    // collision
    const real_t bound_x_min = 0.0;
    const real_t bound_x_max = 10.0;
    const real_t bound_y_min = 0.0;
    const real_t bound_y_max = 20.0;

    if (px[idx] < bound_x_min) {
        px[idx] = bound_x_min;
        vx[idx] *= -e;
    }
    if (bound_x_max < px[idx]) {
        px[idx] = bound_x_max;
        vx[idx] *= -e;
    }
    if (py[idx] < bound_y_min) {
        py[idx] = bound_y_min;
        vy[idx] *= -e;
    }
    if (bound_y_max < py[idx]) {
        py[idx] = bound_y_max;
        vy[idx] *= -e;
    }
}

void launchUpdateParticles(real_t* px_d, real_t* py_d, real_t* vx_d, real_t* vy_d, int num_particles, float dt) {
    const int threads_per_block = THREADS_PER_BLOCK;
    // ceil(num_particles / threads_per_block)
    int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;

    updateParticlesKernel<<<blocks_per_grid, threads_per_block>>>(px_d, py_d, vx_d, vy_d, num_particles, dt);
}
