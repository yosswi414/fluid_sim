#include <cmath>

#include "common.hpp"

__global__ void updateParticlesKernel(real_t* px, real_t* py, real_t* vx, real_t* vy, int num_particles, real_t radius,
                                      real_t dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_particles) return;

    const real_t g = -9.8;
    const real_t r = radius;
    const real_t k = 500;
    const real_t k_w = 5000;
    const real_t e = 0.5;
    const real_t mu_w = 0.3;
    const real_t d_w = -2.0 * std::log(e < EPS ? EPS : e) * std::sqrt(k_w);
    const real_t tau = 10;

    real_t ax_p = 0;
    real_t ay_p = 0;
    real_t ax_v = 0;
    real_t ay_v = 0;
    real_t overlap = 0;
    real_t f_press = 0;
    real_t f_visc = 0;
    real_t dist = 0;
    real_t dx = 0;
    real_t dy = 0;
    real_t du = 0;
    real_t dv = 0;
    real_t weight_sum = 0;
    for (int i = 0; i < num_particles; ++i) {
        if (i == idx) continue;

        dx = px[idx] - px[i];
        dy = py[idx] - py[i];
        du = vx[idx] - vx[i];
        dv = vy[idx] - vy[i];
        dist = std::sqrt(dx * dx + dy * dy);
        if (dist < EPS) dist = EPS;

        overlap = r - dist;
        if (overlap > 0.0) {
            f_press = k * overlap / dist;
            real_t weight = overlap / r;
            f_visc = tau * weight;
            ax_p += dx * f_press;
            ax_v -= du * f_visc;
            ay_p += dy * f_press;
            ay_v -= dv * f_visc;
            weight_sum += weight;
        }
    }
    if (weight_sum < EPS) weight_sum = EPS;
    ax_v /= weight_sum;
    ay_v /= weight_sum;

    // collision
    const real_t bound_x_min = 0.0;
    const real_t bound_x_max = 40.0;
    const real_t bound_y_min = 0.0;

    if (px[idx] < bound_x_min) {
        real_t penet = bound_x_min - px[idx];
        real_t fn = k_w * penet - d_w * vx[idx];
        if (fn > 0.0) {
            ax_p += fn;
            ay_p -= mu_w * vy[idx];
        }
    }

    if (bound_x_max < px[idx]) {
        real_t penet = px[idx] - bound_x_max;
        real_t fn = k_w * penet - d_w * (-vx[idx]);
        if (fn > 0.0) {
            ax_p -= fn;
            ay_p -= mu_w * vy[idx];
        }
    }

    if (py[idx] < bound_y_min) {
        real_t penet = bound_y_min - py[idx];
        real_t fn = k_w * penet - d_w * vy[idx];
        if (fn > 0.0) {
            ay_p += fn;
            ax_p -= mu_w * vx[idx];
        }
    }

    vx[idx] += (ax_p + ax_v) * dt;
    vy[idx] += (ay_p + ay_v + g) * dt;

    px[idx] += vx[idx] * dt;
    py[idx] += vy[idx] * dt;
}

void launchUpdateParticles(real_t* px_d, real_t* py_d, real_t* vx_d, real_t* vy_d, int num_particles, real_t radius,
                           real_t dt) {
    const int threads_per_block = THREADS_PER_BLOCK;
    // ceil(num_particles / threads_per_block)
    int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;

    updateParticlesKernel<<<blocks_per_grid, threads_per_block>>>(px_d, py_d, vx_d, vy_d, num_particles, radius, dt);
}
