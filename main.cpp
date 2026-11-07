#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
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

    auto init_particles = [&]() {
        for (int i = 0; i < p_height; ++i) {
            for (int j = 0; j < p_width; ++j) {
                auto idx = i * p_width + j;
                px[idx] = 0.5 + static_cast<real_t>(j) * p_radius;
                py[idx] = 0.5 + static_cast<real_t>(i) * p_radius;
                vx[idx] = vy[idx] = 0;
            }
    }

    CUDA_CHECK(cudaMemcpy(px_d.get(), px.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(py_d.get(), py.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vx_d.get(), vx.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vy_d.get(), vy.data(), size, cudaMemcpyHostToDevice));
    };

    init_particles();

    const std::size_t w_width = 800;
    const std::size_t w_height = 600;
    const std::size_t w_fps = 0;
    sf::RenderWindow window(sf::VideoMode(w_width, w_height), "SFML: fluid_sim");
    window.setFramerateLimit(w_fps);
    sf::VertexArray vertices(sf::Points, num_particles);
    for (int i = 0; i < num_particles; ++i) vertices[i].color = sf::Color::Green;

    const real_t scale_x = 20.0;
    const real_t scale_y = 20.0;

    sf::Clock fps_clock;
    int cnt_frames = 0;
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            switch (event.type) {
                case sf::Event::Closed:
                    window.close();
                    break;
                case sf::Event::KeyPressed:
                    switch (event.key.code) {
                        case sf::Keyboard::Escape:
                            window.close();
                            break;
                        case sf::Keyboard::R:
                            init_particles();
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(px.data(), px_d.get(), size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(py.data(), py_d.get(), size, cudaMemcpyDeviceToHost));

        for (int i = 0; i < num_particles; ++i) {
            vertices[i].position = sf::Vector2f(px[i] * scale_x, w_height - py[i] * scale_y);
        }

        ++cnt_frames;
        if (fps_clock.getElapsedTime().asSeconds() >= 0.5) {
            real_t fps = cnt_frames / fps_clock.restart().asSeconds();
            std::string title = "SFML: fluid_sim";
            title += " | #p: " + std::to_string(num_particles);
            title += " | fps: " + std::to_string(static_cast<int>(fps));
            window.setTitle(title);
            cnt_frames = 0;
        }

        window.clear(sf::Color::Black);
        window.draw(vertices);

        const auto siz_group = 1003;
        const auto num_markers = num_particles / siz_group;
        const real_t mark_radius = 5.0;
        sf::CircleShape circle_mark(mark_radius);
        for (int i = 0; i < num_markers; ++i) {
            circle_mark.setFillColor(sf::Color::Red);
            circle_mark.setOrigin(mark_radius, mark_radius);
            circle_mark.setPosition(px[siz_group * i] * scale_x, w_height - py[siz_group * i] * scale_y);
            window.draw(circle_mark);
        }

        window.display();
    }

    return 0;
}
