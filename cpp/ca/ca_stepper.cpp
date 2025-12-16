#include "xenogenesis/ca_stepper.hpp"
#include <vector>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace xenogenesis {

static float gaussian(float x, float mu, float sigma) {
    const float diff = x - mu;
    return std::exp(-(diff * diff) / (2.0f * sigma * sigma));
}

std::vector<float> ca_step(const std::vector<float>& state, std::size_t width, std::size_t height, const CAParams& params) {
    std::vector<float> out(state.size(), 0.0f);
    const float inner_r2 = params.inner_radius * params.inner_radius;
    const float outer_r2 = params.outer_radius * params.outer_radius;

    #pragma omp parallel for collapse(2) if(width * height > 1024)
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            float m_acc = 0.0f;
            float n_acc = 0.0f;
            int m_count = 0;
            int n_count = 0;
            for (int dy = -(int)params.outer_radius; dy <= (int)params.outer_radius; ++dy) {
                for (int dx = -(int)params.outer_radius; dx <= (int)params.outer_radius; ++dx) {
                    int nx = (int)x + dx;
                    int ny = (int)y + dy;
                    if (nx < 0 || ny < 0 || nx >= (int)width || ny >= (int)height) continue;
                    float dist2 = static_cast<float>(dx * dx + dy * dy);
                    float val = state[ny * width + nx];
                    if (dist2 <= inner_r2) {
                        m_acc += val;
                        ++m_count;
                    } else if (dist2 <= outer_r2 && dist2 > inner_r2 * params.ring_ratio) {
                        n_acc += val;
                        ++n_count;
                    }
                }
            }
            float m = m_count > 0 ? m_acc / m_count : 0.0f;
            float n = n_count > 0 ? n_acc / n_count : 0.0f;
            float g = gaussian(n, params.mu, params.sigma);
            float updated = state[y * width + x] + params.dt * (2.0f * g - 1.0f);
            if (updated < 0.0f) updated = 0.0f;
            if (updated > 1.0f) updated = 1.0f;
            out[y * width + x] = updated;
        }
    }
    return out;
}

}
