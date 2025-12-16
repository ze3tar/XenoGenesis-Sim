#pragma once
#include <vector>
#include <cstddef>

namespace xenogenesis {
struct CAParams {
    float mu;
    float sigma;
    float dt;
    float inner_radius;
    float outer_radius;
    float ring_ratio;
};

std::vector<float> ca_step(const std::vector<float>& state, std::size_t width, std::size_t height, const CAParams& params);
}
