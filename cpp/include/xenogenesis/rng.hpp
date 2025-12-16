#pragma once
#include <random>

namespace xenogenesis {
inline std::mt19937 make_rng(unsigned int seed) { return std::mt19937(seed); }
}
