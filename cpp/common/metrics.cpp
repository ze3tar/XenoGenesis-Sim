#include "xenogenesis/metrics.hpp"
#include <cmath>
#include <unordered_map>

namespace xenogenesis {

float entropy(const std::vector<float>& data) {
    std::unordered_map<int, int> bins;
    for (float v : data) {
        int bucket = static_cast<int>(v * 10.0f);
        bins[bucket]++;
    }
    float total = static_cast<float>(data.size());
    float h = 0.0f;
    for (auto& kv : bins) {
        float p = kv.second / total;
        h -= p * std::log2(p + 1e-9f);
    }
    return h;
}

}
