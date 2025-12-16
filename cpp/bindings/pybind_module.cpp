#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "xenogenesis/ca_stepper.hpp"
#include "xenogenesis/metrics.hpp"

namespace py = pybind11;
using namespace xenogenesis;

py::array_t<float> py_ca_step(py::array_t<float, py::array::c_style | py::array::forcecast> state,
                              float mu, float sigma, float dt,
                              float inner_radius, float outer_radius, float ring_ratio) {
    auto buf = state.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("state must be 2D");
    }
    std::size_t height = static_cast<std::size_t>(buf.shape[0]);
    std::size_t width = static_cast<std::size_t>(buf.shape[1]);
    const float* ptr = static_cast<float*>(buf.ptr);
    std::vector<float> input(ptr, ptr + width * height);
    CAParams p{mu, sigma, dt, inner_radius, outer_radius, ring_ratio};
    auto out_vec = ca_step(input, width, height, p);
    return py::array_t<float>({static_cast<long>(height), static_cast<long>(width)}, out_vec.data());
}

PYBIND11_MODULE(xenogenesis_native, m) {
    m.def("ca_step", &py_ca_step, "Step CA grid once");
    m.def("entropy", &entropy, "Compute simple entropy");
}
