#include <pybind11/pybind11.h>
#include <torch/extension.h>  // 提供 PyTorch tensor 与 libtorch tensor 的转换

#include "sleep_us.h"
#include "scaled_multiply.h"

namespace py = pybind11;

PYBIND11_MODULE(ops, m) {
    m.doc() = "CUDA ops function bindings";

    m.def("sleep_us", &sleep_us, 
          "Sleep on GPU for specified microseconds",
          py::arg("time_us"));

    m.def("scaled_multiply", &scaled_multiply,
          "Scale a tensor by a factor",
          py::arg("data"),
          py::arg("scale"));
}
