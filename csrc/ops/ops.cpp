#include <pybind11/pybind11.h>
#include <torch/extension.h>  // 提供 PyTorch tensor 与 libtorch tensor 的转换

#include "sleep_us.h"
#include "scaled_multiply.h"

namespace py = pybind11;

PYBIND11_MODULE(ops, m) {
    m.doc() = "CUDA ops function bindings";

    m.def("sleep_us", [](uint64_t time_us, py::object stream_obj = py::none()) {
        cudaStream_t stream = 0;
        if (!stream_obj.is_none()) {
            // 从 PyTorch Stream 对象获取 cudaStream_t
            // cuda_stream 属性返回的是 Python int，表示 void* 指针值
            try {
                py::object cuda_stream_attr = stream_obj.attr("cuda_stream");
                // 将 Python int 转换为 void*
                uintptr_t stream_ptr = py::cast<uintptr_t>(cuda_stream_attr);
                stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid stream object: " + std::string(e.what()));
            }
        }
        return sleep_us(time_us, stream);
    }, 
          "Sleep on GPU for specified microseconds",
          py::arg("time_us"),
          py::arg("stream") = py::none());

    m.def("scaled_multiply", &scaled_multiply,
          "Scale a tensor by a factor",
          py::arg("data"),
          py::arg("scale"));
}
