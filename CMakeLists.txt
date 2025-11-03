# CMakeLists.txt (at project root)
cmake_minimum_required(VERSION 3.30)
project(llm-inference LANGUAGES CXX CUDA)

# 设置标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # 可选：禁用 GNU 扩展，更标准
set(CMAKE_CUDA_ARCHITECTURES "90")

# 添加子目录：CUDA Stream 示例
add_subdirectory(csrc/cuda_stream)
