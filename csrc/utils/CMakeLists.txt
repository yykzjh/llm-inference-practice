# 查找 libtorch
# 方式1: 通过环境变量 TORCH_PATH 指定 libtorch 路径
if(DEFINED ENV{TORCH_PATH})
    set(Torch_DIR "$ENV{TORCH_PATH}/share/cmake/Torch")
    message(STATUS "Using TORCH_PATH: $ENV{TORCH_PATH}")
endif()

# 方式2: 通过 CMAKE_PREFIX_PATH 指定（或通过命令行 -DCMAKE_PREFIX_PATH=/path/to/libtorch）
# 方式3: 使用 find_package 在系统路径中查找
find_package(Torch REQUIRED)

message(STATUS "Found Torch: ${TORCH_INSTALL_PREFIX}")
message(STATUS "Torch Version: ${TORCH_VERSION}")
message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")

# 创建静态库目标（作为基础组件供其他模块复用）
add_library(utils STATIC
    cuda_profiler.cpp
)
set_target_properties(utils PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 链接 CUDA 运行时库和 libtorch
target_link_libraries(utils PRIVATE cudart)
target_link_libraries(utils PUBLIC ${TORCH_LIBRARIES})

# 设置包含目录
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 设置 libtorch 的编译选项
if(Torch_FOUND)
    # 设置 C++17 标准（libtorch 需要）
    set_property(TARGET utils PROPERTY CXX_STANDARD 17)
    
    # 添加 libtorch 的包含目录
    target_include_directories(utils PUBLIC ${TORCH_INCLUDE_DIRS})
    
    # 设置编译定义（如果需要）
    if(TORCH_CUDA_ARCH_LIST)
        set_property(TARGET utils PROPERTY CUDA_ARCHITECTURES ${TORCH_CUDA_ARCH_LIST})
    endif()
endif()
