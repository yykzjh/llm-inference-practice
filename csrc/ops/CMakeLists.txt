# 使用 FetchContent 获取 pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

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

# 检查是否有 TORCH_PYTHON_LIBRARIES（用于 Python 绑定）
if(DEFINED TORCH_PYTHON_LIBRARIES)
    message(STATUS "Torch Python Libraries: ${TORCH_PYTHON_LIBRARIES}")
endif()

# 创建 pybind11 模块（直接包含 CUDA 代码）
pybind11_add_module(ops ops.cpp sleep_us.cu scaled_multiply.cu)

# 链接 CUDA 运行时库和 libtorch
# 注意：需要链接 PyTorch 的 Python 绑定库以支持 tensor 类型转换
target_link_libraries(ops PRIVATE 
    cudart 
    ${TORCH_LIBRARIES}
)

# 尝试链接 PyTorch Python 绑定库（如果存在）
# 方法1: 检查是否有 torch::Python 目标
if(TARGET torch::Python)
    target_link_libraries(ops PRIVATE torch::Python)
    message(STATUS "Linked torch::Python target")
# 方法2: 检查 TORCH_PYTHON_LIBRARIES 变量
elseif(DEFINED TORCH_PYTHON_LIBRARIES)
    target_link_libraries(ops PRIVATE ${TORCH_PYTHON_LIBRARIES})
    message(STATUS "Linked TORCH_PYTHON_LIBRARIES")
# 方法3: 在 libtorch 安装目录中查找
else()
    find_library(TORCH_PYTHON_LIB 
        NAMES torch_python
        PATHS ${TORCH_INSTALL_PREFIX}/lib
        NO_DEFAULT_PATH
    )
    if(TORCH_PYTHON_LIB)
        target_link_libraries(ops PRIVATE ${TORCH_PYTHON_LIB})
        message(STATUS "Found and linked torch_python: ${TORCH_PYTHON_LIB}")
    else()
        # 方法4: 尝试从 Python 环境查找 PyTorch 库
        find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__file__)"
            OUTPUT_VARIABLE TORCH_PYTHON_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(TORCH_PYTHON_PATH)
            get_filename_component(TORCH_PYTHON_DIR ${TORCH_PYTHON_PATH} DIRECTORY)
            get_filename_component(TORCH_PYTHON_DIR ${TORCH_PYTHON_DIR} DIRECTORY)
            set(TORCH_PYTHON_LIB_DIR "${TORCH_PYTHON_DIR}/lib")
            
            find_library(TORCH_PYTHON_LIB_FROM_PYTHON
                NAMES torch_python
                PATHS ${TORCH_PYTHON_LIB_DIR}
                NO_DEFAULT_PATH
            )
            if(TORCH_PYTHON_LIB_FROM_PYTHON)
                target_link_libraries(ops PRIVATE ${TORCH_PYTHON_LIB_FROM_PYTHON})
                message(STATUS "Found torch_python from Python environment: ${TORCH_PYTHON_LIB_FROM_PYTHON}")
            else()
                message(WARNING "Could not find PyTorch Python binding library. "
                    "Tensor conversion may fail. Tried: ${TORCH_PYTHON_LIB_DIR}")
            endif()
        else()
            message(WARNING "Could not find PyTorch Python binding library. "
                "Tensor conversion may fail. Please ensure PyTorch is installed in Python environment.")
        endif()
    endif()
endif()

# 设置包含目录
target_include_directories(ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 设置 libtorch 的编译选项
if(Torch_FOUND)
    # 设置 C++17 标准（libtorch 需要）
    set_property(TARGET ops PROPERTY CXX_STANDARD 17)
    
    # 添加 libtorch 的包含目录
    target_include_directories(ops PRIVATE ${TORCH_INCLUDE_DIRS})
    
    # 设置编译定义（如果需要）
    if(TORCH_CUDA_ARCH_LIST)
        set_property(TARGET ops PROPERTY CUDA_ARCHITECTURES ${TORCH_CUDA_ARCH_LIST})
    endif()
endif()
