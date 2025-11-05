import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
build_ops_path = project_root / "build" / "csrc" / "ops"
sys.path.insert(0, str(build_ops_path))

# 导入 ops 模块
try:
    import ops  # type: ignore
except ImportError as e:
    raise ImportError(
        f"无法导入 ops 模块。请确保已编译项目：\n"
        f"  cd {project_root} && cmake -B build && cmake --build build\n"
        f"原始错误: {e}"
    )

import torch
import time
import ctypes

# Load the CUDA runtime library
cudart = ctypes.CDLL('libcudart.so')

# Define cudaMemcpyKind enumeration as in the CUDA API
cudaMemcpyHostToHost = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3
cudaMemcpyDefault = 4

# Setup the prototype of the cudaMemcpyAsync function
cudaMemcpyAsync = cudart.cudaMemcpyAsync
cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p,          # void* dst
    ctypes.c_void_p,          # const void* src
    ctypes.c_size_t,          # size_t count
    ctypes.c_int,             # enum cudaMemcpyKind
    ctypes.c_void_p           # cudaStream_t stream
]
cudaMemcpyAsync.restype = ctypes.c_int


def get_async_engine_count_cuda():
    """使用CUDA原生API获取async engine数量"""
    try:
        cudart = ctypes.CDLL('libcudart.so')
        
        # 方法1: 使用cudaDeviceGetAttribute
        cudaDevAttrAsyncEngineCount = 89
        
        cudaDeviceGetAttribute = cudart.cudaDeviceGetAttribute
        cudaDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        cudaDeviceGetAttribute.restype = ctypes.c_int
        
        async_engine_count = ctypes.c_int()
        device = torch.cuda.current_device()
        
        result = cudaDeviceGetAttribute(
            ctypes.byref(async_engine_count), 
            cudaDevAttrAsyncEngineCount, 
            device
        )
        
        if result == 0:
            return async_engine_count.value
        else:
            print(f"cudaDeviceGetAttribute失败，错误码: {result}")
            return None
    except Exception as e:
        print(f"获取async engine count时出错: {e}")
        return None

def multi_stream_memory_ops_profiling():
    """在4个不同的stream上分别执行不同tensor的内存分配和拷贝操作，并使用profiler记录trace"""
    # 获取async engine数量
    async_engine_count = get_async_engine_count_cuda()
    print(f"async engine数量: {async_engine_count}")
    
    # 初始化4个不同的stream
    num_streams = 4
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    print(f"已创建 {num_streams} 个 CUDA stream")
    print("-" * 100)
    
    # 定义不同大小的tensor配置（每个stream使用不同的tensor大小）
    # 格式: (shape, dtype)
    tensor_configs = [((4096, 4096), torch.float32)] * num_streams
    
    print("各Stream的Tensor配置:")
    for i, (shape, dtype) in enumerate(tensor_configs):
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        print(f"  Stream {i}: shape={shape}, dtype={dtype}, 元素数={num_elements:,}, 大小={num_elements * dtype.itemsize / 1024 / 1024:.2f} MB")
    print("-" * 100)
    
    # 使用torch profiler进行性能分析
    trace_file = "async_engine_trace_memory_ops.json"
    
    # 对整个执行过程进行计时
    start_time = time.time()
    
    # 存储创建的tensor，避免被垃圾回收
    # 关键：使用pinned memory才能实现真正的异步拷贝
    host_tensors = []
    device_tensors = []
    
    for shape, dtype in tensor_configs:
        # 创建pinned memory的host tensor（这是关键！）
        host_tensor = torch.randn(shape, dtype=dtype, device='cpu').pin_memory()
        host_tensors.append(host_tensor)
        
        # 在GPU上预分配device tensor
        device_tensor = torch.empty(shape, dtype=dtype, device='cuda')
        device_tensors.append(device_tensor)
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function("multi_stream_memory_ops"):
            # 关键：先发起所有操作，不要立即等待
            # 这样多个async engine才能并行工作
            for i, (stream, (shape, dtype)) in enumerate(zip(streams, tensor_configs)):
                with torch.cuda.stream(stream):
                    with torch.profiler.record_function(f"stream_{i}_memCpy_H2D"):
                        # 使用CUDA原生的cudaMemcpyAsync进行异步拷贝
                        host_ptr = host_tensors[i].data_ptr()
                        device_ptr = device_tensors[i].data_ptr()
                        num_bytes = host_tensors[i].numel() * host_tensors[i].element_size()
                        
                        # 调用cudaMemcpyAsync
                        result = cudaMemcpyAsync(
                            device_ptr,                    # dst
                            host_ptr,                      # src
                            num_bytes,                    # count
                            cudaMemcpyHostToDevice,       # kind
                            stream.cuda_stream            # stream
                        )
                        
                        if result != 0:
                            raise RuntimeError(f"cudaMemcpyAsync failed with error code: {result}")
            
            # # 同步所有stream（等待所有操作完成）
            # with torch.profiler.record_function("synchronize_all_streams"):
            #     for stream in streams:
            #         stream.synchronize()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 输出结果
    print(f"执行完成！")
    print(f"总耗时: {elapsed_time * 1000:.2f} ms")
    print(f"创建了 {len(device_tensors)} 个GPU tensor")
    print("-" * 100)
    
    # 导出trace为JSON文件
    prof.export_chrome_trace(trace_file)
    print(f"Profiler trace已导出到: {trace_file}")
    
    # 打印profiler摘要
    print("\nProfiler摘要:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print("-" * 100)
    
    # 清理内存
    del device_tensors
    del host_tensors
    torch.cuda.empty_cache()
    
    return prof, elapsed_time 


def multi_stream_sleep_us_profiling():
    """在4个不同的stream上分别执行sleep_us操作，并使用profiler记录trace"""
    # 获取async engine数量
    async_engine_count = get_async_engine_count_cuda()
    print(f"async engine数量: {async_engine_count}")
    
    # 初始化4个不同的stream
    num_streams = 32
    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    sleep_us_times = [100000] * num_streams

    # 使用torch profiler进行性能分析
    trace_file = "async_engine_trace_sleep_us.json"
    
    # 对整个执行过程进行计时
    start_time = time.time()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function("multi_stream_sleep_us"):
            for i, (stream, sleep_us_time) in enumerate(zip(streams, sleep_us_times)):
                with torch.profiler.record_function(f"stream_{i}_sleep_us"):
                    ops.sleep_us(sleep_us_time, stream)
        
        with torch.profiler.record_function("synchronize_all_streams"):
            for stream in streams:
                stream.synchronize()
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 输出结果
    print(f"执行完成！")
    print(f"总耗时: {elapsed_time * 1000:.2f} ms")
    print("-" * 100)

    # 导出trace为JSON文件
    prof.export_chrome_trace(trace_file)
    print(f"Profiler trace已导出到: {trace_file}")
    
    # 打印profiler摘要
    print("\nProfiler摘要:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print("-" * 100)
    
if __name__ == "__main__":
    # # 运行多stream内存操作性能分析
    # multi_stream_memory_ops_profiling()
    # 运行多streams的sleep_us操作性能分析
    multi_stream_sleep_us_profiling()

