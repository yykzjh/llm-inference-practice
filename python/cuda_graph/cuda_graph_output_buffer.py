import torch
from contextlib import contextmanager


@contextmanager
def graph_capture(pool=None, stream=None, capture_error_mode: str = "global", dump_path=None):
    g = torch.cuda.CUDAGraph()
    if dump_path is not None:
        g.enable_debug_mode()
    with torch.cuda.graph(cuda_graph=g, pool=pool, stream=stream, capture_error_mode=capture_error_mode):
        yield g
    if dump_path is not None:
        g.debug_dump(dump_path)

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

# Bind cudaDeviceGetGraphMemAttribute to query CUDA Graph memory pool usage
cudaDeviceGetGraphMemAttribute = cudart.cudaDeviceGetGraphMemAttribute
cudaDeviceGetGraphMemAttribute.argtypes = [
    ctypes.c_int,     # int device
    ctypes.c_int,     # enum cudaGraphMemAttributeType
    ctypes.c_void_p   # void* value
]
cudaDeviceGetGraphMemAttribute.restype = ctypes.c_int

# cudaGraphMemAttributeType enum values (as defined by CUDA runtime)
cudaGraphMemAttrUsedMemCurrent = 0
cudaGraphMemAttrUsedMemHigh = 1
cudaGraphMemAttrReservedMemCurrent = 2
cudaGraphMemAttrReservedMemHigh = 3

def _get_graph_mem_stats(device: int | None = None):
    if device is None:
        device = torch.cuda.current_device()
    used_curr = ctypes.c_size_t(0)
    used_high = ctypes.c_size_t(0)
    reserved_curr = ctypes.c_size_t(0)
    reserved_high = ctypes.c_size_t(0)
    rc = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemCurrent, ctypes.byref(used_curr))
    if rc != 0:
        raise RuntimeError(f"cudaDeviceGetGraphMemAttribute(UsedMemCurrent) failed with code {rc}")
    rc = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemHigh, ctypes.byref(used_high))
    if rc != 0:
        raise RuntimeError(f"cudaDeviceGetGraphMemAttribute(UsedMemHigh) failed with code {rc}")
    rc = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrReservedMemCurrent, ctypes.byref(reserved_curr))
    if rc != 0:
        raise RuntimeError(f"cudaDeviceGetGraphMemAttribute(ReservedMemCurrent) failed with code {rc}")
    rc = cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrReservedMemHigh, ctypes.byref(reserved_high))
    if rc != 0:
        raise RuntimeError(f"cudaDeviceGetGraphMemAttribute(ReservedMemHigh) failed with code {rc}")
    return used_curr.value, used_high.value, reserved_curr.value, reserved_high.value


MAX_BATCHSIZE = 128

# Placeholder input used for capture
static_a = torch.zeros((MAX_BATCHSIZE, 1024), device="cpu").pin_memory()
static_b = torch.zeros((MAX_BATCHSIZE, 1024), device="cpu").pin_memory()

def compute(batchsize):
    a = static_a[:batchsize].to("cuda", non_blocking=True)
    b = static_b[:batchsize].to("cuda", non_blocking=True)
    output = (a ** 2 + b * 2)
    return output

# Warmup before capture
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(1, MAX_BATCHSIZE + 1):
        _ = compute(i)
torch.cuda.current_stream().wait_stream(s)

def report_memory(prefix):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    print(f"{prefix} [Global]: Used: {used / 1024 / 1024:.2f} MB, Free: {free / 1024 / 1024:.2f} MB, Total: {total / 1024 / 1024:.2f} MB")

    # PyTorch 默认缓存分配器（默认内存池）统计
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    max_reserved = torch.cuda.max_memory_reserved()
    print(f"{prefix} [PyTorch Default Pool]: Allocated: {allocated / 1024 / 1024:.2f} MB, Reserved: {reserved / 1024 / 1024:.2f} MB, MaxAllocated: {max_allocated / 1024 / 1024:.2f} MB, MaxReserved: {max_reserved / 1024 / 1024:.2f} MB")

    # CUDA Graph 内存池统计（基于 CUDA 运行时）
    try:
        g_used_curr, g_used_high, g_reserved_curr, g_reserved_high = _get_graph_mem_stats()
        print(f"{prefix} [CUDA Graph Pool]: UsedCurrent: {g_used_curr / 1024 / 1024:.2f} MB, UsedHigh: {g_used_high / 1024 / 1024:.2f} MB, ReservedCurrent: {g_reserved_curr / 1024 / 1024:.2f} MB, ReservedHigh: {g_reserved_high / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"{prefix} [CUDA Graph Pool]: Stats unavailable ({e})")

# Captures the graph
# To allow capture, automatically sets a side stream as the current stream in the context
torch.cuda.empty_cache()
report_memory("Before capture")
graphs = [0] # 0 is a placeholder for 0 batchsize
output_buffers = [None]
memory_pool = None
for i in range(1, MAX_BATCHSIZE + 1, 1):
    with graph_capture(pool=memory_pool) as g:
        out = compute(i)
    graphs.append(g)
    output_buffers.append(out)
    memory_pool = g.pool()
report_memory("After capture")
# Run the graph
static_a[0:] += 1
static_b[0:] += 2
for i in range(1, len(graphs)):
    graphs[i].replay()
torch.cuda.current_stream().synchronize()
for i in range(1, len(graphs)):
    graphs[i].replay()
torch.cuda.current_stream().synchronize()
report_memory("After replay")
