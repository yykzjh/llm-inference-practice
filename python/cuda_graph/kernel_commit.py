import sys
from pathlib import Path
import threading
import time
import torch

project_root = Path(__file__).parent.parent.parent
build_ops_path = project_root / "build" / "csrc" / "ops"
sys.path.insert(0, str(build_ops_path))

num_outer_steps = 5
num_inner_steps = 1000
sleep_us_time = 100

# 导入 ops 模块
try:
    import ops  # type: ignore
except ImportError as e:
    raise ImportError(
        f"无法导入 ops 模块。请确保已编译项目：\n"
        f"  cd {project_root} && cmake -B build && cmake --build build\n"
        f"原始错误: {e}"
    )


def thread1_sync_each_sleep(stream, profiler_context):
    """第一个线程：每次执行完sleep_us后都同步stream"""
    with torch.cuda.stream(stream):
        with torch.profiler.record_function("thread1_sync_each_sleep"):
            for outer in range(num_outer_steps):
                with torch.profiler.record_function(f"thread1_outer_{outer}"):
                    for inner in range(num_inner_steps):
                        ops.sleep_us(sleep_us_time, stream)
                        stream.synchronize()


def thread2_sync_after_inner_loop(stream, profiler_context):
    """第二个线程：每次执行完内层循环后同步stream"""
    with torch.cuda.stream(stream):
        with torch.profiler.record_function("thread2_sync_after_inner_loop"):
            for outer in range(num_outer_steps):
                with torch.profiler.record_function(f"thread2_outer_{outer}"):
                    for inner in range(num_inner_steps):
                        ops.sleep_us(sleep_us_time, stream)
                    stream.synchronize()


def thread3_cuda_graph(stream, graph, profiler_context):
    """第三个线程：执行已捕获的CUDA Graph"""
    with torch.cuda.stream(stream):
        with torch.profiler.record_function("thread3_cuda_graph"):
            # 执行5次graph
            for outer in range(num_outer_steps):
                with torch.profiler.record_function(f"thread3_outer_{outer}"):
                    graph.replay()
                    stream.synchronize()


def compare_kernel_commit():
    """比较三种不同的kernel提交方式的性能"""
    # 创建三个独立的stream
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    stream3 = torch.cuda.Stream()
    
    # 先为线程3准备graph（在profiler之外完成，避免影响其他线程）
    # Warmup：先执行一次相同的操作
    with torch.cuda.stream(stream3):
        for inner in range(num_inner_steps):
            ops.sleep_us(sleep_us_time, stream3)
        stream3.synchronize()
    
    # 创建graph并捕获内层循环（在独立的环境中完成）
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph=graph, stream=stream3):
        for inner in range(num_inner_steps):
            ops.sleep_us(sleep_us_time, stream3)
    
    # 使用torch profiler进行性能分析
    trace_file = "kernel_commit_comparison.json"
    
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
        with torch.profiler.record_function("compare_kernel_commit"):
            # 创建三个线程
            thread1 = threading.Thread(
                target=thread1_sync_each_sleep,
                args=(stream1, prof)
            )
            thread2 = threading.Thread(
                target=thread2_sync_after_inner_loop,
                args=(stream2, prof)
            )
            thread3 = threading.Thread(
                target=thread3_cuda_graph,
                args=(stream3, graph, prof)
            )
            
            # 启动所有线程
            thread1.start()
            thread2.start()
            thread3.start()
            
            # 等待所有线程完成
            thread1.join()
            thread2.join()
            thread3.join()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 输出结果
    print("=" * 100)
    print("内核提交方式对比测试完成！")
    print(f"总耗时: {elapsed_time * 1000:.2f} ms")
    print("-" * 100)
    
    # 导出trace为JSON文件
    prof.export_chrome_trace(trace_file)
    print(f"Profiler trace已导出到: {trace_file}")
    
    # 打印profiler摘要
    print("\nProfiler摘要:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print("=" * 100)


if __name__ == "__main__":
    compare_kernel_commit()
