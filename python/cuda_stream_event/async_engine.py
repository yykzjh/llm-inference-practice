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


def example_usage():
    """示例：如何使用 sleep_us 函数"""
    print("开始使用 sleep_us 函数...")
    
    # 在 GPU 上睡眠 1000 微秒（1 毫秒）
    ops.sleep_us(1000)
    print("完成 1ms 睡眠")
    
    # 在 GPU 上睡眠 5000 微秒（5 毫秒）
    ops.sleep_us(5000)
    print("完成 5ms 睡眠")
    
    # 示例：在异步操作中使用
    device = torch.device("cuda:0")
    tensor = torch.randn(1000, 1000, device=device)
    
    # 执行一些 GPU 操作
    result = tensor @ tensor.T
    
    # 使用 sleep_us 在 GPU 上等待一段时间
    ops.sleep_us(10000)  # 10ms
    
    print("异步操作完成")

    print(result[:5, :5])
    print("-" * 100)

    # 使用 scaled_multiply 在 GPU 上乘以一个因子
    ops.scaled_multiply(result, 2.0)
    print(result[:5, :5])
    print("-" * 100)

if __name__ == "__main__":
    example_usage()

