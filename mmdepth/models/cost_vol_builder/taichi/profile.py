import torch
import taichi as ti

from kernels import CorrCostVolKernels


def profile_correlation_cost_volume(batch_size=2, channels=64, height=256, width=960, max_disp=192, num_repeats=100):
    """使用 Taichi 的性能分析器测试 cost volume 的性能"""
    # 创建输入数据
    left_feature = torch.randn(batch_size, channels, height, width, device='cuda')
    right_feature = torch.randn(batch_size, channels, height, width, device='cuda')
    cost_volume = torch.zeros((batch_size, max_disp, height, width), device='cuda')

    # 获取 CorrCostVolKernels 实例
    kernels = CorrCostVolKernels()

    # 预热
    for _ in range(10):
        kernels.forward(
            left_feature, right_feature, cost_volume,
            0, 1, batch_size, channels, height, width, max_disp
        )

    ti.profiler.clear_kernel_profiler_info()  # 清除信息
    # 开始记录性能数据
    ti.sync()  # 确保GPU操作完成

    for _ in range(num_repeats):
        kernels.forward(
            left_feature, right_feature, cost_volume,
            0, 1, batch_size, channels, height, width, max_disp
        )

    ti.sync()

    # 打印性能统计信息
    ti.profiler.print_kernel_profiler_info()  # 或者用 'count' 来显示累积统计

    # 如果也想测试反向传播
    grad_output = torch.randn_like(cost_volume)
    grad_left = torch.zeros_like(left_feature)
    grad_right = torch.zeros_like(right_feature)

    # ti.clear_kernel_profile_info()
    # ti.profiler.clear_kernel_profiler()

    # 预热反向传播
    for _ in range(10):
        kernels.backward(
            grad_output, left_feature, right_feature,
            grad_left, grad_right,
            0, 1, batch_size, channels, height, width, max_disp
        )

    ti.profiler.clear_kernel_profiler_info()

    ti.sync()

    for _ in range(num_repeats):
        kernels.backward(
            grad_output, left_feature, right_feature,
            grad_left, grad_right,
            0, 1, batch_size, channels, height, width, max_disp
        )

    ti.sync()

    # 打印反向传播的性能统计
    print("\nBackward pass profiling:")
    ti.profiler.print_kernel_profiler_info()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--max-disp', type=int, default=192)
    parser.add_argument('--num-repeats', type=int, default=100)

    args = parser.parse_args()

    print(f"Running profiling with settings:")
    print(f"Batch size: {args.batch_size}")
    print(f"Channels: {args.channels}")
    print(f"Size: {args.height}x{args.width}")
    print(f"Max disparity: {args.max_disp}")
    print(f"Number of repeats: {args.num_repeats}\n")

    profile_correlation_cost_volume(
        args.batch_size,
        args.channels,
        args.height,
        args.width,
        args.max_disp,
        args.num_repeats
    )

