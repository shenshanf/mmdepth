import taichi as ti

TAICHI_ARCH = ti.cuda
ti.init(arch=TAICHI_ARCH, debug=False, kernel_profiler=True, verbose=False)


@ti.data_oriented
class CorrCostVolKernels:

    @ti.kernel
    def forward(self,
                feat_left: ti.types.ndarray(),
                feat_right: ti.types.ndarray(),
                cost_volume: ti.types.ndarray(),
                disp_min: ti.i32,
                step: ti.i32,
                batch_size: ti.i32,
                channels: ti.i32,
                height: ti.i32,
                width: ti.i32,
                num_samples: ti.i32):
        inv_channels = 1.0 / ti.cast(channels, ti.f32)
        block_size = 16
        shared_left = ti.block_local(ti.f32, (block_size, channels))
        shared_right = ti.block_local(ti.f32, (block_size, channels))

        for b, h in ti.ndrange(batch_size, height):
            for w in ti.ndrange(width):
                ti.block_sync()
                for c in ti.range(channels):
                    shared_left[w % block_size, c] = feat_left[b, c, h, w]
                    shared_right[w % block_size, c] = feat_right[b, c, h, w]
                ti.block_sync()
                for d in range(num_samples):
                    cur_disp = disp_min + d * step
                    right_w = w - cur_disp
                    if 0 <= right_w < width:
                        acc = 0.0
                        for c in ti.static(range(channels)):
                            acc += shared_left[w % block_size, c] * shared_right[right_w % block_size, c]
                        cost_volume[b, d, h, w] = acc * inv_channels
                    else:
                        cost_volume[b, d, h, w] = 0.0

    @ti.kernel
    def backward(self,
                 grad_output: ti.types.ndarray(),
                 feat_left: ti.types.ndarray(),
                 feat_right: ti.types.ndarray(),
                 grad_feat_left: ti.types.ndarray(),
                 grad_feat_right: ti.types.ndarray(),
                 disp_min: ti.i32,
                 step: ti.i32,
                 batch_size: ti.i32,
                 channels: ti.i32,
                 height: ti.i32,
                 width: ti.i32,
                 num_samples: ti.i32):
        inv_channels = 1.0 / ti.cast(channels, ti.f32)
        block_size = 16
        shared_grad_output = ti.block_local(ti.f32, (block_size, num_samples))

        for b, h in ti.ndrange(batch_size, height):
            for w in ti.ndrange(width):
                ti.block_sync()
                for d in ti.range(num_samples):
                    shared_grad_output[w % block_size, d] = grad_output[b, d, h, w]
                ti.block_sync()
                for c in ti.static(range(channels)):
                    left_grad = 0.0
                    for d in range(num_samples):
                        cur_disp = disp_min + d * step
                        right_w = w - cur_disp
                        if 0 <= right_w < width:
                            grad_scale = shared_grad_output[w % block_size, d] * inv_channels
                            right_feat = feat_right[b, c, h, right_w]
                            left_grad += grad_scale * right_feat
                            left_feat = feat_left[b, c, h, w]
                            grad_feat_right[b, c, h, right_w] += grad_scale * left_feat
                    grad_feat_left[b, c, h, w] = left_grad

