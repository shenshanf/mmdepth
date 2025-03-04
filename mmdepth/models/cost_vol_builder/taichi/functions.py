import torch
import torch.autograd as autograd
import taichi as ti

from .kernels import CorrCostVolKernels


class CorrCostVolFn(autograd.Function):
    """
    Correlation-based cost volume computation using Taichi acceleration.
    This function computes the matching cost between left and right feature maps
    across different disparity levels.
    """
    kernels = CorrCostVolKernels()

    @staticmethod
    def forward(ctx, feat_left, feat_right, disp_max, disp_min, step, act_cfg):
        """
        Forward pass of correlation cost volume computation.

        Args:
            ctx: Context object to store information for backward pass
            feat_left: Left feature map, shape [B, C, H, W]
            feat_right: Right feature map, shape [B, C, H, W]
            disp_max: Maximum disparity value (exclusive)
            disp_min: Minimum disparity value (inclusive)
            step: Disparity sampling step
            act_cfg: Activation configuration (not used in current implementation)

        Returns:
            cost_volume: Computed cost volume, shape [B, D, H, W]
                        where D = (disp_max - disp_min) // step
        """
        # Save tensors and parameters for backward pass
        ctx.save_for_backward(feat_left, feat_right)
        ctx.disp_max = disp_max
        ctx.disp_min = disp_min
        ctx.step = step

        # Get feature dimensions
        batch_size, channels, height, width = feat_left.shape
        # Calculate number of disparity samples
        num_samples = (disp_max - disp_min) // step

        # Initialize output cost volume
        cost_volume = feat_left.new_zeros((batch_size, num_samples, height, width))

        # Call Taichi kernel for cost volume computation
        CorrCostVolFn.kernels.forward(
            feat_left,
            feat_right,
            cost_volume,
            disp_min,
            step,
            batch_size,
            channels,
            height,
            width,
            num_samples
        )

        return cost_volume

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for correlation cost volume computation.

        Args:
            ctx: Context object containing saved tensors and parameters
            grad_output: Gradient of the loss with respect to cost volume output,
                        shape [B, D, H, W]

        Returns:
            Tuple containing:
            - grad_feat_left: Gradient w.r.t. left feature map
            - grad_feat_right: Gradient w.r.t. right feature map
            - None: Gradient w.r.t. other parameters (not needed)
        """
        # Retrieve saved tensors and parameters
        feat_left, feat_right = ctx.saved_tensors
        disp_min = ctx.disp_min
        step = ctx.step

        # Get dimensions
        batch_size, channels, height, width = feat_left.shape
        num_samples = (ctx.disp_max - ctx.disp_min) // ctx.step

        # Initialize gradient tensors
        grad_feat_left = torch.zeros_like(feat_left)
        grad_feat_right = torch.zeros_like(feat_right)

        # Call Taichi kernel for gradient computation
        CorrCostVolFn.kernels.backward(
            grad_output,
            feat_left,
            feat_right,
            grad_feat_left,
            grad_feat_right,
            disp_min,
            step,
            batch_size,
            channels,
            height,
            width,
            num_samples
        )

        # Return gradients for all inputs
        return grad_feat_left, grad_feat_right, None, None, None, None



# =======



