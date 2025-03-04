import torch
import torch.nn as nn


class DisparityClassificationLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        初始化视差分类损失模块，使用CrossEntropyLoss
        """
        super(DisparityClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, prob_volume, sample_grid, gt_disp, valid_mask):
        """
        前向传播函数
        Args:
            prob_volume: [B,D,H,W] - 每个视差候选的概率分布
            sample_grid: [B,D,H,W] - 每个概率对应的实际视差候选值
            gt_disp: [B,1,H,W] - 真实的视差图（浮点型）
            valid_mask: [B,1,H,W] - 布尔类型，表示有效像素
        Returns:
            loss: 标量损失值
        """
        # 获取输入形状
        B, D, H, W = prob_volume.shape

        # 压缩 gt_disp 和 valid_mask 的通道维度
        gt_disp = gt_disp.squeeze(1)  # [B, H, W]
        valid_mask = valid_mask.squeeze(1).bool()  # [B, H, W]

        # 确保 sample_grid 的形状是 [B, D, H, W]
        sample_grid = sample_grid.squeeze(1)  # [B, D, H, W]

        # 计算每个视差候选与真实视差的差异，找到最接近的视差候选的索引
        diff = torch.abs(sample_grid - gt_disp.unsqueeze(1))  # [B, D, H, W]
        labels = torch.argmin(diff, dim=1)  # [B, H, W]

        # 应用有效掩码，只保留有效像素的标签
        labels = labels[valid_mask]

        # 从 prob_volume 提取有效像素的 logits
        # prob_volume 的形状是 [B, D, H, W]
        # valid_mask 的形状是 [B, H, W]
        # 我们需要根据 valid_mask 选择 logits
        # 首先将 prob_volume  reshape 为 [B, D, H*W]
        prob_volume = prob_volume.reshape(B, D, -1)
        # 将 valid_mask  reshape 为 [B, H*W]
        valid_mask_flat = valid_mask.reshape(B, -1)
        # 选择有效位置的 logits，形状是 [B, D, num_valid]
        logits = prob_volume[valid_mask_flat].view(B, D, -1)
        # 转置 logits 为 [B, num_valid, D]
        logits = logits.permute(0, 2, 1)
        # 展平 logits 为 [B*num_valid, D]
        logits = logits.reshape(-1, D)

        # 展平 labels 为 [B*num_valid]
        labels = labels.reshape(-1)

        # 计算损失
        loss = self.criterion(logits, labels)

        return loss
