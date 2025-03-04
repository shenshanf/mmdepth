import torch.nn.functional as F


def spatial_2d_maxpool_as(x, target):
    """

    Args:
        x:
        target:

    Returns:

    """
    return F.adaptive_max_pool2d(x, target.shape[-2:])


def spatial_3d_maxpool_as(x, target):
    """

    Args:
        x:
        target:

    Returns:

    """
    return F.adaptive_max_pool3d(x, target.shape[-3:])

