import torch


def fold_features(feat, fold_dim):
    bs, ch, ht, wd = feat.size()

    # [B,C,H,W1] -> [B, C1, C2, H, W1]
    feat = feat.view(bs, fold_dim, -1, ht, wd)
    # [B, C1, C2, H, W1] -> [C1, B, H, W1, C2] -> [(C1*B*H), W1, C2]
    feat = feat.permute(1, 0, 3, 4, 2).reshape(fold_dim * bs * ht, wd, -1)
    return feat


def flatten_features(feat, flatten_dim, size):
    bs, ht, wd = size
    # [(C1*B*H), W1, C2] -> [C1, B, H, W1, C2]
    feat = feat.reshape(flatten_dim, bs, ht, wd, -1)
    # [C1, B, H, W1, C2] -> [B, C1, C2, H, W1] -> [B, C, H, W1]
    feat = feat.permute(1, 0, 4, 2, 3).reshape(bs, -1, ht, wd)
    return feat


def gather_features(src_tensor, mask, sparse=False):
    """

    Args:
        src_tensor:
        mask:
        sparse:

    Returns:

    """
    bs, ch, ht, wd = src_tensor.size()
    # [b,c,h,w] -> [c,b,h,w] -> [c, b*h*w(gather dim)]
    src_tensor = src_tensor.permute(1, 0, 2, 3).reshape(ch, -1)
    # [b,1,h,w] -> [b*h*w] -> sparse [nnz] -> [nnz, ch]
    index = mask.view(-1).nonzero().squeeze(dim=1).expand(ch, -1)
    out_tensor = torch.gather(src_tensor, dim=1, index=index, sparse_grad=False)

    #
    if not sparse:
        # [C, N]
        return out_tensor.t().contiguous()
    else:
        # note: 设置成一个[B,N,C]的稀疏张量,其中b和n是稀疏维度，c是dense维度

        # [b,1,h,w] -> [b, n=h*w] -> [nnz, 2]
        indices = mask.view(bs, -1).nonzero()
        sparse_out = torch.sparse_coo_tensor(indices=indices.t(),
                                             values=out_tensor.t().contiguous(),
                                             size=(bs, ht * wd, ch))
        return sparse_out


def scatter_features(target_tensor, src_tensor, mask, reduce='sum'):
    bs, ch, ht, wd = target_tensor.size()
    target_tensor = target_tensor.permute(1, 0, 2, 3).reshape(ch, -1)
    index = mask.view(-1).nonzero().squeeze(dim=1).expand(ch, -1)

    target_tensor.scatter_reduce_(dim=1, index=index, src=src_tensor.t(), reduce=reduce)
    target_tensor = target_tensor.view(ch, bs, ht, wd).permute(1, 0, 2, 3)
    return target_tensor.contiguous()

# def _debug():
#     """
#
#     Returns:
#
#     """
#     src_tensor = torch.tensor([
#         [
#             [
#                 [1, 2, 7, 8],
#                 [3, 4, 3, 1]
#             ],
#             [
#                 [5, 6, 2, 2],
#                 [7, 8, 9, 0]
#             ]
#         ],
#         [
#             [
#                 [-1, -2, -7, -8],
#                 [-3, -4, -3, -1]
#             ],
#             [
#                 [-5, -6, -2, -2],
#                 [-7, -8, -2, -2]
#             ]
#         ]
#     ], dtype=torch.float, requires_grad=True)
#
#     mask = torch.tensor(
#         [
#             [
#                 [False, True, True, False],
#                 [True, False, True, False]
#             ],
#             [
#                 [True, False, False, False],
#                 [False, False, True, False]
#             ]
#         ]
#     )
#
#     # 调用函数进行特征采集
#     out_tensor = gather_features(src_tensor, mask)
#     print(out_tensor)
#
#     target_tensor = scatter_features(target_tensor=torch.zeros_like(src_tensor),
#                                      src_tensor=out_tensor,
#                                      mask=mask)
#     print("src_tensor:", src_tensor)
#     print("target_tensor:", target_tensor)
#
#     src_tensor1 = src_tensor.detach().clone()
#     mask = mask.unsqueeze(dim=1).expand(-1, 2, -1, -1)
#     src_tensor1[~mask] = 0
#
#     print("check:", torch.allclose(src_tensor1, target_tensor))
#
#     target_tensor.sum().backward()
#     # out_tensor.sum().backward()
#
#
# if __name__ == "__main__":
#     _debug()
#     pass
