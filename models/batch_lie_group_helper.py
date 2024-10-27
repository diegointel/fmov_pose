# the following code is adapted from https://github.com/HannahHaensen/nerftrinsic_four/blob/main/utils/lie_group_helper.py

import torch


def vec2skew(v):
    """
    :param v:  (B, 3, ) torch tensor
    :return:   (B, 3, 3)
    """
    zero = torch.zeros(v.shape[0], 1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], dim=-1)  # (B, 3)
    skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], dim=-1)
    skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], dim=-1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)  # (B, 3, 3)
    return skew_v  # (B, 3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (B, 3, ) axis-angle, torch tensor
    :return:  (B, 3, 3)
    """
    skew_r = vec2skew(r)  # (B, 3, 3)
    norm_r = r.norm(dim=1, keepdim=True) + 1e-15  # (B, 1)
    eye = (
        torch.eye(3, dtype=torch.float32, device=r.device)
        .unsqueeze(0)
        .repeat(r.shape[0], 1, 1)
    )
    R = (
        eye
        + (torch.sin(norm_r) / norm_r)[..., None] * skew_r
        + ((1 - torch.cos(norm_r)) / norm_r**2)[..., None] * (skew_r @ skew_r)
    )
    return R


def batch_make_c2w(r, t):
    """
    :param r:  (B, 3, ) axis-angle             torch tensor
    :param t:  (B, 3, ) translation vector     torch tensor
    :return:   (B, 4, 4)
    """
    R = Exp(r)  # (B, 3, 3)
    c2w = torch.cat([R, t.unsqueeze(-1)], dim=2)  # (B, 3, 4)
    return c2w
