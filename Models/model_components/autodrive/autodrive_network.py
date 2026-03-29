import torch
import torch.nn as nn
from Models.model_components.autodrive.autodrive_backbone import AutoDriveBackbone
from Models.model_components.common_layers import Conv

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 512


def fuse_conv(conv, norm):
    """Fuse Conv2d + BatchNorm2d into a single Conv2d for inference."""
    fused_conv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class AutoDrive(nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.backbone = AutoDriveBackbone(width, depth, csp)
        # self.head = AutoDriveHead(...)  — to be wired once head is designed

    def forward(self, x):
        sppf_out, c2psa_out = self.backbone(x)
        # return self.head(sppf_out, c2psa_out)  — uncomment once head is ready
        return sppf_out, c2psa_out

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class AutoDriveNetwork:
    """
    Scale configurations mirror AutoSpeedNetwork conventions:
        width : channel widths at each stage   [in, p1, p2, p3, p4, p5]
        depth : C3K2 repetitions per stage
        csp   : [use_residual_at_shallow, use_residual_at_deep]
    """

    def __init__(self):
        self.dynamic_weighting = {
            'n': {'csp': [False, True], 'depth': [1, 1, 1, 1, 1, 1], 'width': [3, 16,  32,  64,  128, 256]},
            's': {'csp': [False, True], 'depth': [1, 1, 1, 1, 1, 1], 'width': [3, 32,  64,  128, 256, 512]},
            'm': {'csp': [True,  True], 'depth': [1, 1, 1, 1, 1, 1], 'width': [3, 64,  128, 256, 512, 512]},
            'l': {'csp': [True,  True], 'depth': [2, 2, 2, 2, 2, 2], 'width': [3, 64,  128, 256, 512, 512]},
            'x': {'csp': [True,  True], 'depth': [2, 2, 2, 2, 2, 2], 'width': [3, 96,  192, 384, 768, 768]},
        }

    def build_model(self, version: str) -> AutoDrive:
        cfg = self.dynamic_weighting[version]
        return AutoDrive(cfg['width'], cfg['depth'], cfg['csp'])
