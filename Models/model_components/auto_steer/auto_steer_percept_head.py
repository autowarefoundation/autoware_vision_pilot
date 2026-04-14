import torch
from torch import nn

from Models.model_components.common_layers import (Conv)


class AutoSteerPerceptHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        in_ch2 = int(in_ch /2)
        in_ch4 = int(in_ch /4)

        self.SiLU = nn.SiLU()
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()

        self.up = torch.nn.Upsample(scale_factor=2)
        self.c1 = Conv(in_ch2, 1, torch.nn.SiLU(), k=3, s=1, p=1)
        self.c2 = Conv(in_ch2, 1, torch.nn.SiLU(), k=3, s=1, p=1)
        self.v1 = nn.Conv2d(in_ch2, in_ch4, kernel_size=(2, 1), stride=(2, 1))
        self.v2 = nn.Conv2d(in_ch2, in_ch4, kernel_size=(2, 1), stride=(2, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.h1 = nn.Conv2d(1, 1, kernel_size=(1, 16), stride=(1, 16))
        self.h2 = nn.Conv2d(1, 1, kernel_size=(1, 16), stride=(1, 16))
        # self.h3 = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=(1, 2))
        # self.height_value = nn.Linear(1024, 2)

    def forward(self, x):
        device = x[0].device
        p2, p3 = x

        p2 = self.v1(p2)
        p2 = self.SiLU(p2)
        p3 = self.v2(p3)
        p3 = self.SiLU(p3)

        # Feature vector from backbone and neck
        features = torch.cat(tensors=[self.up(p3), p2], dim=1)

        # Lane detection
        lanes = self.c1(features)
        lanes = self.SiLU(lanes)
        lanes = self.softmax(lanes)
        B, C, H, W = lanes.shape

        row_position = torch.arange(W, dtype=torch.int, device=device)
        row_multiplier = row_position[: , None]
        row_multiplier = row_multiplier.view(1, 1, 1, W).expand(B, 1, 1, W)
        lanes = lanes * row_multiplier
        lane_value = torch.sum(lanes, dim=-1, keepdim=True)
        lane_value = lane_value / W # normalize to the range [0..1]

        # Lane height detection
        height = self.c2(features)
        height = self.SiLU(height)
        height = self.h1(height)
        height = self.SiLU(height)
        height = self.h2(height)
        height = self.SiLU(height)
        # height = self.h3(height)

        # height = torch.flatten(height)
        # height_value = self.height_value(height)
        # height_value = self.ReLU(height)
        # height_value = self.Tanh(2 * height_value)

        return lane_value, height
