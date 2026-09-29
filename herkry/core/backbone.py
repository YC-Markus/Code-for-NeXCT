"""Adapted NAF-style image blocks; see licenses/NAFNet.txt.

This variant uses GELU and grouped multi-kernel convolutions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F





class LayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2






class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=1, FFN_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand
        assert dw_channel % 4 == 0, f"dw_channel ({dw_channel}) must be divisible by 4 for group=4 convolutions."
        self.conv1 = nn.Conv2d(c, dw_channel, 1)

        self.conv_3x3 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=4)

        self.conv_1x5 = nn.Conv2d(dw_channel, dw_channel, kernel_size=(5, 1), padding=(2, 0), groups=4)
        self.conv_5x1 = nn.Conv2d(dw_channel, dw_channel, kernel_size=(1, 5), padding=(0, 2), groups=4)

        self.conv_1x7 = nn.Conv2d(dw_channel, dw_channel, kernel_size=(7, 1), padding=(3, 0), groups=4)
        self.conv_7x1 = nn.Conv2d(dw_channel, dw_channel, kernel_size=(1, 7), padding=(0, 3), groups=4)


        self.conv_fusion = nn.Conv2d(3 * dw_channel, dw_channel, 1)
        self.conv3 = nn.Conv2d(dw_channel, c, 1)
        self.sg = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.sg(x)

        x_3x3 = self.conv_3x3(x)

        x_1x5 = self.conv_1x5(x)
        x_5x5 = self.conv_5x1(x_1x5)

        x_1x7 = self.conv_1x7(x)
        x_7x1 = self.conv_7x1(x_1x7)

        x_all = torch.cat([x_3x3, x_5x5, x_7x1], dim=1)
        x = self.conv_fusion(x_all) * x
        x = self.conv3(x)
        y = inp + x

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x
