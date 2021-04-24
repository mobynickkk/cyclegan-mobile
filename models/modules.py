import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, block):
        super(ResidualBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return self.block(x) + x


class SelfAttention(nn.Module):
    """ Self-attention Layer"""

    def __init__(self, in_channels, scale):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, in_channels//scale, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//scale, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, W * H)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(B, -1, W * H)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, in_channels, n_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.heads = []

        for _ in range(n_heads):
            self.heads.append(ResidualBlock(SelfAttention(in_channels, n_heads)))

        self.compress = nn.Sequential(
            nn.Conv2d(in_channels*n_heads, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], 1)
        return self.compress(x)


class MobileBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale=6):
        super(MobileBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * scale, 1),
            nn.BatchNorm2d(in_channels * scale),
            nn.ReLU6(True),
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels * scale, in_channels * scale, 3, dilation=2, groups=in_channels * scale),
            nn.BatchNorm2d(in_channels * scale),
            nn.ReLU6(True),
            nn.Conv2d(in_channels * scale, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):

    def __init__(self, in_channels, n_heads=8):
        super(TransformerBlock, self).__init__()

        self.self_attention = ResidualBlock(MultiHeadAttention(in_channels, n_heads))
        self.feed_forward = ResidualBlock(MobileBlock(in_channels, in_channels))

    def forward(self, x):
        x = self.self_attention(x)
        return self.feed_forward(x)
