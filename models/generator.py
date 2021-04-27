import torch.nn as nn
from .modules import MobileBlock, TransformerBlock


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.e0 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 32, 3, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.pool0 = nn.MaxPool2d(2, return_indices=True)
        self.e1 = MobileBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.e2 = MobileBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.e3 = MobileBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, return_indices=True)

        self.bottleneck0 = TransformerBlock(256)
        self.bottleneck1 = TransformerBlock(256)
        self.bottleneck2 = TransformerBlock(256)

        self.d0 = MobileBlock(256, 128)

        self.unpool0 = nn.MaxUnpool2d(2)

        self.d1 = MobileBlock(128, 64)

        self.unpool1 = nn.MaxUnpool2d(2)

        self.d2 = MobileBlock(64, 32)

        self.unpool2 = nn.MaxUnpool2d(2)

        self.d3 = MobileBlock(32, 3)

        self.unpool3 = nn.MaxUnpool2d(2)

    def forward(self, x):
        e0 = self.e0(x)
        x, i0 = self.pool0(e0)
        e1 = self.e1(x)
        x, i1 = self.pool1(e1)
        e2 = self.e2(x)
        x, i2 = self.pool2(e2)
        e3 = self.e3(x)
        x, i3 = self.pool3(e3)

        b0 = self.bottleneck0(x)
        b1 = self.bottleneck1(b0 + x)
        b2 = self.bottleneck2(b0 + b1 + x)

        d0 = self.d0(self.unpool0(b0 + b1 + b2 + x, i3) + e3)
        d1 = self.d1(self.unpool1(d0, i2) + e2)
        d2 = self.d2(self.unpool2(d1, i1) + e1)
        d3 = self.d3(self.unpool3(d2, i0) + e0)

        return d3
