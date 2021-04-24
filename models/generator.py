import torch.nn as nn
from .modules import TransformerBlock


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.e0 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 6, 3, dilation=2),
            nn.BatchNorm2d(6),
            nn.ReLU(True)
        )
        self.pool0 = nn.MaxPool2d(2, return_indices=True)
        self.e1 = TransformerBlock(6, 8, 3)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.e2 = TransformerBlock(8, 12, 4)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.e3 = TransformerBlock(12, 16, 4)
        self.pool3 = nn.MaxPool2d(2, return_indices=True)

        self.bottleneck0 = TransformerBlock(16, 16)

        self.d0 = TransformerBlock(16, 12, 4)

        self.unpool0 = nn.MaxUnpool2d(2)

        self.d1 = TransformerBlock(12, 8, 4)

        self.unpool1 = nn.MaxUnpool2d(2)

        self.d2 = TransformerBlock(8, 6, 4)

        self.unpool2 = nn.MaxUnpool2d(2)

        self.d3 = nn.Conv2d(6, 3, 1)

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

        b = self.bottleneck0(x)

        d0 = self.d0(self.unpool0(b, i3) + e3)
        d1 = self.d1(self.unpool1(d0, i2) + e2)
        d2 = self.d2(self.unpool2(d1, i1) + e1)
        d3 = self.d3(self.unpool3(d2, i0) + e0)

        return d3
