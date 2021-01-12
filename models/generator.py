import torch.nn as nn


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.e0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.pool0 = nn.MaxPool2d(2, return_indices=True)
        self.e1 = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.e2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.e3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, return_indices=True)

        self.bottleneck0 = nn.Sequential(
            nn.Conv2d(512, 2560, 1),
            nn.BatchNorm2d(2560),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2560, 2560, 3, groups=2560, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(2560),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2560, 512, 1),
            nn.BatchNorm2d(512)
        )

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(512, 2560, 1),
            nn.BatchNorm2d(2560),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2560, 2560, 3, groups=2560, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(2560),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2560, 512, 1),
            nn.BatchNorm2d(512)
        )

        self.d0 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.unpool0 = nn.MaxUnpool2d(2)

        self.d1 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.unpool1 = nn.MaxUnpool2d(2)

        self.d2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.unpool2 = nn.MaxUnpool2d(2)

        self.d3 = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1, padding_mode='reflect')
        )

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
        b = self.bottleneck1(b) + b

        d0 = self.d0(self.unpool0(b, i3) + e3)
        d1 = self.d1(self.unpool1(d0, i2) + e2)
        d2 = self.d2(self.unpool2(d1, i1) + e1)
        d3 = self.d3(self.unpool3(d2, i0) + e0)

        return d3
