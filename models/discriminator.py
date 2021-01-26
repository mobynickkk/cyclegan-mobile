import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.c0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),  # 256x256 -> 128x128
            nn.BatchNorm2d(32),
            nn.ReLU6(True)
        )

        self.inv_res0 = nn.Sequential(
            nn.Conv2d(32, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU6(True),
            nn.Conv2d(192, 192, 3, groups=192, stride=2),  # 128x128 -> 64x64
            nn.BatchNorm2d(192),
            nn.ReLU6(True),
            nn.Conv2d(192, 32, 1),
            nn.BatchNorm2d(32)
        )

        self.inv_res1 = nn.Sequential(
            nn.Conv2d(32, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU6(True),
            nn.Conv2d(192, 192, 3, groups=192, stride=2),  # 64x64 -> 32x32
            nn.BatchNorm2d(192),
            nn.ReLU6(True),
            nn.Conv2d(192, 64, 1),
            nn.BatchNorm2d(64)
        )

        self.c1 = nn.Sequential(
            nn.Conv2d(64, 384, 1),
            nn.BatchNorm2d(384),
            nn.ReLU6(True),
            nn.Conv2d(384, 384, 3, groups=384, stride=2),  # 32x32 -> 16x16
            nn.BatchNorm2d(384),
            nn.ReLU6(True),
            nn.Conv2d(384, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.c0(x)
        x = self.inv_res0(x)
        x = self.inv_res1(x)
        x = self.c1(x)
        return x
