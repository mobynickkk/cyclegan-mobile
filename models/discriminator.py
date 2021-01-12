import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.c0 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU6(True)
        )

        self.inv_res0 = nn.Sequential(
            nn.Conv2d(32, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU6(True),
            nn.Conv2d(192, 192, 3, groups=192),
            nn.BatchNorm2d(192),
            nn.ReLU6(True),
            nn.Conv2d(192, 32, 1),
            nn.BatchNorm2d(32)
        )

        self.inv_res1 = nn.Sequential(
            nn.Conv2d(32, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU6(True),
            nn.Conv2d(192, 192, 3, groups=192),
            nn.BatchNorm2d(192),
            nn.ReLU6(True),
            nn.Conv2d(192, 32, 1),
            nn.BatchNorm2d(32)
        )

        self.inv_res2 = nn.Sequential(
            nn.Conv2d(32, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU6(True),
            nn.Conv2d(192, 192, 3, groups=192),
            nn.BatchNorm2d(192),
            nn.ReLU6(True),
            nn.Conv2d(192, 64, 1),
            nn.BatchNorm2d(64)
        )

        self.c1 = nn.Sequential(
            nn.Conv2d(64, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU6(True),
            nn.Conv2d(512, 512, 3, groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU6(True),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.c0(x)
        a = self.inv_res0(x)
        a = self.inv_res1(a) + x
        x = self.inv_res2(a)
        x = self.c1(x)
        return x
