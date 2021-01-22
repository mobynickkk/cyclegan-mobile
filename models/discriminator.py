import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.c0 = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU6(True)
        )

        self.inv_res0 = nn.Sequential(
            nn.Conv2d(16, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU6(True),
            nn.Conv2d(96, 96, 3, groups=96, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(96),
            nn.ReLU6(True),
            nn.Conv2d(96, 16, 1),
            nn.BatchNorm2d(16)
        )

        self.inv_res1 = nn.Sequential(
            nn.Conv2d(16, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU6(True),
            nn.Conv2d(96, 96, 3, groups=96, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(96),
            nn.ReLU6(True),
            nn.Conv2d(96, 32, 1),
            nn.BatchNorm2d(32)
        )

        self.c1 = nn.Sequential(
            nn.Conv2d(32, 196, 1),
            nn.BatchNorm2d(196),
            nn.ReLU6(True),
            nn.Conv2d(196, 196, 3, groups=196, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(196),
            nn.ReLU6(True),
            nn.Conv2d(196, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.c0(x)
        x = self.inv_res0(x) + x
        x = self.inv_res1(x)
        x = self.c1(x)
        return x
