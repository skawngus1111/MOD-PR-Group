import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

class InceptionModule(pl.LightningModule) :
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(InceptionModule, self).__init__()

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=(1, 1)),
            nn.BatchNorm2d(n1x1), nn.ReLU(inplace=True)
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=(1, 1)),
            nn.BatchNorm2d(n3x3_reduce), nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(n3x3), nn.ReLU(inplace=True)
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=(1, 1)),
            nn.BatchNorm2d(n5x5_reduce), nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(n5x5), nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(n5x5), nn.ReLU(inplace=True)
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=(1, 1)),
            nn.BatchNorm2d(pool_proj), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class GoogleNet(pl.LightningModule):
    def __init__(self, ch_in=3, n_classes=10):
        super(GoogleNet, self).__init__()

        self.prelayer = nn.Sequential(
            nn.Conv2d(ch_in, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
        )

        self.a3 = InceptionModule(192, 64, 96, 128, 16, 32, 32) # channel : 192 -> 64 + 128 + 32 +32 = 256
        self.b3 = InceptionModule(256, 128, 128, 192, 32, 96, 64) # channel : 256 -> 128 + 192 + 96 + 64 = 480

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = InceptionModule(480, 192, 96, 208, 16, 48, 64) # channel : 480 -> 192 + 208 + 48 + 64 = 512
        self.b4 = InceptionModule(512, 160, 112, 224, 24, 64, 64)  # channel : 512 -> 160 + 224 + 64 + 64 = 512
        self.c4 = InceptionModule(512, 128, 128, 256, 24, 64, 64)  # channel : 512 -> 128 + 256 + 64 + 64 = 512
        self.d4 = InceptionModule(512, 112, 144, 288, 32, 64, 64)  # channel : 512 -> 160 + 224 + 64 + 64 = 528
        self.e4 = InceptionModule(528, 256, 160, 320, 32, 128, 128)  # channel : 512 -> 160 + 224 + 64 + 64 = 832

        self.a5 = InceptionModule(832, 256, 160, 320, 32, 128, 128) # channel : 480 -> 192 + 208 + 48 + 64 = 512
        self.b5 = InceptionModule(832, 384, 192, 384, 48, 128, 128)  # channel : 512 -> 160 + 224 + 64 + 64 = 512

        # input feature map size : 8 * 8 * 1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, n_classes)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = self.prelayer(x)

        x = self.maxpool(x)

        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)

        return loss