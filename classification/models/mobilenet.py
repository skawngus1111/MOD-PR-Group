import torch.nn as nn

from .basic_model import Basic_Model

def conv_bn(inp, oup, stride) :
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride) :
    return nn.Sequential(
        # Depthwise convolution
        nn.Conv2d(inp, inp, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), groups=inp, bias=False),
        nn.BatchNorm2d(inp), nn.ReLU(inplace=True),

        # Pointwise convolution
        nn.Conv2d(inp, oup, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True)
    )

def mobilenetv1(ch_in, n_classes):
    feature = nn.Sequential(
                conv_bn(ch_in, 32, 2),
                conv_dw(32, 64, 1),
                conv_dw(64, 128, 2),
                conv_dw(128, 128, 1),
                conv_dw(128, 256, 2),
                conv_dw(256, 256, 1),
                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 1024, 2),
                conv_dw(1024, 1024, 2),
                nn.AdaptiveAvgPool2d(1)
            )
    fc = nn.Linear(1024, n_classes)

    return feature, fc

class MobileNet(Basic_Model) :
    def __init__(self, model, ch_in=3, n_classes=10):
        super(MobileNet, self).__init__()

        if model == 'mobilenet-v1' :
            self.feature, self.fc = mobilenetv1(ch_in, n_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x