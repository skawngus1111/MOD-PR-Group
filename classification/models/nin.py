import torch.nn as nn

from .basic_model import Basic_Model

class NIN(Basic_Model) :
    def __init__(self, ch_in=3, n_classes=10):
        super(NIN, self).__init__()

        self.n_classes = n_classes

        self.features = nn.Sequential(
            nn.Conv2d(ch_in, 192, kernel_size=(5, 5), padding=(2, 2)), nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=(1, 1), padding=(0, 0)), nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=(1, 1), padding=(0, 0)), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=(2, 2), padding=(1, 1), ceil_mode=True), nn.Dropout2d(inplace=True),

            nn.Conv2d(96, 192, kernel_size=(5, 5), padding=(2, 2)), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(1, 1), padding=(0, 0)), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(1, 1), padding=(0, 0)), nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=(2, 2), padding=(1, 1), ceil_mode=True), nn.Dropout2d(inplace=True),

            nn.Conv2d(192, 192, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(1, 1), padding=(0, 0)), nn.ReLU(inplace=True),
            nn.Conv2d(192, self.n_classes, kernel_size=(1, 1), padding=(0, 0)), nn.ReLU(inplace=True),
            nn.AvgPool2d(8, stride=(1, 1), padding=(0, 0))
        )

    def forward(self, x) :
        x = self.features(x)
        x = x.view(x.size(0), self.n_classes)

        return x