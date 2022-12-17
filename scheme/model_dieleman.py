import torch.nn as nn
import torch


class Dieleman(nn.Module):
    def __init__(self, num_classes=1000):
        super(Dieleman, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=6, bias=True),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(32, 64, kernel_size=5, bias=True),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(64, 128, kernel_size=3, bias=True),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, bias=True),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2 * 2 * 128, 2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes ,bias=True),
        )
        # if init_weights:
        #     self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

