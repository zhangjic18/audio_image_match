import torch
import torch.nn as nn


class my_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(nn.Conv2d(60, 64, kernel_size=(3, 3), stride=(1, 2), padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d((1, 2)),  # 64 * 64

                                     nn.AdaptiveAvgPool2d((64, 64)),

                                     nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d((2, 2)),  # 16 * 16

                                     nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d((2, 2)),  # 4 * 4
                                     )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
