import os.path

import torch
import torch.nn as nn


class match_net(nn.Module):
    def __init__(self, device):
        super().__init__()

        project_dir = "/".join(os.getcwd().split("/")[:-1])

        image_classify_model = torch.load(os.path.join(project_dir, "checkpoints", "resnet18__to_image_classify.pth"),
                                          map_location=device)

        audio_classify_model = torch.load(os.path.join(project_dir, "checkpoints", "resnet18__to_audio_classify.pth"),
                                          map_location=device)

        self.image_feature = nn.Sequential(*list(image_classify_model.children())[:-2],
                                           nn.AdaptiveAvgPool2d((1, 1))
                                           )

        self.audio_feature = nn.Sequential(*list(audio_classify_model.children())[:-2],
                                           nn.AdaptiveAvgPool2d((1, 1))
                                           )

        self.linear = nn.Sequential(nn.Linear(512 * 1 * 1 + 512 * 1 * 1, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 2))

    def forward(self, image, audio):
        image = self.image_feature(image)
        image = torch.flatten(image, 1)

        audio = self.audio_feature(audio)
        audio = torch.flatten(audio, 1)

        image_audio = torch.cat([image, audio], dim=1)

        return self.linear(image_audio)
