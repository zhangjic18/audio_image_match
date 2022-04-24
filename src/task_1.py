import os

import torch

from preprocess import audio_to_features
from src.model.from_torch.resnet import *


def single_audio_classify(model, device, audio_path):
    audio_tensor = audio_to_features(audio_path)

    audio_tensor = audio_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(audio_tensor)
        pred = torch.argmax(output, dim=1)

    return pred.item()


def audio_classify(root_path):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    model = resnet18(num_classes=10, in_channels=8, initial_stride=(1, 4))
    model.load_state_dict(torch.load("../checkpoints/resnet18__to_audio_classify.pth"))

    model = model.to(device)

    results = {}
    for audio_name in os.listdir(root_path):
        audio_path = os.path.join(root_path, audio_name)
        pred = single_audio_classify(model, device, audio_path)
        results[audio_name] = pred

    return results


if __name__ == "__main__":
    results = audio_classify(root_path="../data/original_data/task1/test")
    print(results)
