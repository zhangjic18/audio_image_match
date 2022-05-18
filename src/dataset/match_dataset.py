import os.path
import random

import torch
from torch.utils.data import Dataset


class match_dataset(Dataset):
    def __init__(self):
        project_dir = "/".join(os.getcwd().split("/")[:-1])

        self.match_total_list = []
        self.label_total_list = []
        for class_name in os.listdir(os.path.join(project_dir, "data", "processed_data")):

            parent = os.path.join(project_dir, "data", "processed_data", class_name)

            image_path_list = [os.path.join(parent, item) for item in os.listdir(parent) if "image" in item]
            audio_path_list = [os.path.join(parent, item.split("/")[-1].replace("image", "audio")) for item in image_path_list]

            self.match_total_list += list(zip(image_path_list, audio_path_list))
            self.label_total_list += [0] * len(image_path_list)

            random.shuffle(audio_path_list)

            self.match_total_list += list(zip(image_path_list, audio_path_list))
            self.label_total_list += [1] * len(image_path_list)

            length = len(image_path_list)
            for i in range(-1, -length - 1, -1):  # 做一些校正
                if self.match_total_list[i] == self.match_total_list[i - length]:
                    self.label_total_list[i] = 0

        self.len = len(self.match_total_list)

    def __getitem__(self, index):
        image_path, audio_path = self.match_total_list[index]
        label = self.label_total_list[index]

        image = torch.load(image_path)
        audio = torch.load(audio_path)

        return image, audio, label

    def __len__(self):
        return self.len


if __name__ == "__main__":
    match_dataset()
