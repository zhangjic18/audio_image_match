import os
import torch
from torch.utils.data import Dataset


class image_classify_dataset(Dataset):
    def __init__(self, flag="train"):
        assert flag in ["train", "val", "test"]

        self.flag = flag
        if self.flag == "train":
            self.parent = "../data/image/train"
        elif self.flag == "val":
            self.parent = "../data/image/val"
        else:
            self.parent = "../data/image/test"

        self.class_number = {
            '061_foam_brick': 0,
            'green_basketball': 1,
            'salt_cylinder': 2,
            'shiny_toy_gun': 3,
            'stanley_screwdriver': 4,
            'strawberry': 5,
            'toothpaste_box': 6,
            'toy_elephant': 7,
            'whiteboard_spray': 8,
            'yellow_block': 9
        }

        self.img_path_list = []
        self.label_list = []

        for obj_dir in os.listdir(self.parent):
            obj_path = os.path.join(self.parent, obj_dir)

            self.img_path_list += [os.path.join(obj_path, item) for item in os.listdir(obj_path)]
            self.label_list += [self.class_number[obj_dir]] * len(os.listdir(obj_path))

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        label = self.label_list[index]

        img = torch.load(f=img_path)

        return img, label

    def __len__(self):
        return len(self.label_list)
