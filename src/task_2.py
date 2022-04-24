import os

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from preprocess import get_img_tensor_from_sequence, audio_to_features


def get_similarity_matrix(root_path):
    model = torch.load("../checkpoints/match_net.pth")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)

    video_tensor_list, audio_tensor_list = [], []

    video_name_list = [item for item in os.listdir(root_path) if "video" in item]
    for video_dir in video_name_list:
        video_path = os.path.join(root_path, video_dir)
        video_tensor_list.append(get_img_tensor_from_sequence(video_path))

    audio_name_list = [item for item in os.listdir(root_path) if "audio" in item]
    for audio_dir in audio_name_list:
        audio_path = os.path.join(root_path, audio_dir)
        audio_tensor_list.append(audio_to_features(audio_path))

    similarity = np.zeros(len(video_tensor_list), len(audio_tensor_list))  # 相似度矩阵
    for i, video_tensor in enumerate(video_tensor_list):
        for j, audio_tensor in enumerate(audio_tensor_list):
            model.eval()

            with torch.no_grad():
                video_tensor = video_tensor.unsqueeze(0).to(device)
                audio_tensor = audio_tensor.unsqueeze(0).to(device)

                output = model(video_tensor, audio_tensor)

                probability = nn.Softmax(dim=1)(output).squeeze().cpu().tolist()

                similarity[i, j] = probability[0]

    return similarity, video_name_list, audio_name_list


def perfect_match(root_path):
    similarity, video_name_list, audio_name_list = get_similarity_matrix(root_path)
    row_ind, col_ind = linear_sum_assignment(-similarity)

    results = {}

    for i in range(len(col_ind)):
        video_name = video_name_list[row_ind[i]]

        results[audio_name_list[col_ind[i]]] = int(video_name.split("_")[-1])

    return results


if __name__ == "__main__":
    results = perfect_match(root_path="../data/original_data/task2/test/0/")
    print(results)
