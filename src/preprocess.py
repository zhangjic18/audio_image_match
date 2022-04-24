import os.path

import numpy as np
import torch
import cv2
from torch import tensor
import librosa
import librosa.display
import random
import shutil


def get_img_tensor_from_sequence(parent: str) -> tensor:
    """
    函数用途：将image sequence转为对应的tensor
    参数举例：
    parent:"../data/original_data/train/061_foam_brick/1"
    返回值:60*440*440的tensor(取20张序列中间的图片用于后续的训练)
    """

    img_list = []
    for jpg_name in os.listdir(os.path.join(parent, "rgb")):
        png_name = jpg_name.split(".")[0] + ".png"

        rgb_file_path = os.path.join(parent, "rgb", jpg_name)
        mask_file_path = os.path.join(parent, "mask", png_name)

        rgb = cv2.imread(rgb_file_path)
        rgb = rgb[20:rgb.shape[0] - 20, 100:rgb.shape[1] - 100, :]
        mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.bitwise_and(src1=rgb, src2=rgb, mask=mask)

        img = img.transpose(2, 0, 1)

        img_list.append(img)

    if len(img_list) < 20:
        img_list += [img_list[-1]] * (20 - len(img_list))
    elif len(img_list) > 20:
        reduced_len = len(img_list) - 20
        img_list = img_list[reduced_len // 2:]
        img_list = img_list[:20]

    assert len(img_list) == 20

    result = np.concatenate(img_list, axis=0)

    result = (result - 4.04) / 16.63

    return torch.tensor(result, dtype=torch.float)


def audio_to_features(audio_src_path: str):
    """
    函数用途：将audio数据转为梅尔频谱，返回对应的tensor
    """

    data = np.load(audio_src_path, allow_pickle=True)

    mel_array = np.zeros((4, 64, 256))
    mfcc_array = np.zeros((4, 64, 256))
    for index in range(4):
        mel_spec = librosa.feature.melspectrogram(y=data['audio'][:, index], sr=data['audio_samplerate'], n_mels=64)
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

        mfcc = librosa.feature.mfcc(y=data['audio'][:, index], sr=data['audio_samplerate'], n_mfcc=64)

        mel_array[index, :, :] = cv2.resize(mel_spec_db, (256, 64))
        mfcc_array[index, :, :] = cv2.resize(mfcc, (256, 64))

    mel_array = (mel_array - (-77.36)) / 8.26
    mfcc_array = (mfcc_array - (-5.15)) / 73.61

    feature = np.concatenate((mel_array, mfcc_array), axis=0)

    return torch.tensor(feature, dtype=torch.float)


def preprocess():
    """
    函数用途：将image sequence和audio转为tensor存储起来
    """

    parent = "../data/original_data/train"
    dst = "../data/processed_data"

    if not os.path.exists(dst):
        os.mkdir(dst)

    for class_dir in os.listdir(parent):
        if not os.path.exists(os.path.join(dst, class_dir)):
            os.mkdir(os.path.join(dst, class_dir))

        class_dir_path = os.path.join(parent, class_dir)

        for index in os.listdir(class_dir_path):
            index_path = os.path.join(class_dir_path, index)

            print(index_path)

            sequence_img_tensor = get_img_tensor_from_sequence(index_path)
            torch.save(obj=sequence_img_tensor, f=os.path.join(dst, class_dir, "image_" + str(index) + ".pth"))

            sequence_audio_tensor = audio_to_features(os.path.join(index_path, "audio_data.pkl"))
            torch.save(obj=sequence_audio_tensor, f=os.path.join(dst, class_dir, "audio_" + str(index) + ".pth"))


def generate_audio_or_image_data(choice="audio"):
    """
    函数用途：为 audio classify 或者 image classify 产生需要的数据
    """

    parent = "../data/processed_data"
    dst = os.path.join("../data", choice)

    if not os.path.exists(dst):
        os.mkdir(dst)

    for item in ["train", "val", "test"]:
        if not os.path.exists(os.path.join(dst, item)):
            os.mkdir(os.path.join(dst, item))

    for class_dir in os.listdir(parent):

        for item in ["train", "val", "test"]:
            if not os.path.exists(os.path.join(dst, item, class_dir)):
                os.mkdir(os.path.join(dst, item, class_dir))

        class_path = os.path.join(parent, class_dir)

        data_path_list = [os.path.join(class_path, item) for item in os.listdir(class_path) if choice in item]

        for data_path in data_path_list:
            p = random.random()

            if 0 <= p < 0.8:
                shutil.copy(src=data_path, dst=os.path.join(dst, "train", class_dir))
                print(os.path.join(dst, "train", class_dir))
            elif 0.8 <= p < 0.9:
                shutil.copy(src=data_path, dst=os.path.join(dst, "val", class_dir))
                print(os.path.join(dst, "val", class_dir))
            else:
                shutil.copy(src=data_path, dst=os.path.join(dst, "test", class_dir))
                print(os.path.join(dst, "test", class_dir))


if __name__ == "__main__":
    preprocess()
    generate_audio_or_image_data(choice="audio")
    generate_audio_or_image_data(choice="image")
