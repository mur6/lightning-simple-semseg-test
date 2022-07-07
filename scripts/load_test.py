from pathlib import Path
import numpy as np


root_dir = Path("./train")


def check_length(path):
    files = tuple(p.stem for p in (path).glob("*"))
    return len(files)


dataset_num = 800
# 000319.npy
# print(np.load('data/temp/np_save.npy'))
# assert (
#     check_length(root_dir / "train_images")
#     == check_length(root_dir / "train_depth")
#     == check_length(root_dir / "train_masks")
# )

from sklearn.model_selection import train_test_split


def split_data(dataset_num):
    data = list(range(dataset_num))
    X_trainval, X_test = train_test_split(data, test_size=0.1, random_state=19)
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)
    return X_train, X_val, X_test


X_train, X_val, X_test = split_data(dataset_num)
print("Train Size   : ", len(X_train))
print("Val Size     : ", len(X_val))
print("Test Size    : ", len(X_test))

# from torch.utils.data import Dataset, DataLoader
# import cv2


# datasets
from finetune.datasets import SimpleDepthDataset, Nutrition5k

train_set = Nutrition5k("train", root_dir, X_train)
print(len(train_set))
# val_set = Nutrition5k("val", root_dir, X_val)
# print(len(val_set))

# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
rgb_image, inverse_depth, mask = train_set[0]
