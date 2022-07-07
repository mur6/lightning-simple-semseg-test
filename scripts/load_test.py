from pathlib import Path
from typing import Optional
import numpy as np

import sys

base = str(Path("__file__").resolve().parent)
sys.path.append(base)


from sklearn.model_selection import train_test_split


def split_data(dataset_num):
    data = list(range(dataset_num))
    X_trainval, X_test = train_test_split(data, test_size=0.1, random_state=19)
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)
    return X_train, X_val, X_test


def check_length(path):
    files = tuple(p.stem for p in (path).glob("*"))
    return len(files)


# from torch.utils.data import Dataset, DataLoader
# import cv2


# datasets
from finetune.datasets import SimpleDepthDataset, Nutrition5k


# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
# rgb_image, inverse_depth, mask = train_set[0]

import torch

import flash
from flash.image import SemanticSegmentation, SemanticSegmentationData

import torch
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl


dataset_num = 800

X_train, X_val, X_test = split_data(dataset_num)
print("Train Size   : ", len(X_train))
print("Val Size     : ", len(X_val))
print("Test Size    : ", len(X_test))

# print(len(train_set), len(val_set))
# print(len(val_set))

batch_size = 4


class SemanticSegmentationData(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size: int = 32):
        print(root_dir)
        self.root_dir = root_dir
        self.num_classes = 3
        self.batch_size = batch_size
        super().__init__()
        # self.save_hyperparameters()
        self.prepare_data_per_node = False

    # def setup(self, stage):
    #     self.train_dataset = torchvision.datasets.MNIST(self.hparams.root_dir, train=True, transform=self.transform)
    #     self.test_dataset = torchvision.datasets.MNIST(self.hparams.root_dir, train=False, transform=self.transform)
    def prepare_data(self):
        # download
        pass

    def setup(self, stage: Optional[str] = None):
        self.custom_test = SimpleDepthDataset(self.root_dir, X_test)
        self.custom_train = SimpleDepthDataset(self.root_dir, X_train)
        self.custom_val = SimpleDepthDataset(self.root_dir, X_val)

    def train_dataloader(self):
        return DataLoader(self.custom_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.custom_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.custom_test, batch_size=self.batch_size)


datamodule = SemanticSegmentationData(root_dir=Path("./train").resolve(), batch_size=4)

model = SemanticSegmentation(
    backbone="mobilenetv3_large_100",
    head="fpn",
    # backbone="xception",
    # head="unetplusplus",
    # backbone="efficientnet-b2",
    # head="unet",
    num_classes=datamodule.num_classes,
)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(
    max_epochs=10,
    gpus=torch.cuda.device_count(),
)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")
trainer.save_checkpoint("models/model-b4-ep10.pt")
