# import os
from pathlib import Path
import numpy as np
from skimage.transform import warp, AffineTransform

import torch
import torch.utils.data as data
import torchvision.utils
from torchvision.transforms import Lambda, Normalize, ToTensor, Compose
import albumentations as A
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from PIL import Image
import cv2

NYUD_MEAN = [0.48056951, 0.41091299, 0.39225179]
NYUD_STD = [0.28918225, 0.29590312, 0.3093034]


# class CustomDepthDataset(data.Dataset):
#     def __init__(self, root, split="test", transform=None, limit=None):
#         self.root = root
#         self.split = split
#         self.transform = transform
#         self.limit = limit

#         if split == "test":
#             folder = Path(root), "nyu_depth_v2", "labeled", "npy")
#             self.images = np.load(os.path.join(folder, "images.npy"))
#             self.depths = np.load(os.path.join(folder, "depths.npy"))
#         else:
#             folder = os.path.join(root, "nyu_depth_v2", "npy")
#             self.file_paths = [
#                 os.path.join(folder, n) for n in sorted(os.listdir(folder))
#             ]

#     def __len__(self):
#         if hasattr(self, "images"):
#             length = len(self.images)
#         else:
#             length = len(self.file_paths)
#         if self.limit is not None:
#             length = np.minimum(self.limit, length)
#         return length

#     def __getitem__(self, index):
#         if self.split == "test" or self.debug:
#             image = self.images[index]
#             depth = self.depths[index]
#         else:
#             stacked = np.load(self.file_paths[index])
#             image = stacked[0:3]
#             depth = stacked[3:5]

#         if self.transform is not None:
#             image, depth = transform_chw(self.transform, [image, depth])

#         return image, mask, depth

#     def compute_image_mean(self):
#         return np.mean(self.images / 255, axis=(0, 2, 3))

#     def compute_image_std(self):
#         return np.std(self.images / 255, axis=(0, 2, 3))


class SimpleDepthDataset(data.Dataset):
    def __init__(self, root_dir, X):
        self.img_path = root_dir / "train_images"
        self.depth_path = root_dir / "train_depth"
        self.mask_path = root_dir / "train_masks"
        self.transform = None
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        id = self.X[idx]
        img = cv2.imread(self.img_path / f"image_{id:06}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path / f"image_{id:06}.png", cv2.IMREAD_GRAYSCALE)
        depth = np.load(self.depth_path / f"{id:06}.npy")

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug["image"])
            mask = aug["mask"]

        if self.transform is None:
            img = Image.fromarray(img)

        # t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        t = ToTensor()
        img = t(img)
        mask = torch.from_numpy(mask).long()
        return img, mask, depth


class Nutrition5k(data.Dataset):
    def __init__(
        self,
        split,
        dataset_path,
        X,
        *,
        image_size=(384, 384),
    ):

        self.split = split
        self.dataset_path = Path(dataset_path)
        self.img_path = self.dataset_path / "train_images"
        self.depth_path = self.dataset_path / "train_depth"
        self.mask_path = self.dataset_path / "train_masks"
        self.X = X

        if self.split not in ("train", "val", "test"):
            raise ValueError("Invalid split name.")

        self.augment = A.Compose(
            [
                # A.RandomCrop(480, 480, p=1.0),
                A.Flip(p=0.5),
                A.Blur(blur_limit=5, p=0.5),
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RandomGamma(p=0.5),
            ],
            additional_targets={"depth": "mask"},
        )

        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.transform = Compose(
            [
                Resize(
                    image_size[0],
                    image_size[1],
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                self.normalization,
                PrepareForNet(),
            ]
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        id = self.X[index]
        rgb_image = cv2.imread(str(self.img_path / f"image_{id:06}.jpg"))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            str(self.mask_path / f"image_{id:06}.png"), cv2.IMREAD_GRAYSCALE
        )
        depth_image = np.load(str(self.depth_path / f"{id:06}.npy"))
        # depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        inverse_depth = 1 / (depth_image + 1e-8)
        inverse_depth = inverse_depth.astype(np.float32)

        mask = (depth_image > 0).astype(int)

        if self.split == "train":
            augmented = self.augment(image=rgb_image, depth=inverse_depth, mask=mask)
            rgb_image, inverse_depth, mask = (
                augmented["image"],
                augmented["depth"],
                augmented["mask"],
            )

        rgb_image = rgb_image / 255.0
        rgb_image = rgb_image.astype(np.float32)

        transformed_image = self.transform(
            {"image": rgb_image, "depth": inverse_depth, "mask": mask}
        )

        rgb_image = transformed_image["image"]
        inverse_depth = transformed_image["depth"]
        mask = transformed_image["mask"]

        return rgb_image, inverse_depth, mask


# def get_transform(training=True, size=(256, 192), normalize=True):
#     if training:
#         transforms = [
#             Merge(),
#             RandomFlipHorizontal(),
#             RandomRotate(angle_range=(-5, 5), mode="constant"),
#             RandomCropNumpy(size=size),
#             RandomAffineZoom(scale_range=(1.0, 1.5)),
#             Split([0, 3], [3, 5]),  #
#             # Note: ToTensor maps from [0, 255] to [0, 1] while to_tensor does not
#             [RandomColor(multiplier_range=(0.8, 1.2)), None],
#         ]
#     else:
#         transforms = [
#             [BilinearResize(0.5), None],
#         ]

#     transforms.extend(
#         [
#             # Note: ToTensor maps from [0, 255] to [0, 1] while to_tensor does not
#             [ToTensor(), Lambda(to_tensor)],
#             [Normalize(mean=NYUD_MEAN, std=NYUD_STD), None] if normalize else None,
#         ]
#     )
