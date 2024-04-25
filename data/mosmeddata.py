"""
Author: Ibrahim Almakky
Date: 29/03/2021
"""

import os
import glob
import torch
import monai
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DatasetInit(Dataset):
    """
    Dataset initializer class for the MosMedData

    """

    MASKS_DIR = "masks"
    DATA_DIR = "studies"
    CLASSES_DIRS = ["CT-0", "CT-1", "CT-2", "CT-3", "CT-4"]
    FILE_EXT = ".nii"

    def __init__(self, path, img_size=128, train=True):
        """
        Parameters
        ----------
        path: str
            The path to the directory containing the dataset
        img_size: int
            The size of images. All images will be resized to this size
            using a transformer
        """

        # Initialize the dataset dictionary
        self.dataset = {}
        self.dataset["inputs"] = []
        self.dataset["targets"] = []

        self.num_classes = len(self.CLASSES_DIRS)

        self.img_size = img_size

        self.tsfrm = transforms.Compose([transforms.Resize(img_size)])

        # Initialize the nifti reader
        self.nibable_reader = monai.data.NibabelReader()

        data_path = os.path.join(path, self.DATA_DIR)
        masks_path = os.path.join(path, self.MASKS_DIR)

        # Iterate through the dataset folder and
        # store the image names and paths
        for class_dir in self.CLASSES_DIRS:
            class_path = os.path.join(data_path, class_dir)
            class_imgs = glob.glob(os.path.join(class_path, "*" + self.FILE_EXT))
            self.dataset["inputs"] = self.dataset["inputs"] + class_imgs
            self.dataset["targets"] = self.dataset["targets"] + [
                class_dir for _ in range(0, len(class_imgs))
            ]

    def __len__(self):
        return len(self.dataset["targets"])

    def __getitem__(self, idx):
        inp = self.dataset["inputs"][idx]
        target = self.dataset["targets"][idx]

        inp_img = self.nibable_reader.read(inp)
        img_data = inp_img.get_fdata()

        img_data = torch.Tensor(img_data)

        # This puts the dimensions in the needed order
        img_data = img_data.transpose(0, 2)

        img_data = torch.sum(img_data, 0, keepdim=True)

        target = self.CLASSES_DIRS.index(target)

        # Preprocessing to normalise the images
        min_value, max_value = -1250, 250
        torch.clip(img_data, min_value, max_value, out=img_data)
        img_data = (img_data - min_value) / (max_value - min_value)

        img_data = self.tsfrm(img_data)

        return img_data, target