"""
Module to contain the HUST dataloader class
"""

import os
import glob
from torch.utils.data import Dataset
from torchvision.io.image import read_image
import torchvision.transforms as transforms


class DatasetInit(Dataset):
    """
    Dataset class for the HUST dataset

    """

    CLASSES_DIRS = ["nCT", "NiCT", "pCT"]
    DATA_DIR = "Original CT Scans"
    FILE_EXT = ".jpg"

    def __init__(self, path, img_size=128):
        # Initialize the dataset dictionary
        self.dataset = {}
        self.dataset["inputs"] = []
        self.dataset["targets"] = []

        self.img_size = img_size

        self.tsfrm = transforms.Compose([transforms.Resize(img_size)])

        data_path = os.path.join(path, self.DATA_DIR)

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

        inp_img = read_image(inp)

        inp_img = self.tsfrm(inp_img)

        return inp_img, target