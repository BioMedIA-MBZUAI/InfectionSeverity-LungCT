"""
Author: Ibrahim Almakky
Date: 11/04/2021
"""
import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from experiments import experiment
import data.transforms as dyn_transforms


class DatasetTemplate(Dataset):
    def __init__(self, params: experiment.Hyperparams) -> None:
        self.params = params
        self.train_tansforms = dyn_transforms.ComposeDictTransforms(
            self.params.get_train_transforms()
        )


class DatasetCacher:
    def __init__(
        self,
        name,
        data_path,
        cache_path,
        img_reader,
        cache_preprocessor,
    ) -> None:
        self.name = name
        self.data_path = data_path
        self.cache_path = cache_path
        self.img_reader = img_reader
        self.cach_preprocessor = cache_preprocessor
        self.json_cache_file = os.path.join(self.cache_path, self.name + ".json")
        self.dataset_dict = None

    def add_datadict(self, dataset_dict: dict):
        self.dataset_dict = dataset_dict

    def cache_classification_dataset(self):
        """
        A function that takes the dictionary of images and
        stores them in a chached directory
        """
        if self.dataset_dict is None:
            raise Exception(
                """You need to load the data dict using
            the add_datadict() method before caching the dataset."""
            )

        print("Caching dataset...")
        # Make the neccessary directories
        try:
            os.mkdir(self.cache_path)
        except FileExistsError:
            pass

        cached_dataset = {}
        cached_dataset["inputs"] = []
        cached_dataset["targets"] = self.dataset_dict["targets"]
        cached_dataset["params"] = self.dataset_dict["params"]
        cached_dataset["orig_sizes"] = []
        cached_dataset["pix_dim"] = []

        for i, sample in tqdm(enumerate(self.dataset_dict["inputs"])):
            img_path = os.path.join(self.data_path, sample)

            if not os.path.exists(img_path):
                print("Warning: File not found - " + img_path)
                del cached_dataset["targets"][i]
                continue

            inp_img = self.img_reader.read(img_path)

            ct_header = self.img_reader.read_header(img_path)

            cached_dataset["pix_dim"].append(ct_header["pixdim"].tolist())
            cached_dataset["orig_sizes"].append(ct_header["dim"][1:4].tolist())

            inp_img = self.cach_preprocessor(inp_img)

            # Get the base name for image
            sample = os.path.basename(sample)

            sample = sample.replace(".nii", ".pt")
            cached_dataset["inputs"].append(sample)
            cache_file_name = os.path.join(self.cache_path, sample)
            torch.save(inp_img, cache_file_name)

        with open(self.json_cache_file, "w") as outfile:
            json.dump(cached_dataset, outfile)

    def load_cache(self):
        cache_json_file = open(self.json_cache_file)
        return json.load(cache_json_file)

    def cache_exists(self):
        """
        Check if the dataset has already been cached with the
        same params

        Todo:
            * Check if the same params match with json file.
            if not, then return false.
        """
        return os.path.exists(self.json_cache_file)
