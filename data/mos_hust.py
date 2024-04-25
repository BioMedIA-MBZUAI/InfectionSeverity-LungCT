"""
Author: Ibrahim Almakky
Date: 31/03/2021

"""
import os
import json
import monai
import torch
import torchvision.transforms as transforms
import torchio.transforms

from data.dataset import DatasetCacher, DatasetTemplate
from data.utils import nifti
from experiments import experiment


class DatasetInit(DatasetTemplate):
    """
    Dataset class for the combinations of the MosMed and
    HUST datasets.
    """

    NAME = "MosHUST"
    CLASSES = {
        "5-Class": ["Control", "Mild", "Moderate", "Severe", "Critical"],
        "3-Class": ["Control", "Mild or Moderate", "Severe or Critical"],
    }
    SPLIT_PATH = os.path.join("params", "data_splits")

    MODES = ["5-Class", "3-Class"]

    CACHE_FOLDER = "./cache"

    def __init__(
        self,
        params: experiment.Hyperparams,
        path,
        split_file="split_mosmed_hust_nosuspect_80_20.json",
        img_size=128,
        subset="train",
        transform=None,
        cache_folder_name=None,
    ):

        super().__init__(params)

        # Validate the img size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        elif isinstance(img_size, list):
            img_size = tuple(img_size)
            if len(img_size) == 1:
                img_size = img_size * 2

        if cache_folder_name is not None:
            self.NAME = cache_folder_name

        # Determine the input dimensions
        if isinstance(img_size, tuple):
            if len(img_size) == 3:
                self.dims = 3
            elif len(img_size) == 2:
                self.dims = 2
            else:
                raise Exception("Invalid img size dimensions were speicified")
        else:
            raise TypeError("Invalid data type was used for the img size")

        self.img_size = img_size
        self.transform = transform

        # Set up caching attributes
        self.subset = subset

        self.mode = params.get_parameter(["dataset"], "mode")
        assert self.mode in self.MODES

        moshust_cache_path = os.path.join(self.CACHE_FOLDER, self.NAME)
        self.cache_json_file = os.path.join(
            moshust_cache_path, self.NAME + "_" + subset + ".json"
        )

        # Initialize the dataset dictionary
        self.dataset = {}
        self.dataset["inputs"] = []
        self.dataset["targets"] = []
        self.dataset["params"] = {"path": path, "img_size": img_size, "subset": subset}
        self.dataset["orig_sizes"] = []
        self.dataset["pix_dim"] = []

        # important to be used in the experiment class
        self.num_classes = len(self.CLASSES[self.mode])

        self.load_path = moshust_cache_path

        self.prespective_trasnform = transforms.Compose(
            [transforms.RandomPerspective(distortion_scale=0.5, p=1)]
        )

        if self.dims == 2:
            img_preprocessor = self.img_preprocess
        elif self.dims == 3:
            img_preprocessor = self.img_3d_preprocess

        cacher = DatasetCacher(
            self.NAME + "_" + subset,
            path,
            moshust_cache_path,
            nifti.NiftiReader(),
            img_preprocessor,
        )

        if cacher.cache_exists():
            # Switch loading directory to pre-cached directory
            self.dataset = cacher.load_cache()
            return None
        
        if not os.path.exists(moshust_cache_path):
            os.makedirs(moshust_cache_path)

        if self.dims == 2:
            self.tsfrm = transforms.Compose([transforms.Resize(img_size)])

        # Initialize the nifti reader
        self.nibable_reader = monai.data.NibabelReader()

        # Read the data from the json file
        data_json_file_path = os.path.join(self.SPLIT_PATH, split_file)

        data_json_file = open(data_json_file_path)
        data_json = json.load(data_json_file)

        for o_class in data_json[subset]:
            self.dataset["inputs"] += data_json[subset][o_class]
            self.dataset["targets"] += [
                o_class for x in range(0, len(data_json[subset][o_class]))
            ]

        data_json_file.close()

        cacher.add_datadict(self.dataset)
        cacher.cache_classification_dataset()
        self.dataset = cacher.load_cache()
        return None

    def __len__(self):
        return len(self.dataset["targets"])

    def __getitem__(self, idx):
        inp = self.dataset["inputs"][idx]

        target = self.dataset["targets"][idx]

        img_path = os.path.join(self.load_path, inp)
        inp_img = torch.load(img_path)

        if self.subset == "train":
            if self.dims == 2:
                inp_img = self.prespective_trasnform(inp_img)

        if self.mode == self.MODES[0]:
            target = self.CLASSES[self.mode].index(target)
        elif self.mode == self.MODES[1]:
            for i, _ in enumerate(self.CLASSES[self.mode]):
                if target in self.CLASSES[self.mode][i]:
                    target = i
                    break

        if self.subset == "train":
            inp_img = self.transform(inp_img)

        img_header = {}
        img_header["file_name"] = inp
        img_header["orig_size"] = self.dataset["orig_sizes"][idx]
        img_header["pix_dim"] = self.dataset["pix_dim"][idx]

        return (inp_img, target, img_header)

    def img_preprocess(self, img_data):
        """[summary]"""
        img_data = torch.Tensor(img_data)

        # This puts the dimensions in the needed order
        img_data = img_data.transpose(0, 2)

        img_data = nifti.normalize(img_data)

        img_data = torch.sum(img_data, 0, keepdim=True)

        img_data = self.tsfrm(img_data)

        return img_data

    def img_3d_preprocess(self, img_data):
        """[summary]

        Args:
            img_data ([type]): [description]
        """
        resample_transorm = torchio.transforms.Resample(target=1)
        resize = torchio.transforms.Resize(
            target_shape=(self.img_size[1], self.img_size[2], self.img_size[0])
        )
        img_data = resample_transorm(img_data)
        img_data = resize(img_data)
        img_data = img_data.data

        # This puts the dimensions in the needed order
        img_data = img_data.transpose(1, 3)

        img_data = nifti.normalize(img_data)

        return img_data
