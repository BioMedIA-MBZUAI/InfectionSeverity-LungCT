"""
Author: Ibrahim Almakky
Date: 19/05/2021

"""

import os
import json


class Hyperparams:
    """
    Class to handle experiment(s) hyperparams.
    """

    PARAMETERS_PATH = "./params"

    def __init__(self, params_dict=None, params_file=None):
        """[summary]"""

        if params_dict is not None:
            self.params = params_dict
        elif params_file is not None:
            self.params = self.load_json(params_file)["params"]
        else:
            raise Exception("Either params dictionary or JSON file must be passed.")

        self.general_params = self.params["general"]
        self.dataset_params = self.params["dataset"]
        self.model_params = self.params["model"]
        self.training_params = self.params["training"]
        self.val_params = self.params["validation"]

    def load_dict(self, params_dict):
        self.params = params_dict

    def load_json(self, file_path):
        params_file_path = os.path.join(file_path)

        params_file = open(params_file_path)

        params_json = json.load(params_file)

        params_file.close()

        return params_json

    def get_parameter(self, branches: list, key: str):
        cbranch = self.params
        root = "params/"
        try:
            for branch in branches:
                cbranch = cbranch[branch]
                root += branch
            value = cbranch[key]
            return value
        except KeyError as k_error:
            raise KeyError("Cannot find " + key + " from " + root) from k_error

    def get_params_branch(self, name):
        return self.params[name]

    def get_exp_id(self):
        return self.get_parameter(["general"], "id")

    def get_exp_class(self):
        return self.get_parameter(["general"], "experiment")

    def get_gpu_params(self):
        try:
            return self.general_params["gpu"]
        except KeyError:
            return None

    def get_img_size(self):
        return self.get_parameter(["dataset"], "img_size")

    # Training Params
    def get_train_params(self):
        return self.training_params

    def get_num_epochs(self):
        return self.training_params["epochs"]

    def is_train_profiled(self):
        return self.training_params["profile"]

    def get_train_transforms(self):
        return self.get_parameter(["training"], "transforms")

    def get_sampler_params(self):
        try:
            return self.training_params["sampler"]
        except KeyError:
            return None

    # Validation Params
    def get_val_freq(self):
        return self.val_params["frequency"]

    def get_termination_params(self):
        try:
            return self.training_params["termination"]
        except KeyError:
            return None
