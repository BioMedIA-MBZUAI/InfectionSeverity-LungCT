"""
Author: Ibrahim Almakky
Date: 12/04/2021

Todo:
    * Support more transformation functions
"""

from torchvision import transforms


class RandomRotation(transforms.RandomRotation):
    def __init__(self, params_dict: dict):
        try:
            degrees = tuple(params_dict["degrees"])
        except KeyError as k_error:
            raise Exception(
                "It is required to specified the degree range."
            ) from k_error
        super().__init__(degrees)


class RandomPerspective(transforms.RandomPerspective):
    def __init__(self, params_dict: dict):
        try:
            distortion_scale = params_dict["distortion_scale"]
            p = params_dict["p"]
        except KeyError as k_error:
            raise Exception(
                "It is required to specify the distortion_scale and p."
            ) from k_error
        super().__init__(distortion_scale=distortion_scale, p=p, fill=None)


class Resize(transforms.Resize):
    def __init__(self, params_dict: dict):
        try:
            size = tuple(params_dict["size"])
        except KeyError as k_error:
            raise Exception(
                "It is required to specify the size for the Resize transformation."
            ) from k_error
        super().__init__(size)


class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __init__(self, params_dict: dict):
        try:
            prob = float(params_dict["probability"])
        except KeyError as k_error:
            raise Exception(
                "It is required to specify the vertical flip probability."
            ) from k_error
        super().__init__(p=prob)


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, params_dict: dict):
        try:
            prob = float(params_dict["probability"])
        except KeyError as k_error:
            raise Exception(
                "It is required to specify the horizontal flip probability."
            ) from k_error
        super().__init__(p=prob)


class ComposeDictTransforms:

    JSON_TRANSFORMS = {
        "Resize": Resize,
        "RandomPerspective": RandomPerspective,
        "RandomRotation": RandomRotation,
        "RandomVerticalFlip": RandomVerticalFlip,
        "RandomHorizontalFlip": RandomHorizontalFlip,
    }

    def __init__(self, transforms_strs: dict) -> None:
        self.transforms = self.form_tansforms(transforms_strs)

    def form_tansforms(self, transforms_strs: dict) -> transforms.Compose:
        trans_list = []
        for trans_name, params in transforms_strs.items():
            if trans_name in self.JSON_TRANSFORMS:
                trans = self.JSON_TRANSFORMS[trans_name]
                trans_list.append(trans(params))
            else:
                raise Exception(str(trans_name) + " is not a supported transofrmation.")
        trans_comp = transforms.Compose(trans_list)
        return trans_comp

    def __call__(self, inp):
        return self.transforms(inp)
