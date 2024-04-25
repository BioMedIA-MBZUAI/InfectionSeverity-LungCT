"""
Author: Ibrahim Almakky
Date: 06/12/2021

"""
from torchio import transforms


def try_get(dict, name, var):
    try:
        return dict[name]
    except KeyError:
        return var


class RandomNoise(transforms.RandomNoise):
    def __init__(self, params: dict, **kwargs):
        mean = 0
        std = (0, 0.25)
        mean, std = try_get(params, "mean", mean), try_get(params, "std", std)

        if isinstance(std, list):
            std = tuple(std)
        if isinstance(mean, list):
            mean = tuple(mean)

        super().__init__(mean=mean, std=std, **kwargs)


class RandomFlip(transforms.RandomFlip):
    def __init__(self, params: dict, **kwargs):
        axes = 0
        prob = 0.5

        axes, prob = try_get(params, "axes", axes), try_get(params, "prob", prob)

        super().__init__(axes=axes, flip_probability=prob, **kwargs)


class RandomAffine(transforms.RandomAffine):
    def __init__(self, params, **kwargs):
        scales = 1.0
        degrees = 0
        translation = 0

        scales, degrees, translation = (
            try_get(params, "scales", scales),
            try_get(params, "degrees", degrees),
            try_get(params, "translation", translation),
        )

        print(scales)

        super().__init__(
            scales=scales,
            degrees=degrees,
            translation=translation,
            # isotropic=isotropic,
            # center=center,
            # default_pad_value=default_pad_value,
            # image_interpolation=image_interpolation,
            # check_shape=check_shape,
            **kwargs
        )


class ComposeDictTransforms:

    TRANSFORMS = {
        "RandomNoise": RandomNoise,
        "RandomFlip": RandomFlip,
        "RandomAffine": RandomAffine,
    }

    def __init__(self, transforms_strs: dict) -> None:
        self.transforms = self.form_tansforms(transforms_strs)

    def form_tansforms(self, transforms_strs: dict):
        trans_list = []
        for trans_name, params in transforms_strs.items():
            if trans_name in self.TRANSFORMS:
                trans = self.TRANSFORMS[trans_name]
                trans_list.append(trans(params))
            else:
                raise Exception(str(trans_name) + " is not a supported transofrmation.")

        trans_comp = transforms.Compose(trans_list)
        return trans_comp

    def __call__(self, inp):
        return self.transforms(inp)
