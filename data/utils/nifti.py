"""
Author: Ibrahim Almakky
Date: 11/04/2021
"""
import torch
import monai
from scipy import ndimage
from torchio import Image


class NiftiReader:
    def __init__(self) -> None:
        # Initialize the nifti reader
        self.nibable_reader = monai.data.NibabelReader()

    def read(self, img_path) -> torch.Tensor:
        inp_img = Image(path=img_path)

        return inp_img

    def read_header(self, img_path):
        ct_img = self.nibable_reader.read(img_path)

        return ct_img.header


def normalize(img_data):
    """Preprocessing to normalise the images

    Args:
        img_data ([type]): [description]

    Returns:
        [type]: [description]
    """
    min_value, max_value = -1250, 250
    torch.clip(img_data, min_value, max_value, out=img_data)
    img_data = (img_data - min_value) / (max_value - min_value)
    return img_data


def resize_volume(img, desired_size):
    """Resize across z-axis
    Refernce: https://keras.io/examples/vision/3D_image_classification/
    """
    # Set the desired depth
    desired_depth = desired_size[0]
    desired_width = desired_size[1]
    desired_height = desired_size[2]
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def fast_resize_valume(img, desired_size) -> torch.Tensor:
    """[summary]

    Args:
        img ([type]): [description]
        desired_size ([type]): [description]
    """
    desired_depth = desired_size[0]
    desired_width = desired_size[1]

    linear = torch.linspace(-1, 1, desired_width)
    linear = linear.unsqueeze(0)
    grid = linear.clone()
    for i in range(0, desired_depth):
        grid = torch.cat((grid, linear.clone()), 0)

    print(grid.shape)
