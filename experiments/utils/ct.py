"""
Author: Ibrahim Almakky
Date: 01/11/2021
"""
from typing import Optional
import numpy as np
import nibabel as nib
from scipy import ndimage


def generate_3dmask(img_dims: tuple, cube_dims: tuple, cubes: list):
    assert len(img_dims) == 3
    assert len(img_dims) == len(cube_dims)
    mask = np.zeros(img_dims)

    for cube in cubes:
        mask[
            cube[0] : cube[0] + cube_dims[0],
            cube[1] : cube[1] + cube_dims[1],
            cube[2] : cube[2] + cube_dims[2],
        ] = 1

    return mask


def save_nifti(img: np.ndarray, path: str, pixdim: Optional[list] = None):
    nifti_img = nib.Nifti1Image(img, np.eye(4))
    if pixdim:
        nifti_img.header["pixdim"] = pixdim
    nib.save(nifti_img, path + ".nii.gz")


def resize_volume(img: np.ndarray, new_size: tuple):
    # Set the desired depth
    desired_depth = new_size[2]
    desired_width = new_size[0]
    desired_height = new_size[1]
    # Get current depth
    current_depth = img.shape[2]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # DEBUG
    # print(depth_factor, width_factor, height_factor)
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img
