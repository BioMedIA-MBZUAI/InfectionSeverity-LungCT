import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def linear_value_change(array, min_value, max_value, data_type="float32"):
    # linearly cast to [min_value, max_value]
    max_original = np.max(array) + 0.000001
    min_original = np.min(array)
    assert max_value > min_value
    assert max_original > min_original
    return_array = np.array(array, data_type)
    return_array -= min_original
    return_array = (
        return_array / ((max_original - min_original) * (max_value - min_value))
        + min_value
    )
    return return_array


def image_save(picture, path, gray=False, high_resolution=False, dpi=None):
    save_dict = path[: -len(path.split("/")[-1])]
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)
    picture = linear_value_change(picture, 0, 1)
    if not gray:
        plt.cla()
        plt.axis("off")
        plt.imshow(picture)
        if dpi is not None:
            plt.savefig(path, pad_inches=0.0, bbox_inches="tight", dpi=dpi)
            return None
        if high_resolution:
            plt.savefig(path, pad_inches=0.0, bbox_inches="tight", dpi=600)
        else:
            plt.savefig(path, pad_inches=0.0, bbox_inches="tight")
    else:
        gray_img = np.zeros([np.shape(picture)[0], np.shape(picture)[1], 3], "float32")
        gray_img[:, :, 0] = picture
        gray_img[:, :, 1] = picture
        gray_img[:, :, 2] = picture
        if dpi is not None:
            plt.savefig(path, pad_inches=0.0, bbox_inches="tight", dpi=dpi)
            return None
        if high_resolution:
            plt.cla()
            plt.axis("off")
            plt.imshow(gray_img)
            plt.savefig(path, pad_inches=0.0, bbox_inches="tight", dpi=600)
        else:
            plt.cla()
            plt.imshow(gray_img)
            plt.savefig(path)
    return None


def get_heat_map(cam_map, target_shape=None):
    # input a numpy array with shape (a, b)
    min_value, max_value = np.min(cam_map), np.max(cam_map)
    cam_map = (cam_map - min_value) / (max_value + 0.00001) * 255
    cam_map = np.array(cam_map, "int32")
    if target_shape is not None:
        assert len(target_shape) == 2

        cam_map = cv2.resize(
            np.array(cam_map, "float32"), target_shape
        )  # must in float to resize
    colored_cam = cv2.normalize(
        cam_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    colored_cam = cv2.applyColorMap(colored_cam, cv2.COLORMAP_JET)

    return_image = np.zeros(np.shape(colored_cam), "int32")
    return_image[:, :, 0] = colored_cam[:, :, 2]
    return_image[:, :, 1] = colored_cam[:, :, 1]
    return_image[:, :, 2] = colored_cam[:, :, 0]

    return return_image / 255


def merge_with_heat_map(data_image, cam_map, signal_rescale=False):
    """

    :param signal_rescale: 0-1 rescale of data_image
    :param data_image: a numpy array with shape (a, b) or (a, b, 3)
    :param cam_map: a numpy array with shape (c, d)
    :return: merged image with shape (a, b, 3), in float32, min 0 max 1.0
    """
    shape_image = np.shape(data_image)
    if shape_image != np.shape(cam_map):
        heat_map = get_heat_map(cam_map, target_shape=(shape_image[0], shape_image[1]))
    else:
        heat_map = get_heat_map(cam_map, target_shape=None)
    if signal_rescale:
        min_value, max_value = np.min(data_image), np.max(data_image)
        data_image = (data_image - min_value) / (max_value + 0.00001)
    cam_map = cv2.resize(
        np.array(cam_map, "float32"), (shape_image[0], shape_image[1])
    )  # must in float to resize
    weight_map = cam_map / (np.max(cam_map) + 0.00001)
    weight_map_image = 1 - weight_map
    return_image = np.zeros((shape_image[0], shape_image[1] * 2, 3), "float32")
    if len(shape_image) == 2:
        return_image[:, 0 : shape_image[1], 0] = data_image
        return_image[:, 0 : shape_image[1], 1] = data_image
        return_image[:, 0 : shape_image[1], 2] = data_image
    else:
        return_image[:, 0 : shape_image[1], :] = data_image

    return_image[:, shape_image[1] : :, 0] = (
        weight_map_image * return_image[:, 0 : shape_image[1], 0]
        + weight_map * heat_map[:, :, 0]
    )
    return_image[:, shape_image[1] : :, 1] = (
        weight_map_image * return_image[:, 0 : shape_image[1], 1]
        + weight_map * heat_map[:, :, 1]
    )
    return_image[:, shape_image[1] : :, 2] = (
        weight_map_image * return_image[:, 0 : shape_image[1], 2]
        + weight_map * heat_map[:, :, 2]
    )
    return return_image
