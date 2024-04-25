"""
Author: Ibrahim Almakky
Date: 28/11/2021
"""

import numpy as np


def per_class_acc(conf_matrix: np.ndarray):
    if isinstance(conf_matrix, list):
        conf_matrix = np.asarray(conf_matrix)
    return np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
