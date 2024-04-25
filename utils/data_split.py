"""
Author: Ibrahim Almakky
Date: 29/03/2021
This file will read all data samples from the MosMedData and HUST datasets
afterwhich it will generate a csv/json file that contains the training,
validation and testing subsets.
"""
import os
import glob
import json
import numpy as np
from pandas_ods_reader import read_ods

PATH = "/nvme2/Datasets/COVID-19/CT/"
OUTPUT_JSON = "split_mosmed_80_20.json"

MOSMED_FOLDER = "MosMedData"
MOSMED_DATA_FOLDER = "studies"
MOSMED_CLASSES_FOLDERS = ["CT-0", "CT-1", "CT-2", "CT-3", "CT-4"]
MOSMED_EXT = ".nii"

HUST_FOLDER = "HUST"
HUST_DATA_FOLDER = "DICOM"
HUST_DETAILS = "details.ods"
HUST_EXT = ".nii"
HUST_SAMPLES_FOLDER = "NIFTI"
HUST_CLASSES = ["Control", "Suspected", "Mild", "Regular", "Severe", "Critically ill"]
HUST_COHORT_SPLIT = {"train": 1170, "val": 1521, "test": 0}

# The percentage split between training, validation and testing subsets
SPLIT = {"train": 0.8, "val": 0.2, "test": 0.0}
OUTPUT_CLASSES = {"Control", "Mild", "Moderate", "Severe", "Critical"}
CLASS_MAP = {
    "CT-0": "Control",
    "Control": "Control",
    "CT-1": "Mild",
    "Suspected": "Mild",
    "Mild": "Mild",
    "CT-2": "Moderate",
    "Regular": "Moderate",
    "CT-3": "Severe",
    "Severe": "Severe",
    "CT-4": "Critical",
    "Critically ill": "Critical",
}
RANDOM_SEED = 54


def read_mosmed():
    dataset = {}

    data_path = os.path.join(PATH, MOSMED_FOLDER)

    for class_dir in MOSMED_CLASSES_FOLDERS:
        if class_dir not in dataset:
            dataset[class_dir] = []

        class_path = os.path.join(data_path, "studies", class_dir)
        class_imgs = glob.glob(os.path.join(class_path, "*" + MOSMED_EXT))
        # Remove the path to the dataset from each sample path
        class_imgs = [x.replace(PATH, "") for x in class_imgs]
        dataset[class_dir] = dataset[class_dir] + class_imgs

    return dataset


def read_hust():
    dataset = {x: [] for x in HUST_CLASSES}

    data_path = os.path.join(PATH, HUST_FOLDER)
    details_ods_path = os.path.join(data_path, HUST_DETAILS)
    samples_path = os.path.join(HUST_FOLDER, HUST_SAMPLES_FOLDER)

    details_sheet = read_ods(details_ods_path, 1)
    for row in details_sheet.iterrows():
        # check that the CT scan is available before adding the patient
        if row[1]["CT"] != "N/A":
            to_add = os.path.join(samples_path, row[1]["Patient"] + HUST_EXT)
            to_add = to_add.replace("Patient ", "Patient_")
            dataset[row[1]["Morbidity"]].append(to_add)

    return dataset


def merge(hust, mosmed):
    """A function that takes the hust and mosmed datasets
    and combines them according to a given mapping

    Args:
        hust ([dict]): The hust dataset dictionary
        mosmed ([dict]): The MosMed dataset dictionary

    Returns:
        [dict]: The combined dataset
    """
    dataset = {x: [] for x in OUTPUT_CLASSES}

    for m_class in mosmed:
        dataset[CLASS_MAP[m_class]] += mosmed[m_class]

    for h_class in hust:
        dataset[CLASS_MAP[h_class]] += hust[h_class]

    return dataset


def split(inp_dataset, subsets):
    """A function to split a dataset into a number of subsets
        with the same split percentage for each class.

    Args:
        inp_dataset ([dict]): A dataset formatted as a dictionary
        where the keys are the classes and the values are the samples.
        subsets ([dict]): A dictionary of the subsets' names and the
        values as the fraction of samples to be allocated to it.

    Returns:
        [dict]: The split dataset where the keys are the subset names.
        The values are then the dictionaries with class names as keys
        and the values are the actual samples.
    """
    dataset = {x: {} for x in subsets}
    for m_class in inp_dataset:
        num_class_samples = len(inp_dataset[m_class])
        random_indices = np.random.permutation(num_class_samples).tolist()
        num_splits = len(subsets)
        class_split = []
        # Calculate the splits for each subset
        i = 0
        for split_name in subsets:
            if i == num_splits - 1:
                num_class_split = num_class_samples - sum(class_split)
            else:
                num_class_split = np.floor(num_class_samples * subsets[split_name])
            class_split.append(int(num_class_split))

            class_split_samples = [
                inp_dataset[m_class][random_indices[x]]
                for x in range(sum(class_split[0:i]), sum(class_split[0 : i + 1]))
            ]
            dataset[split_name][m_class] = class_split_samples
            i += 1

    return dataset


def split_cohorts(inp_dataset: dict, subsets: dict, path=None):
    dataset = {x: {} for x in subsets.keys()}
    for split_class, samples in inp_dataset.items():
        for sample in samples:
            sample_id = str.split(sample, "Patient_")[-1]
            sample_id = int(sample_id.split(".nii")[0])
            for subset, subset_max in subsets.items():
                img_path = os.path.join(path, sample)
                img_exists = os.path.isfile(img_path)
                if not img_exists:
                    print(img_path)
                if sample_id <= subset_max and img_exists:
                    try:
                        dataset[subset][split_class].append(sample)
                    except KeyError:
                        dataset[subset][split_class] = []
                        dataset[subset][split_class].append(sample)
                    break
    return dataset


def del_non_existing(inp_dataset: dict, path=None):
    for _, samples in inp_dataset.items():
        for i, sample in enumerate(samples):
            img_path = os.path.join(path, sample)
            img_exists = os.path.isfile(img_path)
            if not img_exists:
                print(img_path)
                del samples[i]


def main():
    # Set random seed for numpy
    np.random.seed(seed=RANDOM_SEED)

    mosmed_dataset = read_mosmed()
    # hust_dataset = read_hust()
    hust_dataset = {}
    # Remove Suspected class as it equivelant to
    # indeterminate case
    # del hust_dataset["Suspected"]

    merged_dataset = merge(hust_dataset, mosmed_dataset)
    # del_non_existing(merged_dataset, path=PATH)
    split_dataset = split(merged_dataset, SPLIT)

    # split_dataset = split_cohorts(
    #     merged_dataset,
    #     HUST_COHORT_SPLIT,
    #     path=PATH,
    # )

    with open(OUTPUT_JSON, "w") as outfile:
        json.dump(split_dataset, outfile)


if __name__ == "__main__":
    main()
