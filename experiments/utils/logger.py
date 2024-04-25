"""
Author: Ibrahim Almakky
Date: 19/05/2021

"""
from copy import deepcopy
import os
import json
import subprocess
import platform
import traceback
import torch
import torch.utils.tensorboard as tensorboard
from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import save_image


def count_parameters(model: torch.nn.Module):
    """
    Function copied from:
    https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

    Args:
        model ([type]): [description]

    Returns:
        [type]: [description]
    """
    # table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        # table.add_row([name, param])
        total_params += param
    # print(table)
    # print(f"Total Trainable Params: {total_params}")
    return total_params


class Logger:

    METRICS_FILE = "metrics.json"
    CONFUSION_FILE = "confusion.json"
    LOG_FILE = "log.out"
    PARAMS_FILE = "params.json"
    WARNING_FILE = "warning.json"
    ERROR_FILE = "error.json"
    INTERPRET_FILE = "interp.json"

    IMG_SAMPLE_DIR = "img_samples"
    SAMPLE_IMG_EXTENSION = ".png"

    def __init__(self, exp_id: str) -> None:
        # Init tensorboard writer
        self.writer = tensorboard.SummaryWriter(flush_secs=1, comment="_" + exp_id)
        self.log_dir = self.writer.get_logdir()

        # An empty log dict to add the output logs to
        self.metrics = {}
        self.confusion_matrices = {}
        self.val_acc = None
        self.inf_coverage = {}

        self.warning = {}

    def get_git_commid_id(self):
        try:
            git_id = subprocess.check_output(["git", "describe", "--always"]).strip()
            git_id = git_id.decode("utf-8")
        except:
            git_id = "git id not found"
        return git_id

    def get_hostname(self):
        try:
            hostname = platform.node()
        except:
            hostname = "Not found"
        return hostname

    def get_log_dir(self):
        return self.log_dir

    def save_exp_params(self, params: dict):
        # Save the experiment params to a json file in
        # the run directory
        params["code_version"] = self.get_git_commid_id()
        params["node"] = self.get_hostname()
        self.write(params, self.PARAMS_FILE)

    def log_metric(self, key, value):
        self.metrics[key] = value
        self.write(self.metrics, self.METRICS_FILE)

    def log_cont_metric(self, key, value, epoch):
        """Log continues metric recorded during a
        certain training epoch.

        Args:
            epoch ([type]): [description]
            key ([type]): [description]
            value ([type]): [description]
        """
        try:
            metric_dict = self.metrics[key]
        except KeyError:
            metric_dict = {}
            self.metrics[key] = metric_dict

        metric_dict[epoch] = value
        self.metrics[key] = metric_dict
        self.write(self.metrics, self.METRICS_FILE)

    def log_confmatrix(self, confmat, epoch):
        """Log confusion matrix recorded during a
        certain training epoch

        Args:
            epoch ([type]): [description]
            key ([type]): [description]
            matrix ([num.ndarray]): [description]
        """

        cm_list = []
        for i in confmat:
            class_list = []
            for j in i:
                class_list.append(float(j))
            cm_list.append(tuple(class_list))
            # classified_co.append(float(confmat[i][i]))
        self.confusion_matrices[epoch] = tuple(cm_list)  # tuple(classified_co)
        self.write(self.confusion_matrices, self.CONFUSION_FILE)

    def log_warning(self, key, value=None, **args):
        if value is None:
            value = traceback.format_exc()
        self.warning[key] = {"traceback": value}
        for arg_k, arg_v in args.items():
            self.warning[key][arg_k] = str(arg_v)
        self.write(self.warning, self.WARNING_FILE)

    def log_error(self, key, value=None, **args):
        if value is None:
            value = traceback.format_exc()
        error = {}
        error[key] = {"traceback": value}
        for arg_k, arg_v in args.items():
            error[key][arg_k] = str(arg_v)
        self.write(error, self.ERROR_FILE)

    def write(self, out_dict, file_name):
        """Write dictionary to a specified JSON filename.
        The file will be saved in the log directory.

        Args:
            out_dict (dict): The doctionary to output.
        """
        out_log_filename = os.path.join(self.log_dir, file_name)
        out_file = open(out_log_filename, "w")
        json.dump(out_dict, out_file)
        out_file.close()

    def log_info(self, key, value):
        out_log_filename = os.path.join(self.log_dir, self.LOG_FILE)
        out_file = open(out_log_filename, "a")
        output = str(key) + ": " + str(value) + "\n"
        out_file.write(output)
        out_file.close()

    def plot_img(self, tag: str, imgs: torch.Tensor, epoch=None):
        # reshape_img = torch.transpose(img, 1, 2)
        reshaped_img = make_grid(imgs)
        self.writer.add_image(tag, reshaped_img, global_step=epoch, dataformats="CHW")

    def plot_img_bbxs(
        self, tag: str, img: torch.Tensor, bbxs: torch.Tensor, epoch=None
    ):
        # bbxs = bbxs.type(torch.ByteTensor)
        img = img.type(torch.ByteTensor)
        # Covert image to RGB
        img = torch.stack([img, img, img])
        bbx_img = draw_bounding_boxes(img, bbxs)
        self.plot_img(tag, bbx_img, epoch)

    def save_img_sample(
        self,
        tag: str,
        img_tensor: torch.Tensor,
        epoch=None,
        bounding_boxes=None,
        labels=None,
        colours=None,
    ):
        """
        If no epoch is passed the image will files with the same tag
        will be overwritten.

        Args:
            tag (str): [description]
            img_tensor (torch.Tensor): [description]
            epoch ([type], optional): [description]. Defaults to None.
            bounding_boxes (torch.Tensor):
            colours (list): A list of
        """

        if bounding_boxes is not None:

            # if colours is not None:
            # assert len(colours) == len(bounding_boxes)
            # assert labels is not None

            # Covert image to RGB to draw the bounding boxes
            img_tensor = img_tensor * 255
            img_tensor = img_tensor.type(torch.ByteTensor)
            # Image with a channel dimension
            if len(img_tensor.shape) == 2:
                img_tensor = torch.stack([img_tensor, img_tensor, img_tensor])
            elif len(img_tensor.shape) == 3:
                if img_tensor.shape[0] == 1:
                    img_tensor = img_tensor.expand(
                        (3, img_tensor.shape[1], img_tensor.shape[2])
                    )
            elif len(img_tensor.shape) == 4:
                # TODO: process mini-batch of images
                pass

            bbx_groups = []
            # Case with a single set of bounding boxes
            if isinstance(bounding_boxes, torch.Tensor):
                if len(bounding_boxes.shape) == 2:
                    bbx_groups.append(bounding_boxes)

            # Case with different sets of bounding boxes
            if isinstance(bounding_boxes, list):
                for i in range(len(bounding_boxes)):
                    bbx_groups.append(bounding_boxes[i])

            for i, bbx_group in enumerate(bbx_groups):
                img_tensor = draw_bounding_boxes(
                    img_tensor,
                    bbx_group,
                    colors=[colours[i] for _ in range(bbx_group.shape[0])],
                )

            # Convert the image back to float
            img_tensor = (img_tensor / 255).type(torch.FloatTensor)

        samples_path = os.path.join(self.log_dir, self.IMG_SAMPLE_DIR)
        if not os.path.isdir(samples_path):
            os.mkdir(samples_path)
        if epoch is None:
            file_name = tag
        else:
            file_name = tag + "_epoch_" + str(epoch)

        file_name += self.SAMPLE_IMG_EXTENSION
        img_file_name = os.path.join(samples_path, file_name)
        save_image(img_tensor, img_file_name)

    def log_infection_coverage(
        self,
        cuboids_coverage: dict,
        epoch: int,
    ):
        self.inf_coverage[epoch] = deepcopy(cuboids_coverage)
        self.write(self.inf_coverage, self.INTERPRET_FILE)
