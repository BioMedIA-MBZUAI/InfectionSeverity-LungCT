"""
Author: Ibrahim Almakky
Date: 28/03/2021

Todo:
    * Complete documentation.
    * Refactor and reduce the number of class
    attributes.
"""

import os
import importlib
import time
import traceback
import random

import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np

from experiments.utils.params import Hyperparams
from experiments.utils.logger import Logger
from experiments.utils.logger import count_parameters
from experiments.utils.terminator import TerminationChecker
from data.utils.transforms import ComposeDictTransforms

# Data initilization module
DATA_INIT_PATH = "./data"
# Models classes path
MODELS_PATH = "./modelling"

# List of supported criterions
CRITERIONS = {
    "CE": nn.CrossEntropyLoss,
    "BCE": nn.BCELoss,
}


class Experiment:
    """
    Base experiment class to be used when creating
    new experiments or when running one of the
    default experiments.
    """

    def __init__(self, params: Hyperparams):

        # Reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        self.parameters = params
        self.logger = Logger(self.parameters.get_exp_id())
        self.logger.save_exp_params(self.parameters.params)

        self.termination_checker = TerminationChecker()

        # self.termination_checker.add_termination_cnd(
        #     "val_acc",
        #     self.parameters.get_parameter(["training", "termination"], "validation"),
        #     increasing=True,
        # )

        self.train_loader, self.val_loader, self.test_loader = self.init_dataset(
            self.parameters.dataset_params
        )

        self.ngpus = self.setup_gpu(self.parameters.get_gpu_params())

        self.epoch_num = 0
        self.batch_num = 0

        self.train_loss = []
        self.batch_loss = None

        # A dict to contain the different model being trained within
        # the experiment. At the moment this enables the caching
        # of the various models
        self.models = {}
        self.optims = {}

    def setup_gpu(self, params):
        try:
            ngpus = params["ngpus"]
        except (KeyError, TypeError):
            ngpus = 0

        try:
            gpu_id = params["gpu_id"]
        except (KeyError, TypeError):
            gpu_id = 0

        # Decide which device we want to run on
        self.device = torch.device(
            "cuda:" + str(gpu_id)
            if (torch.cuda.is_available() and ngpus > 0)
            else "cpu"
        )

        return ngpus

    def init_dataset(self, params):
        """
        Initialize the dataset(s) for the experiment
        using the parameters passed through the json file.
        This can be overriden in child experiments to
        implement further customization for the loaded
        dataset.

        Args:
            params (dict): A dictionary of params for the
            dataset. This must include the following:
            1. "class": This must match with the classname
            from the data directory.
            2. "path": The absolute/relative path for the
            dataset files.
            3. "img_size": The input image size for the
            model.

        Returns:
            tuple: Returns three instances of the
            torch.utils.data.DataLoader for the training,
            validation and testing sets respectively.

        """

        dataset_class = importlib.import_module("." + params["class"], package="data")

        train_transform = ComposeDictTransforms(params["transform"])

        self.trainset = dataset_class.DatasetInit(
            self.parameters,
            params["data_path"],
            split_file=params["split_file"],
            img_size=params["img_size"],
            subset="train",
            transform=train_transform,
            cache_folder_name=params["cache_folder_name"],
        )

        self.valset = dataset_class.DatasetInit(
            self.parameters,
            params["data_path"],
            split_file=params["split_file"],
            img_size=params["img_size"],
            subset="val",
            cache_folder_name=params["cache_folder_name"],
        )
        self.testset = dataset_class.DatasetInit(
            self.parameters,
            params["data_path"],
            split_file=params["split_file"],
            img_size=params["img_size"],
            subset="test",
            cache_folder_name=params["cache_folder_name"],
        )

        sampler = self.init_sampler(self.parameters.get_sampler_params())
        if sampler is None:
            shuffle = self.parameters.training_params["shuffle"]
            if shuffle is False:
                self.logger.log_warning(
                    "Sampling",
                    """Shuffle was turned off without selecting a Sampler.
                    It is strognly recommended to use turn on Shuffle when
                    no sampler is selected.""",
                )
        else:
            shuffle = False

        def most_hust_collate(batch):
            targets, headers = [], []

            for i, (img, target, header) in enumerate(batch):
                if i == 0:
                    imgs = img.unsqueeze(0)
                else:
                    imgs = torch.cat((imgs, img.unsqueeze(0)), dim=0)
                targets.append(target)
                headers.append(header)

            return (
                imgs,
                torch.LongTensor(targets),
                headers,
            )

        train_loader = DataLoader(
            self.trainset,
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=most_hust_collate,
        )

        val_loader = DataLoader(
            self.valset,
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            shuffle=False,
            collate_fn=most_hust_collate,
        )

        test_loader = DataLoader(
            self.testset,
            batch_size=params["batch_size"],
            num_workers=params["num_workers"],
            collate_fn=most_hust_collate,
        )

        return train_loader, val_loader, test_loader

    def log_models_params(self):
        """
        Iterate through the models and the number of parameters for each
        of them as well as the total number of parameters
        """
        model_num_params = {}
        total = 0
        for name, model in self.models.items():
            current = count_parameters(model)
            model_num_params[name] = current
            total += current
        model_num_params["total"] = total
        self.logger.log_metric("number_parameters", model_num_params)

    def init_sampler(self, params: dict):
        """
        Initialize a sampler using given parameters.

        Args:
            params (dict): The params for the sampler.
            This must be a dictionary that contains the following:
            1. "class": The class of the sampler. Currently the
            following samplers are supported: ["WeightedRandomSampler"]
            2. "weights": The class weights must be provided if the
            WeightedRandomSampler has been selected.

            Example:
            "sampler": {"class": "WeightedRandomSampler",
             "weights": [0.3, 0.2, 0.2, 0.4, 0.5]}

        Raises:
            Exception: An unsupported sampler has been selected.
            Exception: Weights have not been specified for the WeightedRandomSampler.
            Exception: The class weight list provided does not have the same
            number classes as the dataset.

        Returns:
            [torch.utils.data.Sampler]: Sampler object.
        """

        try:
            sampler_class = params["class"]

            if sampler_class == "WeightedRandomSampler":

                int_targets = []
                for class_name in self.trainset.dataset["targets"]:
                    int_targets.append(self.trainset.CLASSES.index(class_name))

                mos_hust_class_weights = []
                for d_class in range(0, len(self.trainset.CLASSES)):
                    mos_hust_class_weights.append(
                        int_targets.count(d_class) / len(self.trainset)
                    )

                # Inverse proportational
                mos_hust_class_weights = [(1 / (w)) for w in mos_hust_class_weights]
                sum_w = sum(mos_hust_class_weights)
                # Normalise
                mos_hust_class_weights = [(w / sum_w) for w in mos_hust_class_weights]
                # print(mos_hust_class_weights)

                try:
                    class_weights = params["weights"]
                except KeyError as error:
                    raise Exception(
                        "Weights must be provided for the WeightedRandomSampler."
                    ) from error
                if len(class_weights) != len(self.trainset.CLASSES):
                    raise Exception(
                        "The number of weights does not match the number of classes."
                    )

                sample_weights = []
                for i in int_targets:
                    sample_weights.append(class_weights[i])

                sampler = data.WeightedRandomSampler(
                    sample_weights, len(sample_weights), replacement=True
                )
            else:
                raise Exception("Sampler " + str(sampler_class) + " is not supported.")
        except (KeyError, TypeError):

            sampler = None

        return sampler

    def init_model(self, params):

        model_name = params["class"]
        model_class_path = os.path.join(MODELS_PATH, model_name)
        model_class = importlib.import_module(model_class_path)
        return model_class
        # model = model_class()

    def init_multi_gpu_model(self, model):
        # Handle multi-gpu if desired
        if (self.device.type == "cuda") and (self.ngpus > 1):
            model = nn.DataParallel(model, list(range(self.ngpus)))

    def init_training(self, model_params):
        """
        Method to be overridden by inherting class
        """
        self.epoch_num = 0
        self.batch_num = 0
        training_params = self.parameters.training_params
        learning_rate = training_params["lr"]

        self.criterion = self.init_criterion(
            self.parameters.training_params["criterion"]
        )

        self.optim, self.scheduler = self.init_optim(
            model_params,
            learning_rate,
            training_params["optimizer"],
            training_params["scheduler"],
        )

    def init_validate(self):
        """
        Method to be overridden by inherting class
        """

    def train(self):
        self.log_models_params()
        training_params = self.parameters.training_params
        self.epochs = training_params["epochs"]
        try:
            self.run_epochs(self.epochs)
        # Allow Ctrl+c to skip to the next experiment
        except KeyboardInterrupt:
            self.logger.log_metric("ManualTermination", "True")
        except Exception as exception:
            # In case of any error during training, log the file name
            # and automatically log the traceback log.
            print(exception)
            traceback.print_exc()
            self.logger.log_error("Fatal training error", value=None)

    def run_epochs(self, num_epochs):
        self.train_loss = []
        self.val_accs = [0]
        val_freq = self.parameters.get_val_freq()
        tic = time.time()
        for epoch in range(0, num_epochs):
            self.logger.log_info("Epoch ", epoch)

            self.batch_num = 0
            self.epoch_num = epoch
            self.batch_loss = []

            self.epoch()

            iter_loss = np.mean(self.batch_loss)
            self.train_loss.append(iter_loss)

            self.logger.log_cont_metric("train_loss", self.train_loss[epoch], epoch)
            self.logger.writer.add_scalar("Loss/Train", self.train_loss[epoch], epoch)

            toc = time.time()
            avrg_epoch_time = (toc - tic) / (epoch + 1)
            # print("Average epoch time: {:.2f}s".format(avrg_epoch_time))
            self.logger.log_metric("average epoch time", avrg_epoch_time)
            if (self.epoch_num + 1) % val_freq == 0 and self.epoch_num > 0:
                val_acc, _ = self.validate()
                self.val_accs.append(val_acc)
                # self.termination_checker.update_metric("val_acc", val_acc)
                self.logger.log_cont_metric("val_acc", val_acc, epoch)
                self.cache_models(self.val_accs)
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(metrics=self.train_loss[epoch])
                else:
                    self.scheduler.step()

            if self.termination_checker.terminate():
                self.logger.log_metric("terminated", "True")
                break

    def epoch(self):
        self.run_batches()

    def run_batches(self):
        for batch_num, batch_data in enumerate(self.train_loader):
            self.batch(batch_data)
            self.batch_num = batch_num

    def batch(self, data):
        """
        Method to be overridden by inherting class
        """
        raise NotImplementedError(
            "batch() method must be implmented in the experiment."
        )

    def validate(self):
        """
        Method to be overridden by inherting class
        """
        acc, num = None, None
        return acc, num

    def init_criterion(self, params):
        try:
            name = params["class"]

            if name == "CE" or name == "BCE":
                try:
                    weights = torch.Tensor(params["weights"]).to(self.device)
                except KeyError:
                    weights = None

                try:
                    criterion = CRITERIONS[name](weight=weights, reduction="mean")
                except KeyError as error:
                    raise Exception("Specified criterion not supported.") from error
            elif name == "FOCAL":
                try:
                    criterion = CRITERIONS[name]()
                except KeyError as error:
                    raise Exception("Specified criterion not supported.") from error
        except KeyError as error:
            raise Exception("A criterion class must be specified.") from error

        return criterion

    def init_optim(
        self,
        model_params,
        lr: float,
        optim_params,
        scheduler_params,
    ):
        """
        Method to setup the optimizer and the schduler for a given model.
        The method also links the optimizer with the scheduler. Do not forget
        to class the scheduler.step() method during training to ensure that
        the schduler is working.
        """
        # Handle the opimizer
        # Get the optimizer class
        if optim_params["class"] is None:
            raise Exception(
                """No optimizer was specified. Please specify optimizer
                   class in the training JSON object."""
            )

        optim_name = optim_params["class"]

        # Initialize the optimizer depending on the class and params
        if optim_name == "SGD":
            # Check momentum
            try:
                momentum = optim_params["momentum"]
                weight_decay = optim_params["weight_decay"]
            except KeyError:
                momentum = 0
                weight_decay = 0

            exp_optim = optim.SGD(
                model_params,
                lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        elif optim_name == "ADAM":
            # Check BETA
            try:
                beta1 = optim_params["beta1"]
            except KeyError:
                beta1 = 0.9
            try:
                beta2 = optim_params["beta2"]
            except KeyError:
                beta2 = 0.999
            try:
                weight_decay = optim_params["weight_decay"]
            except KeyError:
                weight_decay = 0

            exp_optim = optim.Adam(
                model_params,
                lr,
                (beta1, beta2),
                weight_decay=weight_decay,
            )

        elif optim_name == "Adagrad":
            try:
                lr_decay = optim_params["lr_decay"]
            except KeyError:
                lr_decay = 0
            try:
                weight_decay = optim_params["weight_decay"]
            except KeyError:
                weight_decay = 0
            try:
                eps = optim_params["eps"]
            except KeyError:
                eps = 1e-10
            try:
                initial_accumulator_value = optim_params["initial_accumulator_value"]
            except KeyError:
                initial_accumulator_value = 0

            exp_optim = optim.Adagrad(
                model_params,
                lr,
                lr_decay,
                weight_decay,
                initial_accumulator_value,
                eps,
            )

        elif optim_name == "ASGD":
            try:
                lambd = optim_params["lambd"]
            except KeyError:
                lambd = 0.0001
            try:
                alpha = optim_params["alpha"]
            except KeyError:
                alpha = 0.75
            try:
                t0 = optim_params["t0"]
            except KeyError:
                t0 = 1000000.0
            try:
                weight_decay = optim_params["weight_decay"]
            except KeyError:
                weight_decay = 0

            exp_optim = optim.ASGD(model_params, lr, lambd, alpha, t0, weight_decay)

        else:
            raise Exception("Specified optimizer not supported.")

        # Get the schudler class
        try:
            # Handle the schduler
            scheduler_name = scheduler_params["class"]
        except KeyError:
            scheduler_name = None

        if scheduler_name == "StepLR":
            if scheduler_params["step_size"] is None:
                raise Exception(
                    "You can't use the StepLR scheduler without specifying a step size."
                )
            step_size = scheduler_params["step_size"]
            gamma = (
                0.01 if scheduler_params["gamma"] is None else scheduler_params["gamma"]
            )
            last_epoch = (
                -1
                if scheduler_params["last_epoch"] is None
                else scheduler_params["last_epoch"]
            )

            scheduler = optim.lr_scheduler.StepLR(
                exp_optim, step_size=step_size, gamma=gamma, last_epoch=last_epoch
            )
        elif scheduler_name == "MultiStepLR":
            try:
                milestones = scheduler_params["milestones"]
            except KeyError:
                raise Exception(
                    """Milesones list must be specified for 
                                   the MultiStepLR scheduler."""
                )
            try:
                gamma = scheduler_params["gamma"]
            except KeyError:
                raise Exception("""gamma must be specified for the scheduler. """)

            scheduler = optim.lr_scheduler.MultiStepLR(
                exp_optim,
                milestones=milestones,
                gamma=gamma,
            )

        elif scheduler_name == "ExponentialLR":
            try:
                gamma = scheduler_params["gamma"]
            except KeyError:
                raise Exception("""gamma must be specified for the scheduler.""")

            scheduler = optim.lr_scheduler.ExponentialLR(
                exp_optim,
                gamma=gamma,
            )

        elif scheduler_name == "ReduceLROnPlateau":
            try:
                mode = scheduler_params["mode"]
                factor = scheduler_params["factor"]
                patience = scheduler_params["patience"]
                min_lr = scheduler_params["min_lr"]
            except KeyError:
                raise Exception(
                    """Missing arguments for ReduceLROnPlateau scheduler."""
                )

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                exp_optim,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )

        elif scheduler_name == "None":
            scheduler = None

        else:
            raise Exception(
                str(scheduler_name) + " is not a supported scheduler class."
            )

        return exp_optim, scheduler

    def cache_models(self, val_acc, **params):
        try:
            acc_thresh = params["acc_threshold"]
        except KeyError:
            acc_thresh = 0
        out_path = self.logger.get_log_dir()
        if val_acc[-1] == np.max(val_acc) and val_acc[-1] > acc_thresh:
            for name, model in self.models.items():
                out_file = os.path.join(out_path, name + ".pt")
                torch.save(model.state_dict(), out_file)
            for name, opt_im in self.optims.items():
                out_file = os.path.join(out_path, name + ".pt")
                torch.save(opt_im.state_dict(), out_file)

    def load_cache(self, path):
        for name, model in self.models.items():
            model_path = os.path.join(path, name + ".pt")
            print("Loading from path %s" % (model_path))
            model.load_state_dict(torch.load(model_path))
        for name, opt_im in self.optims.items():
            optim_path = os.path.join(path, name + ".pt")
            print("Loading from path %s" % (optim_path))
            opt_im.load_state_dict(torch.load(optim_path))
