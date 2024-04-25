"""
Author: Ibrahim Almakky
"""

import importlib
import os
import traceback
import torch

from experiments.utils.params import Hyperparams

EXPERIMENTS_MODULE = "experiments"

ERROR_FILE = "error.txt"


class Dispatcher:
    """[summary]"""

    def __init__(self, params_file, debug=False):
        self.params_file = params_file
        self.params_path = str.split(params_file, os.path.basename(self.params_file))[0]
        print(self.params_path)
        self.debug = debug

    def run(self):

        params = Hyperparams(params_file=self.params_file)
        expr_class = params.get_exp_class()

        # Load the experiment class
        exp_class = importlib.import_module(
            "." + expr_class, package=EXPERIMENTS_MODULE
        )

        try:
            experiment = exp_class.ExpInit(params)

            try:
                experiment.train()
            except KeyboardInterrupt:
                experiment.logger.log_metric("ManualTermination", "True")
            except Exception as exception:
                # In case of any error during training, log the file name
                # and automatically log the traceback log.
                print(exception)
                traceback.print_exc()
                experiment.logger.log_error(
                    "Fatal training error", value=None, param_filename=self.params_file
                )
        except Exception as exception:
            print(exception)
            traceback.print_exc()
            print(str.split(os.path.basename(self.params_file), ".")[0])
            error_file_path = os.path.join(
                self.params_path,
                str.split(os.path.basename(self.params_file), ".")[0]
                + "_"
                + ERROR_FILE,
            )
            error_file = open(error_file_path, "w")
            error_file.write(str(traceback.format_exc()))
            error_file.write(str(exception))

        # Free GPU cache memory for the next experiment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
