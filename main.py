"""
Author: Ibrahim Almakky
Date: 21/03/2021
"""

import argparse

from experiments import exp_dispatcher

# Experiments path
EXPERIMENTS_MODULE = "experiments"


def main():
    parser = argparse.ArgumentParser(
        description="Run the experiment(s) in the specifid path."
    )
    parser.add_argument(
        "-p",
        "--params_path",
        default="./params/",
        type=str,
        help="""
            Path to the json parameter files for the experiment(s) to be run.
        """,
    )
    parser.add_argument(
        "-D",
        "--debug",
        default=False,
        action="store_true",
        help="""
            Flag to set the experiment to debug mode.
        """,
    )
    args = parser.parse_args()

    exprs_batch = exp_dispatcher.Dispatcher(args.params_path, args.debug)
    exprs_batch.run()


if __name__ == "__main__":
    main()
