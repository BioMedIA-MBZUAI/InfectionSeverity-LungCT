#!/usr/bin/env python3

import glob
import os
import pprint
import traceback

import matplotlib.pyplot as plt
from cycler import cycler
import click
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame

    Parameters
    ----------
    path : str
        path to tensorflow log file

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


def plot_metrics(
    exp_files: dict,
    metric: str,
    plt_axis: plt.axes,
    legend_loc="lower right",
    rng=-1,
    alpha=1,
    xlabel="Epoch",
    ylabel=None,
):
    metrics = {}
    steps = {}
    if rng == -1:
        shortest_len = 1000000
        shortest_name = None
    for exp_name, exp_file in exp_files.items():
        exp_df = tflog2pandas(exp_file)
        metrics[exp_name] = exp_df[exp_df["metric"] == metric]["value"].to_list()
        steps[exp_name] = exp_df[exp_df["metric"] == metric]["step"].to_list()
        if rng == -1:
            if len(metrics[exp_name]) < shortest_len:
                shortest_len = len(metrics[exp_name])
                shortest_name = exp_name
    for exp_name, exp_metric in metrics.items():
        if rng == -1:
            plt_axis.plot(
                steps[shortest_name],
                exp_metric[:shortest_len],
                label=exp_name,
                linewidth=1.2,
            )
        else:
            last_ind = rng if len(exp_metric) > rng else len(exp_metric)
            plt_axis.plot(
                range(last_ind),
                exp_metric[:last_ind],
                label=exp_name,
                alpha=alpha,
                linewidth=1.2,
            )
    plt_axis.set_xlabel(xlabel, size=11)
    plt_axis.set_ylabel(ylabel, rotation=90, size=11)
    plt_axis.legend(loc=legend_loc)


def plot_subplots(
    exp_files: dict,
    metrics: list,
    fig_size=[10, 6],
    rng=-1,
    alpha=1,
    xlabels=None,
    ylabels=None,
    titles=None,
    legend_loc="lower right",
):
    if xlabels is None:
        xlabels = ["Epoch" for _ in metrics]
    if ylabels is None:
        ylabels = [None for _ in metrics]
    if titles is None:
        titles = metrics
    # dcycler = cycler(color=["xkcd:blue", "c", "xkcd:azure", "r", "y", "g", "y"])
    dcycler = cycler(
        color=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )
    if len(metrics) == 1:
        fig, axs = plt.subplots(1)
        fig.set_size_inches(fig_size)
        axs.set_prop_cycle(dcycler)
        plot_metrics(
            exp_files,
            metric=metrics[0],
            plt_axis=axs,
            rng=rng,
            alpha=alpha,
            xlabel=xlabels[0],
            ylabel=ylabels[0],
            legend_loc=legend_loc,
        )
        axs.set_title(titles[0])
    else:
        fig, axs = plt.subplots(len(metrics))
        fig.set_size_inches(fig_size)
        for i, metric in enumerate(metrics):
            plot_metrics(
                exp_files,
                metric=metric,
                plt_axis=axs[i],
                rng=rng,
                alpha=alpha,
                xlabel=xlabels[i],
                ylabel=ylabels[i],
                legend_loc=legend_loc,
            )
            axs[i].set_title(titles[i])


@click.command()
@click.argument("logdir-or-logfile")
@click.option(
    "--write-pkl/--no-write-pkl", help="save to pickle file or not", default=False
)
@click.option(
    "--write-csv/--no-write-csv", help="save to csv file or not", default=True
)
@click.option("--out-dir", "-o", help="output directory", default=".")
def main(logdir_or_logfile: str, write_pkl: bool, write_csv: bool, out_dir: str):
    """This is a enhanced version of
    https://gist.github.com/ptschandl/ef67bbaa93ec67aba2cab0a7af47700b

    This script exctracts variables from all logs from tensorflow event
    files ("event*"),
    writes them to Pandas and finally stores them a csv-file or
    pickle-file including all (readable) runs of the logging directory.

    Example usage:

    # create csv file from all tensorflow logs in provided directory (.)
    # and write it to folder "./converted"
    tflogs2pandas.py . --write-csv --no-write-pkl --o converted

    # creaste csv file from tensorflow logfile only and write into
    # and write it to folder "./converted"
    tflogs2pandas.py tflog.hostname.12345 --write-csv --no-write-pkl --o converted
    """
    pp = pprint.PrettyPrinter(indent=4)
    if os.path.isdir(logdir_or_logfile):
        # Get all event* runs from logging_dir subdirectories
        event_paths = glob.glob(os.path.join(logdir_or_logfile, "event*"))
    elif os.path.isfile(logdir_or_logfile):
        event_paths = [logdir_or_logfile]
    else:
        raise ValueError(
            "input argument {} has to be a file or a directory".format(
                logdir_or_logfile
            )
        )
    # Call & append
    if event_paths:
        pp.pprint("Found tensorflow logs to process:")
        pp.pprint(event_paths)
        all_logs = many_logs2pandas(event_paths)
        pp.pprint("Head of created dataframe")
        pp.pprint(all_logs.head())

        os.makedirs(out_dir, exist_ok=True)
        if write_csv:
            print("saving to csv file")
            out_file = os.path.join(out_dir, "all_training_logs_in_one_file.csv")
            print(out_file)
            all_logs.to_csv(out_file, index=None)
        if write_pkl:
            print("saving to pickle file")
            out_file = os.path.join(out_dir, "all_training_logs_in_one_file.pkl")
            print(out_file)
            all_logs.to_pickle(out_file)
    else:
        print("No event paths have been found.")


if __name__ == "__main__":
    main()
