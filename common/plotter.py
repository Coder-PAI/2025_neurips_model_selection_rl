import itertools
import os
import parser
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from global_utils import GeneralUtils, stat_from_seq
from selection_algorithms import SelectionBaselines


class Plotter:
    def __init__(self):
        self.colors = [
            "#d0e6f4",
            "#b5e0d1",
            "#e3f2fd",
            "#f8bbd0",
            "#e6ee9c",
            "#c8e6c9",
            "#b9a3f5",
            "#ffc3a0",
            "#d5a6bd",
            "#f0e6f6",
            "#c3e0e5",
            "#f3e5f5",
            "#d1c4e9",
            "#ffadad",
            "#ffd6a5",
            "#fdffb6",
            "#caffbf",
            "#a0c4ff",
            "#e4f1fe",
        ]
        self.toolbox = GeneralUtils()
        self.selector = SelectionBaselines()

    def stat_from_loggers(self, loggers, confidence):
        metric_names = list(
            list(loggers[0][self.selector.baselines[0]].values())[0].keys()
        )
        dataset_indices = parser.args.ground_truth_for_traverse
        mean = {}
        low = {}
        high = {}
        for selection_name in self.selector.baselines:
            for dataset_index in dataset_indices:
                for metric_name in metric_names:
                    all_samples = [
                        logger[selection_name][dataset_index][metric_name]
                        for logger in loggers
                    ]
                    if selection_name not in mean:
                        mean[selection_name] = {}
                        low[selection_name] = {}
                        high[selection_name] = {}
                    if dataset_index not in mean[selection_name]:
                        mean[selection_name][dataset_index] = {}
                        low[selection_name][dataset_index] = {}
                        high[selection_name][dataset_index] = {}

                    mean[selection_name][dataset_index][metric_name] = np.mean(
                        all_samples
                    )
                    low[selection_name][dataset_index][metric_name] = np.percentile(
                        all_samples, (100 - confidence) / 2
                    )
                    high[selection_name][dataset_index][metric_name] = np.percentile(
                        all_samples, (100 + confidence) / 2
                    )
        return mean, low, high

    def plot(self, loggers, confidence=95, num_resamples=32):
        mean, low, high = self.stat_from_loggers(loggers, confidence)
        ground_truth = [
            self.toolbox.load_env(index)
            for index in parser.args.ground_truth_for_traverse
        ]
        gravities = [
            "gravity=" + str(self.toolbox.hopper_get_params(env)["gravity"][2])
            for env in ground_truth
        ]
        metric_names = list(list(mean[self.selector.baselines[0]].values())[0].keys())
        dataset_indices = parser.args.ground_truth_for_traverse

        for selection_name in self.selector.baselines:
            for metric_name in metric_names:
                metrics = [
                    mean[selection_name][i][metric_name] for i in dataset_indices
                ]
                errors_low = [
                    mean[selection_name][i][metric_name]
                    - low[selection_name][i][metric_name]
                    for i in dataset_indices
                ]
                errors_high = [
                    high[selection_name][i][metric_name]
                    - mean[selection_name][i][metric_name]
                    for i in dataset_indices
                ]
                os.makedirs(
                    f"offline_data/figures/{selection_name}/{metric_name}/",
                    exist_ok=True,
                )
                self.plot_bars(
                    selection_name,
                    gravities,
                    metrics,
                    metric_name,
                    f"offline_data/figures/{selection_name}/{metric_name}",
                    y_error=[errors_low, errors_high],
                )

        for metric_name in metric_names:
            all_samples = [
                [
                    logger[selection_name][i][metric_name]
                    for logger in loggers
                    for i in dataset_indices
                ]
                for selection_name in self.selector.baselines
            ]
            errors_low = []
            errors_high = []
            metric_means = []
            for data in all_samples:
                metric_mean, conf_low, conf_high = stat_from_seq(
                    data, confidence, num_resamples
                )
                metric_means.append(metric_mean)
                errors_low.append(metric_mean - conf_low)
                errors_high.append(conf_high - metric_mean)

            self.plot_bars(
                metric_name,
                self.selector.baselines,
                metric_means,
                metric_name,
                f"offline_data/figures/comparison/{metric_name}",
                y_error=[errors_low, errors_high],
            )
            for i in dataset_indices:
                all_samples = [
                    [logger[selection_name][i][metric_name] for logger in loggers]
                    for selection_name in self.selector.baselines
                ]
                errors_low = []
                errors_high = []
                for data in all_samples:
                    conf_mean, conf_low, conf_high = stat_from_seq(
                        data, confidence, num_resamples
                    )
                    errors_low.append(conf_mean - conf_low)
                    errors_high.append(conf_high - conf_mean)
                self.plot_bars(
                    metric_name,
                    self.selector.baselines,
                    [
                        mean[selection_name][i][metric_name]
                        for selection_name in self.selector.baselines
                    ],
                    metric_name,
                    f"offline_data/figures/comparison/{metric_name}_{i}",
                    y_error=[errors_low, errors_high],
                )

    def plot_bars(
        self,
        title: str,
        mdp_names: List[str],
        metrics: List[float],
        metric_name: str,
        figure_path: str,
        text=None,
        y_error=None,
    ):
        assert len(mdp_names) == len(metrics)
        data_size = len(mdp_names)
        x_labels = range(data_size)

        bars = plt.bar(
            x_labels,
            metrics,
            color=self.colors[:data_size],
            edgecolor="black",
        )
        plt.bar_label(bars)
        if y_error is not None:
            plt.errorbar(
                x_labels,
                metrics,
                yerr=y_error,
                fmt="o",
                linewidth=2,
                capsize=6,
                ecolor="black",
                color="black",
            )
        plt.xticks(x_labels, [str(i) for i in x_labels])
        plt.ylabel(metric_name)
        plt.legend(bars, mdp_names)
        plt.title(
            title
            + f"\n(with gamma={parser.args.gamma}, max_horizon={parser.args.rollout_max_horizon}, "
            f"num_rollout={parser.args.rollout_num_trajectories})"
        )
        text = text if text is not None else parser.text_for_comparison
        plt.figtext(0.5, -0.05, text, ha="center", fontsize=4, color="gray", wrap=True)
        plt.subplots_adjust(bottom=0.25)
        os.makedirs(figure_path, exist_ok=True)
        plt.savefig(
            os.path.join(figure_path, f"figure.pdf"),
            bbox_inches="tight",
        )
        self.toolbox.save_as_pkl(
            os.path.join(figure_path, "result"), [mdp_names, metrics]
        )
        # plt.show()
        plt.close()

    def plot_j_table_with_pi(self, policy_indices, j_table, target: int):
        plt.figure(figsize=(8, 8))
        data_size = len(j_table)
        bars = plt.bar(
            range(len(policy_indices)),
            j_table,
            color=self.colors[:data_size],
            edgecolor="black",
        )
        for bar, label in zip(bars, policy_indices):
            bar.set_label(label)
        plt.legend(bars, policy_indices, loc="upper right")
        plt.xlabel("Indices of Policies")
        plt.ylabel("J Value")
        plt.title(f"J values with different policies (under target MDP {target})")
        os.makedirs(f"offline_data/figures/j_with_pi", exist_ok=True)
        plt.savefig(
            f"offline_data/figures/j_with_pi/j_table_with_pi_{target}.pdf",
            bbox_inches="tight",
        )
        plt.close()

    def plot_j_table_with_env(self, env_indices, j_table, target: int):
        plt.figure(figsize=(8, 8))
        data_size = len(j_table)
        bars = plt.bar(
            range(len(env_indices)),
            j_table,
            color=self.colors[:data_size],
            edgecolor="black",
        )
        for bar, label in zip(bars, env_indices):
            bar.set_label(label)
        plt.legend(bars, env_indices, loc="upper right")
        plt.xlabel("Indices of Environments")
        plt.ylabel("J Value")
        plt.title(f"J values with different MDP's (under target policy {target})")
        os.makedirs(f"offline_data/figures/j_with_env", exist_ok=True)
        plt.savefig(
            f"offline_data/figures/j_with_env/j_table_with_env_{target}.pdf",
            bbox_inches="tight",
        )
        plt.close()
